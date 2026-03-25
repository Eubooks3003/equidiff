from typing import Dict, List
import torch
import torch.nn.functional as F
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from equi_diffpo.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from equi_diffpo.model.common.rotation_transformer import RotationTransformer
from equi_diffpo.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from equi_diffpo.common.replay_buffer import ReplayBuffer
from equi_diffpo.common.sampler import SequenceSampler, get_val_mask
from equi_diffpo.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_voxel_identity_normalizer,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
register_codecs()

class RobomimicReplayVoxelSymDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            voxel_cache_dir: str = '',
            voxel_target_size: int = 64,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            n_demo=100,
            ws_size=0.6,
            ws_x_center=0,
            ws_y_center=0,
        ):
        self.n_demo = n_demo
        self.ws_size = ws_size
        self.ws_center = np.array([ws_x_center, ws_y_center])
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        # Only load lowdim + actions into replay buffer (voxels loaded lazily from .pt)
        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + f'.{n_demo}.lowdim.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _convert_lowdim_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            abs_action=abs_action,
                            rotation_transformer=rotation_transformer,
                            n_demo=n_demo)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_lowdim_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
                n_demo=n_demo)

        voxel_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'voxel':
                voxel_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        # key_first_k only for lowdim (voxels loaded separately)
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)

        # Build episode start/end arrays for global_idx -> (demo, frame) mapping
        episode_ends_arr = replay_buffer.episode_ends[:]
        episode_starts_arr = np.concatenate([[0], episode_ends_arr[:-1]])

        # Scan sparse voxel files to find max points per frame (needed for padding)
        print(f'Scanning sparse voxel files for {n_demo} demos...')
        max_sparse_pts = 0
        for demo_id in range(n_demo):
            demo_start = episode_starts_arr[demo_id] if demo_id < len(episode_starts_arr) else 0
            demo_end = episode_ends_arr[demo_id] if demo_id < len(episode_ends_arr) else 0
            # Just check the first frame of each demo (pts count is roughly consistent)
            path = os.path.join(
                voxel_cache_dir, 'voxel',
                f'demo_{demo_id}', f'frame0_voxels.pt')
            if os.path.exists(path):
                data = torch.load(path, weights_only=False, map_location='cpu')
                max_sparse_pts = max(max_sparse_pts, data['coords'].shape[0])
        # Add 10% headroom for frames with more points
        max_sparse_pts = int(max_sparse_pts * 1.1)
        print(f'Max sparse points per frame: {max_sparse_pts}')

        self.max_sparse_pts = max_sparse_pts
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.voxel_keys = voxel_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        self.voxel_cache_dir = voxel_cache_dir
        self.voxel_target_size = voxel_target_size
        self.episode_starts_arr = episode_starts_arr
        self.episode_ends_arr = episode_ends_arr

    def _global_to_demo_frame(self, global_idx):
        """Convert global buffer index to (demo_id, frame_id)."""
        episode_idx = int(np.searchsorted(self.episode_ends_arr, global_idx, side='right'))
        frame_idx = int(global_idx - self.episode_starts_arr[episode_idx])
        return episode_idx, frame_idx

    def _load_voxel_sparse(self, demo_id, frame_id):
        """Load sparse voxel data from disk.

        Returns (coords [N, 3] int16, values [N, 3] float16, src_size int).
        """
        path = os.path.join(
            self.voxel_cache_dir, 'voxel',
            f'demo_{demo_id}', f'frame{frame_id}_voxels.pt')
        data = torch.load(path, weights_only=False, map_location='cpu')
        return data['coords'], data['values'], int(data['shape'][1])

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                raise NotImplementedError
            else:
                magnitute = max(np.max([stat['max'][:2] - self.ws_center, self.ws_center - stat['min'][:2]]), self.ws_size/2)
                stat['min'][:2] = self.ws_center - magnitute
                stat['max'][:2] = self.ws_center + magnitute
                stat['mean'][:2] = self.ws_center
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)

            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pos'):
                magnitute = max(np.max([stat['max'][:2] - self.ws_center, self.ws_center - stat['min'][:2]]), self.ws_size/2)
                stat['min'][:2] = self.ws_center - magnitute
                stat['max'][:2] = self.ws_center + magnitute
                stat['mean'][:2] = self.ws_center
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # voxels (identity — already in [0,1])
        for key in self.voxel_keys:
            normalizer[key] = get_voxel_identity_normalizer()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        # Load sparse voxel data (no dense reconstruction on CPU)
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.sampler.indices[idx]
        n_data = buffer_end_idx - buffer_start_idx

        # With n_obs_steps=1, we only need one frame's sparse data
        # Handle padding: if at episode boundary, use the first/last valid frame
        if self.n_obs_steps is not None:
            k_data = min(self.n_obs_steps, n_data)
        else:
            k_data = n_data

        # For the voxel obs, we only use T_slice (first n_obs_steps frames)
        # Just load the first frame (n_obs_steps is typically 1 for voxels)
        frame_idx_in_buffer = 0
        if sample_start_idx > 0:
            # Padding at start: use first data frame
            frame_idx_in_buffer = 0
        demo_id, frame_id = self._global_to_demo_frame(buffer_start_idx + frame_idx_in_buffer)
        coords, values, src_size = self._load_voxel_sparse(demo_id, frame_id)

        # Pad coords/values to fixed length so default collate can stack them
        max_pts = self.max_sparse_pts
        N = coords.shape[0]
        if N > max_pts:
            coords = coords[:max_pts]
            values = values[:max_pts]
            N = max_pts
        coords_padded = torch.zeros(max_pts, 3, dtype=coords.dtype)
        values_padded = torch.zeros(max_pts, 3, dtype=values.dtype)
        coords_padded[:N] = coords
        values_padded[:N] = values

        # Add temporal dim to match [T, ...] convention (T=n_obs_steps, typically 1)
        obs_dict['voxel_coords'] = coords_padded.unsqueeze(0)      # [1, max_pts, 3]
        obs_dict['voxel_values'] = values_padded.unsqueeze(0)       # [1, max_pts, 3]
        obs_dict['voxel_n_pts'] = torch.tensor([[N]], dtype=torch.long)  # [1, 1]
        obs_dict['voxel_src_size'] = torch.tensor([[src_size]], dtype=torch.long)  # [1, 1]

        torch_data = {
            'obs': dict_apply(obs_dict,
                lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions


def _convert_lowdim_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer,
        n_demo=100):
    """Load only lowdim observations and actions into replay buffer.
    Voxels are loaded lazily from .pt cache files in __getitem__.
    """
    lowdim_keys = list()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        if type == 'low_dim':
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        n_demo = min(n_demo, len(demos))
        for i in range(n_demo):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        _ = meta_group.array('episode_ends', episode_ends,
            dtype=np.int64, compressor=None, overwrite=True)

        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(n_demo):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer
                )
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
