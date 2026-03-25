"""
Profile one training iteration to find the bottleneck.
Run: python profile_training.py --config-name=train_equi_diffusion_unet_voxel128_abs \
    task_name=mug_cleanup_d0 n_demo=200 \
    dataset_path=/path/to/mug_cleanup_d0.hdf5 \
    voxel_cache_dir=/path/to/mug_cleanup_d0/voxel_cache \
    +training.enable_rollout=False
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import time
import torch
import hydra
import pathlib
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.dataset.base_dataset import BaseImageDataset
from equi_diffpo.model.common.rotation_transformer import RotationTransformer

max_steps = {
    'stack_d1': 400, 'stack_three_d1': 400, 'square_d2': 400,
    'threading_d2': 400, 'coffee_d2': 400, 'three_piece_assembly_d2': 500,
    'hammer_cleanup_d1': 500, 'mug_cleanup_d0': 500, 'mug_cleanup_d1': 500,
    'kitchen_d1': 800, 'nut_assembly_d0': 500, 'pick_place_d0': 1000,
    'coffee_preparation_d1': 800, 'tool_hang': 700,
    'can': 400, 'lift': 400, 'square': 400,
}

def get_ws_x_center(task_name):
    if task_name.startswith('kitchen_') or task_name.startswith('hammer_cleanup_'):
        return -0.2
    return 0.

def get_ws_y_center(task_name):
    return 0.

OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps[x], replace=True)
OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
OmegaConf.register_new_resolver("get_ws_y_center", get_ws_y_center, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('equi_diffpo', 'config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    device = torch.device(cfg.training.device)

    # --- Dataset ---
    t0 = time.time()
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    print(f"\n[PROFILE] Dataset init: {time.time()-t0:.2f}s")
    print(f"[PROFILE] Dataset length: {len(dataset)}")

    dataloader = DataLoader(dataset, **cfg.dataloader)

    # --- Model ---
    t0 = time.time()
    policy = hydra.utils.instantiate(cfg.policy)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    policy.to(device)
    policy.train()
    print(f"[PROFILE] Model init + to GPU: {time.time()-t0:.2f}s")

    # --- Warm up dataloader ---
    print(f"\n--- Profiling {5} iterations ---\n")
    iterator = iter(dataloader)

    for step in range(5):
        torch.cuda.synchronize()

        # 1. Data loading
        t_data = time.time()
        batch = next(iterator)
        t_data = time.time() - t_data

        # 2. Transfer to GPU
        t_transfer = time.time()
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        torch.cuda.synchronize()
        t_transfer = time.time() - t_transfer

        # 3. Normalize
        t_norm = time.time()
        nobs = policy.normalizer.normalize(batch['obs'])
        nactions = policy.normalizer['action'].normalize(batch['action'])
        torch.cuda.synchronize()
        t_norm = time.time() - t_norm

        # 4. Sparse to dense (if applicable)
        t_sparse = time.time()
        if hasattr(policy, '_sparse_to_dense_voxels'):
            nobs = policy._sparse_to_dense_voxels(nobs)
        torch.cuda.synchronize()
        t_sparse = time.time() - t_sparse

        # 5. Rot aug
        t_rot = time.time()
        if policy.rot_aug:
            nobs, nactions = policy.rot_randomizer(nobs, nactions)
        torch.cuda.synchronize()
        t_rot = time.time() - t_rot

        # 6. Encoder forward
        t_enc = time.time()
        nobs_features = policy.enc(nobs)
        torch.cuda.synchronize()
        t_enc = time.time() - t_enc

        # 7. Full compute_loss (includes all above + diffusion)
        # Reset for a clean measurement
        torch.cuda.synchronize()
        t_loss = time.time()
        loss = policy.compute_loss(batch)
        torch.cuda.synchronize()
        t_loss = time.time() - t_loss

        # 8. Backward
        t_back = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t_back = time.time() - t_back

        policy.zero_grad()

        # Print breakdown
        print(f"[Step {step}] "
              f"data={t_data:.3f}s  "
              f"transfer={t_transfer:.3f}s  "
              f"norm={t_norm:.3f}s  "
              f"sparse2dense={t_sparse:.3f}s  "
              f"rot_aug={t_rot:.3f}s  "
              f"encoder={t_enc:.3f}s  "
              f"full_loss={t_loss:.3f}s  "
              f"backward={t_back:.3f}s  "
              f"TOTAL={t_data+t_transfer+t_loss+t_back:.3f}s")

        # Print batch info
        if step == 0:
            for k, v in batch['obs'].items():
                if torch.is_tensor(v):
                    print(f"  obs[{k}]: {v.shape} {v.dtype} {v.device}")


if __name__ == "__main__":
    main()
