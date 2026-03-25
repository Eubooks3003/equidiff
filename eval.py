"""
Standalone evaluation script for equidiff voxel policy checkpoints.
Runs rollouts with live voxelization (depth -> point cloud -> voxels).

Requires lpwm-dev to be a sibling directory (for VoxelGridXYZ).

Usage:
    python eval.py --checkpoint path/to/latest.ckpt \
        --dataset_path path/to/mug_cleanup_d0.hdf5 \
        --n_test 50 --seeds 42 123 456 \
        --device cuda:0

Example:
    python eval.py \
        --checkpoint data/outputs/.../checkpoints/latest.ckpt \
        --dataset_path ~/data/3D-DLP-mimicgen-data/core/mug_cleanup_d0.hdf5 \
        --n_test 50 --seeds 42 123 456 \
        --save_videos
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import argparse
import json
import numpy as np
import torch
import dill
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

# Add lpwm-dev to path for VoxelGridXYZ
LPWM_DEV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lpwm-dev"))
if os.path.isdir(LPWM_DEV) and LPWM_DEV not in sys.path:
    sys.path.append(LPWM_DEV)

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.model.common.rotation_transformer import RotationTransformer

# Per-task voxel bounds (must match preprocessing)
TASK_VOXEL_BOUNDS = {
    "hammer_cleanup": {"xmin": -0.4, "xmax": 0.2, "ymin": -0.4, "ymax": 0.4, "zmin": 0.7, "zmax": 1.4},
    "nut_assembly": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.5, "ymax": 0.4, "zmin": 0.1, "zmax": 1.6},
    "pick_place": {"xmin": -0.5, "xmax": 0.3, "ymin": -0.5, "ymax": 1.0, "zmin": 0.0, "zmax": 1.7},
    "square": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.5, "ymax": 0.4, "zmin": 0.0, "zmax": 1.6},
    "stack_three": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.4, "ymax": 0.4, "zmin": 0.2, "zmax": 1.6},
    "stack": {"xmin": -0.7, "xmax": 0.4, "ymin": -0.5, "ymax": 0.5, "zmin": -0.2, "zmax": 1.6},
    "threading": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.4, "ymax": 0.4, "zmin": 0.2, "zmax": 1.6},
    "kitchen": {"xmin": -1.1, "xmax": 0.3, "ymin": -0.4, "ymax": 0.4, "zmin": 0.7, "zmax": 1.4},
    "coffee": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.4, "ymax": 0.4, "zmin": 0.2, "zmax": 1.6},
    "coffee_preparation": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.5, "ymax": 0.4, "zmin": 0.2, "zmax": 1.6},
    "mug_cleanup": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.4, "ymax": 0.5, "zmin": 0.2, "zmax": 1.6},
    "three_piece_assembly": {"xmin": -0.7, "xmax": 0.4, "ymin": -0.4, "ymax": 0.5, "zmin": 0.2, "zmax": 1.6},
}

max_steps_dict = {
    'stack_d1': 400, 'stack_three_d1': 400, 'square_d2': 400,
    'threading_d2': 400, 'coffee_d2': 400, 'three_piece_assembly_d2': 500,
    'hammer_cleanup_d1': 500, 'mug_cleanup_d0': 500, 'mug_cleanup_d1': 500,
    'kitchen_d1': 800, 'nut_assembly_d0': 500, 'pick_place_d0': 1000,
    'coffee_preparation_d1': 800, 'tool_hang': 700,
    'can': 400, 'lift': 400, 'square': 400,
}


# ======================== Live Voxelization ========================

def compute_K_from_fovy(fovy_deg, W, H):
    """Compute camera intrinsic matrix from vertical FOV."""
    fovy = np.deg2rad(float(fovy_deg))
    fy = (H / 2.0) / np.tan(fovy / 2.0)
    fx = fy * (W / H)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def backproject_depth(depth_z, K, pixel_stride=1):
    """Backproject depth map to 3D camera-frame points."""
    z = depth_z.squeeze().astype(np.float32)
    H, W = z.shape
    vv = np.arange(0, H, pixel_stride, dtype=np.int32)
    uu = np.arange(0, W, pixel_stride, dtype=np.int32)
    U, V = np.meshgrid(uu, vv)
    Z = z[V, U]
    valid = np.isfinite(Z) & (Z > 0)
    if not np.any(valid):
        return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int32)
    Uv, Vv, Zv = U[valid].astype(np.float32), V[valid].astype(np.float32), Z[valid].astype(np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    X = (Uv - cx) / fx * Zv
    Y = (Vv - cy) / fy * Zv
    pts_cam = np.stack([X, Y, Zv], axis=-1)
    idxs = np.stack([V[valid], U[valid]], axis=-1).astype(np.int32)
    return pts_cam, idxs


def obs_to_voxels(sim, obs, cams, voxel_bounds, grid_whd=(128, 128, 128),
                  voxel_mode="avg_rgb", pixel_stride=1):
    """
    Convert robosuite observation to voxel grid via depth backprojection.

    Args:
        sim: robosuite MjSim object (for camera matrices)
        obs: observation dict with {cam}_image and {cam}_depth
        cams: list of camera names
        voxel_bounds: dict with xmin/xmax/ymin/ymax/zmin/zmax
        grid_whd: voxel grid dimensions
        voxel_mode: "avg_rgb", "occupancy", or "density"
        pixel_stride: downsample stride for depth backprojection

    Returns:
        voxels: [C, D, H, W] torch tensor
    """
    from datasets.voxelize_ds_wrapper import VoxelGridXYZ

    all_xyz, all_rgb = [], []

    for cam in cams:
        depth_key = f"{cam}_depth"
        img_key = f"{cam}_image"
        if depth_key not in obs or img_key not in obs:
            continue

        depth_raw = obs[depth_key].squeeze()
        img = obs[img_key]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        H, W = depth_raw.shape[:2]

        # Get camera intrinsics from sim
        fovy = sim.model.cam_fovy[sim.model.camera_name2id(cam)]
        K = compute_K_from_fovy(fovy, W, H)

        # Get camera extrinsics from sim
        cam_id = sim.model.camera_name2id(cam)
        cam_pos = sim.data.cam_xpos[cam_id]
        cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)
        # MuJoCo convention: camera looks along -z, y is down
        # Convert to standard: R_world_cam
        R_mj = cam_mat  # columns are camera axes in world frame
        # MuJoCo: x=right, y=down, z=backward -> OpenCV: x=right, y=down, z=forward
        R_cv = R_mj.copy()
        R_cv[:, 2] = -R_cv[:, 2]  # flip z axis

        # Build cam2world transform
        T_c2w = np.eye(4, dtype=np.float32)
        T_c2w[:3, :3] = R_cv
        T_c2w[:3, 3] = cam_pos

        # Convert depth: MuJoCo depth buffer is in [0, 1], convert to meters
        extent = sim.model.stat.extent
        near = sim.model.vis.map.znear * extent
        far = sim.model.vis.map.zfar * extent
        depth_m = near / (1.0 - depth_raw * (1.0 - near / far))

        # Backproject
        pts_cam, idxs = backproject_depth(depth_m, K, pixel_stride)
        if pts_cam.shape[0] == 0:
            continue

        # Get colors
        v, u = idxs[:, 0], idxs[:, 1]
        rgb = img[v, u].astype(np.float32)
        if rgb.max() > 1.5:
            rgb /= 255.0

        # Transform to world
        ones = np.ones((pts_cam.shape[0], 1), np.float32)
        pts_h = np.concatenate([pts_cam, ones], axis=1)
        pts_world = (T_c2w @ pts_h.T).T[:, :3]

        all_xyz.append(pts_world)
        all_rgb.append(rgb)

    if not all_xyz:
        C = 3 if voxel_mode == "avg_rgb" else 1
        W, H, D = grid_whd
        return torch.zeros(C, D, H, W)

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)

    # Crop to bounds
    b = voxel_bounds
    mask = (
        (xyz[:, 0] >= b["xmin"]) & (xyz[:, 0] <= b["xmax"]) &
        (xyz[:, 1] >= b["ymin"]) & (xyz[:, 1] <= b["ymax"]) &
        (xyz[:, 2] >= b["zmin"]) & (xyz[:, 2] <= b["zmax"])
    )
    if mask.sum() == 0:
        C = 3 if voxel_mode == "avg_rgb" else 1
        W, H, D = grid_whd
        return torch.zeros(C, D, H, W)
    xyz = xyz[mask]
    rgb = rgb[mask]

    # Voxelize
    pmin = torch.tensor([b["xmin"], b["ymin"], b["zmin"]], dtype=torch.float32)
    pmax = torch.tensor([b["xmax"], b["ymax"], b["zmax"]], dtype=torch.float32)
    pts_t = torch.from_numpy(xyz).float()
    colors_t = torch.from_numpy(rgb).float() if voxel_mode == "avg_rgb" else None

    W, H, D = grid_whd
    vg = VoxelGridXYZ(pts_t, colors_t, grid_whd=(W, H, D),
                      bounds=(pmin, pmax), mode=voxel_mode)
    return vg.to_dense()


# ======================== Policy Loading ========================

def load_policy_from_checkpoint(ckpt_path, device='cuda:0'):
    """Load trained policy from checkpoint."""
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']

    # Register resolvers needed by config
    OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps_dict.get(x, 500), replace=True)
    OmegaConf.register_new_resolver("get_ws_x_center",
        lambda x: -0.2 if x.startswith('kitchen_') or x.startswith('hammer_cleanup_') else 0., replace=True)
    OmegaConf.register_new_resolver("get_ws_y_center", lambda x: 0., replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)

    policy = hydra.utils.instantiate(cfg.policy)

    if 'ema_model' in payload['state_dicts']:
        policy.load_state_dict(payload['state_dicts']['ema_model'])
        print("Loaded EMA model weights")
    else:
        policy.load_state_dict(payload['state_dicts']['model'])
        print("Loaded model weights")

    policy.eval()
    policy.to(device)
    return policy, cfg


def get_task_base_name(task_name):
    """Extract base task name (e.g., 'mug_cleanup_d0' -> 'mug_cleanup')."""
    parts = task_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].startswith('d') and parts[1][1:].isdigit():
        return parts[0]
    return task_name


# ======================== Evaluation ========================

@torch.no_grad()
def run_eval(policy, cfg, dataset_path, n_test, seed, max_steps, device,
             grid_whd=(128, 128, 128), cams=("agentview", "sideview"),
             save_videos=False, output_dir='eval_output', video_episodes=4):
    """Run evaluation rollouts for a single seed with live voxelization."""

    import collections

    # Setup env
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta['env_kwargs']['use_object_obs'] = False

    abs_action = OmegaConf.to_container(cfg.task, resolve=True).get('abs_action', True)
    rotation_transformer = None
    if abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    task_name = cfg.task_name
    task_base = get_task_base_name(task_name)

    voxel_bounds = TASK_VOXEL_BOUNDS.get(task_base)
    if voxel_bounds is None:
        raise ValueError(f"No voxel bounds defined for task '{task_base}'. "
                         f"Available: {list(TASK_VOXEL_BOUNDS.keys())}")

    # Determine voxel size from shape_meta
    shape_meta_resolved = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    voxel_shape = shape_meta_resolved['obs']['voxels']['shape']
    voxel_size = voxel_shape[1]  # e.g., 128

    # Setup modality mapping for robomimic
    modality_mapping = collections.defaultdict(list)
    # We need RGB images and depth from cameras
    for cam in cams:
        modality_mapping['rgb'].append(f'{cam}_image')
        modality_mapping['depth'].append(f'{cam}_depth')
    modality_mapping['low_dim'].extend([
        'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'
    ])
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    successes = []
    pbar = tqdm(range(n_test), desc=f"Seed {seed}", unit="ep")

    for ep in pbar:
        ep_seed = seed * 100000 + ep
        np.random.seed(ep_seed)
        raw_obs = env.reset()

        # Video recording
        frames = [] if save_videos and ep < video_episodes else None

        done = False
        t = 0
        max_reward = 0.0

        while not done and t < max_steps:
            # Get raw obs
            if t > 0:
                raw_obs = env.get_observation()

            # Live voxelize
            voxels = obs_to_voxels(
                sim=env.env.sim,
                obs=raw_obs,
                cams=list(cams),
                voxel_bounds=voxel_bounds,
                grid_whd=(voxel_size, voxel_size, voxel_size),
                voxel_mode="avg_rgb",
            )  # [3, D, H, W]

            # Capture frame for video
            if frames is not None and f'{cams[0]}_image' in raw_obs:
                img = raw_obs[f'{cams[0]}_image']
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = np.transpose(img, (1, 2, 0))
                if img.max() <= 1.5:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                frames.append(img)

            # Build observation dict matching policy's expected format
            # Shape: [B, T, ...] where B=1, T=n_obs_steps
            obs_dict = {
                'voxels': voxels.unsqueeze(0).unsqueeze(0).to(device),  # [1, 1, 3, D, H, W]
                'robot0_eef_pos': torch.from_numpy(
                    raw_obs['robot0_eef_pos'].astype(np.float32)
                ).reshape(1, 1, 3).to(device),
                'robot0_eef_quat': torch.from_numpy(
                    raw_obs['robot0_eef_quat'].astype(np.float32)
                ).reshape(1, 1, 4).to(device),
                'robot0_gripper_qpos': torch.from_numpy(
                    raw_obs['robot0_gripper_qpos'].astype(np.float32)
                ).reshape(1, 1, 2).to(device),
            }

            # Run policy
            action_dict = policy.predict_action(obs_dict)
            action = action_dict['action'].detach().cpu().numpy()  # [1, n_action_steps, action_dim]

            if not np.all(np.isfinite(action)):
                print(f"WARNING: non-finite action at ep={ep} t={t}")
                break

            # Execute action steps
            for a_idx in range(action.shape[1]):
                if t >= max_steps:
                    break
                a = action[0, a_idx]

                # Convert from rotation_6d back to axis_angle for env
                env_action = a
                if abs_action and rotation_transformer is not None:
                    pos = a[:3]
                    rot6d = a[3:9]
                    gripper = a[9:]
                    rot_aa = rotation_transformer.inverse(rot6d)
                    env_action = np.concatenate([pos, rot_aa, gripper])

                raw_obs, reward, done, info = env.step(env_action)
                max_reward = max(max_reward, reward)
                t += 1

                if done:
                    break

        successes.append(float(max_reward))
        pbar.set_postfix(sr=f"{np.mean(successes)*100:.0f}%", succ=sum(1 for s in successes if s > 0.5))

        # Save video
        if frames is not None and len(frames) > 0:
            try:
                import imageio
                status = "success" if max_reward > 0.5 else "fail"
                video_dir = os.path.join(output_dir, 'videos', f'seed{seed}')
                os.makedirs(video_dir, exist_ok=True)
                video_path = os.path.join(video_dir, f'ep{ep:02d}_{status}.mp4')
                imageio.mimsave(video_path, frames, fps=20)
            except Exception as e:
                print(f"Failed to save video: {e}")

    env.close()

    success_rate = float(np.mean(successes))
    print(f"Seed {seed}: success_rate = {success_rate:.3f}")
    return success_rate, successes


def main():
    parser = argparse.ArgumentParser(description='Evaluate equidiff voxel policy checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to HDF5 dataset (overrides config)')
    parser.add_argument('--n_test', type=int, default=50)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--cams', type=str, nargs='+', default=['agentview', 'sideview'])
    parser.add_argument('--save_videos', action='store_true')
    parser.add_argument('--video_episodes', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='eval_output')
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    policy, cfg = load_policy_from_checkpoint(args.checkpoint, args.device)

    dataset_path = args.dataset_path or cfg.dataset_path
    dataset_path = os.path.expanduser(dataset_path)
    task_name = cfg.task_name

    if args.max_steps is not None:
        max_steps = args.max_steps
    else:
        max_steps = max_steps_dict.get(task_name, 500)

    # Get voxel size from config
    shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    voxel_size = shape_meta['obs']['voxels']['shape'][1]

    print(f"Task: {task_name}, max_steps: {max_steps}, voxel_size: {voxel_size}")
    print(f"Dataset: {dataset_path}")
    print(f"Cameras: {args.cams}")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Evaluating seed {seed} ({args.n_test} rollouts)")
        print(f"{'='*60}")

        success_rate, rewards = run_eval(
            policy=policy, cfg=cfg,
            dataset_path=dataset_path,
            n_test=args.n_test, seed=seed,
            max_steps=max_steps, device=args.device,
            grid_whd=(voxel_size, voxel_size, voxel_size),
            cams=tuple(args.cams),
            save_videos=args.save_videos,
            output_dir=args.output_dir,
            video_episodes=args.video_episodes,
        )
        all_results[seed] = {'success_rate': success_rate, 'rewards': rewards}

    seed_rates = [r['success_rate'] for r in all_results.values()]
    mean_rate = np.mean(seed_rates)
    std_rate = np.std(seed_rates)

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.n_test} rollouts x {len(args.seeds)} seeds")
    print(f"{'='*60}")
    for seed in args.seeds:
        print(f"  Seed {seed}: {all_results[seed]['success_rate']:.3f}")
    print(f"  Mean: {mean_rate:.3f} +/- {std_rate:.3f}")

    results_path = os.path.join(args.output_dir,
        f'eval_{task_name}_n{args.n_test}_seeds{"_".join(map(str, args.seeds))}.json')
    with open(results_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'task_name': task_name,
            'n_test': args.n_test,
            'seeds': args.seeds,
            'max_steps': max_steps,
            'mean_success_rate': float(mean_rate),
            'std_success_rate': float(std_rate),
            'per_seed': {str(k): v for k, v in all_results.items()},
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
