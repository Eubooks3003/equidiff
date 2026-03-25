"""
Standalone evaluation script for equidiff voxel policy checkpoints.
Runs rollouts with live voxelization (depth -> point cloud -> voxels).

Requires lpwm-dev to be a sibling directory (for VoxelGridXYZ).

Usage:
    python eval.py --checkpoint path/to/latest.ckpt \
        --dataset_path path/to/mug_cleanup_d0.hdf5 \
        --n_test 50 --seeds 42 123 456 \
        --device cuda:0

Example with wandb diagnostics:
    python eval.py \
        --checkpoint data/outputs/.../checkpoints/latest.ckpt \
        --dataset_path ~/data/3D-DLP-mimicgen-data/core/mug_cleanup_d0.hdf5 \
        --n_test 50 --seeds 42 123 456 \
        --save_videos --video_res 512 \
        --wandb --wandb_project equidiff-eval
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


def depth_to_z(depth_raw, near_m, far_m):
    """Convert depth buffer to metric depth.
    Handles both raw OpenGL buffer ([0,1]) and already-metric depth.
    Exact match to EC-Diffuser's depth_to_z with mode='robosuite'.
    """
    d = depth_raw.squeeze().astype(np.float32)
    dmax = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0

    if dmax > 1.05:
        # Already metric — robomimic may have pre-converted
        return d

    d = np.clip(d, 0.0, 1.0)
    n, f = float(near_m), float(far_m)
    return (n * f) / (f - d * (f - n) + 1e-12)


def obs_to_voxels(sim, obs, cams, voxel_bounds, grid_whd=(128, 128, 128),
                  voxel_mode="avg_rgb", pixel_stride=1):
    """
    Convert robosuite observation to voxel grid via depth backprojection.
    Pipeline matches EC-Diffuser's MimicGenDLPWrapper exactly.

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

    # Near/far in meters (multiply by extent — the big gotcha)
    extent = float(sim.model.stat.extent)
    near_m = float(sim.model.vis.map.znear) * extent
    far_m = float(sim.model.vis.map.zfar) * extent

    all_xyz, all_rgb = [], []

    for cam in cams:
        depth_key = f"{cam}_depth"
        img_key = f"{cam}_image"
        if depth_key not in obs or img_key not in obs:
            print(f"  WARNING: {depth_key} or {img_key} not in obs. "
                  f"Available keys: {[k for k in obs if 'image' in k or 'depth' in k]}")
            continue

        depth_raw = obs[depth_key]
        img = obs[img_key]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        # --- Depth to metric (handles both raw buffer and pre-converted) ---
        z = depth_to_z(depth_raw, near_m, far_m)
        H, W = z.shape

        # --- Intrinsics from FOV ---
        cam_id = sim.model.camera_name2id(cam)
        fovy = float(sim.model.cam_fovy[cam_id])
        K = compute_K_from_fovy(fovy, W, H)

        # --- Backproject to camera-frame points (OpenCV: x-right, y-down, z-forward) ---
        vv = np.arange(0, H, pixel_stride, dtype=np.int32)
        uu = np.arange(0, W, pixel_stride, dtype=np.int32)
        U, V = np.meshgrid(uu, vv)
        Z = z[V, U]
        valid = np.isfinite(Z) & (Z > 0)
        if not np.any(valid):
            continue

        Uv = U[valid].astype(np.float32)
        Vv = V[valid].astype(np.float32)
        Zv = Z[valid].astype(np.float32)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        X = (Uv - cx) / fx * Zv
        Y = (Vv - cy) / fy * Zv
        pts_cam = np.stack([X, Y, Zv], axis=-1).astype(np.float32)

        # --- CV -> MuJoCo(OpenGL) camera coords, then to world ---
        # Exact match to EC-Diffuser _cam_to_world
        R = np.array(sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)
        pos = np.array(sim.data.cam_xpos[cam_id], dtype=np.float32)

        pts_gl = pts_cam.copy()
        pts_gl[:, 1] *= -1.0   # y: down -> up
        pts_gl[:, 2] *= -1.0   # z: forward -> backward
        pts_world = (R @ pts_gl.T).T + pos

        # --- Colors ---
        v_idx, u_idx = V[valid], U[valid]
        rgb = img[v_idx, u_idx].astype(np.float32)
        if rgb.max() > 1.5:
            rgb /= 255.0

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


# ======================== Visualization ========================

def render_highres_frame(sim, cam_name, width=512, height=512):
    """Render a high-res RGB frame from MuJoCo sim."""
    frame = sim.render(width=width, height=height, camera_name=cam_name, depth=False)
    frame = frame[::-1].copy()  # MuJoCo renders bottom-up
    return frame


def voxel_projections(voxels):
    """Render voxel grid as 3 max-intensity projection images (XY, XZ, YZ).

    Args:
        voxels: [3, D, H, W] tensor or array (RGB in [0, 1])

    Returns:
        dict of {"xy": img, "xz": img, "yz": img} as uint8 HWC arrays
    """
    if isinstance(voxels, torch.Tensor):
        voxels = voxels.cpu().numpy()

    v = np.clip(voxels, 0, 1)  # [3, D, H, W]
    # Max-intensity projections along each axis
    xy = v.max(axis=1)                   # [3, H, W] — project along D (depth/z)
    xz = v.max(axis=2)                   # [3, D, W] — project along H (y)
    yz = v.max(axis=3)                   # [3, D, H] — project along W (x)

    def to_img(proj):
        img = np.transpose(proj, (1, 2, 0))  # [H, W, 3]
        return (img * 255).astype(np.uint8)

    return {"xy": to_img(xy), "xz": to_img(xz), "yz": to_img(yz)}


def log_voxels_3d(name, voxels, step, output_dir='eval_output',
                  use_wandb=True, topk=60000):
    """Log voxel grid as interactive 3D plotly scatter.

    Saves HTML locally under output_dir/voxels/ AND logs to wandb.

    Args:
        name: log key
        voxels: [3, D, H, W] tensor or array (RGB values in [0, 1])
        step: sim timestep (for filename/caption)
        output_dir: local save directory
        use_wandb: whether to also log to wandb
        topk: max points to render
    """
    import plotly.graph_objects as go

    if isinstance(voxels, torch.Tensor):
        voxels = voxels.cpu().numpy()

    C, D, H, W = voxels.shape
    mag = np.sqrt(np.sum(voxels ** 2, axis=0))  # [D, H, W]
    mask = mag > 0.1
    n_occupied = int(mask.sum())

    if n_occupied == 0:
        print(f"  [voxel_log] t={step}: EMPTY voxel grid, skipping")
        return

    zz, yy, xx = np.where(mask)
    colors = voxels[:, zz, yy, xx].T  # [N, 3]
    scores = mag[zz, yy, xx]

    if len(zz) > topk:
        idx = np.argsort(scores)[-topk:]
        zz, yy, xx = zz[idx], yy[idx], xx[idx]
        colors = colors[idx]

    colors = np.clip(colors, 0, 1) * 255
    color_strs = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in colors]

    fig = go.Figure(data=[go.Scatter3d(
        x=xx.tolist(), y=yy.tolist(), z=zz.tolist(),
        mode='markers',
        marker=dict(size=2, color=color_strs, opacity=0.8),
    )])
    fig.update_layout(
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"{name} t={step} ({len(zz)} pts)",
    )

    # Always save locally as HTML
    voxel_dir = os.path.join(output_dir, 'voxels')
    os.makedirs(voxel_dir, exist_ok=True)
    html_path = os.path.join(voxel_dir, f'{name}_t{step:04d}.html')
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"  [voxel_log] t={step}: {n_occupied} occupied voxels, saved {html_path}")

    # Log to wandb
    if use_wandb:
        import wandb
        wandb.log({name: wandb.Html(open(html_path).read())})


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
             save_videos=False, output_dir='eval_output', video_episodes=4,
             video_res=512, use_wandb=False, log_every_replan=1):
    """Run evaluation rollouts for a single seed with live voxelization."""

    import collections

    # Setup env
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta['env_kwargs']['use_object_obs'] = False
    # Enable depth rendering and set camera resolution (required for live voxelization)
    cam_list = list(cams)
    env_meta['env_kwargs']['camera_names'] = cam_list
    env_meta['env_kwargs']['camera_depths'] = [True] * len(cam_list)
    env_meta['env_kwargs']['camera_heights'] = [256] * len(cam_list)
    env_meta['env_kwargs']['camera_widths'] = [256] * len(cam_list)

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

    try:
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=True,
            use_image_obs=True, use_depth_obs=True)
    except TypeError:
        # older robomimic API
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=True,
            use_image_obs=True)

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

        # Action tracking for wandb
        action_log = []
        replan_count = 0

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

            # Log voxel 3D scatter (local HTML + wandb)
            if ep < video_episodes and replan_count % log_every_replan == 0:
                log_voxels_3d("voxel_obs", voxels, step=t,
                              output_dir=output_dir, use_wandb=use_wandb)

            # Capture high-res frame for video (at replan step)
            if frames is not None:
                frame = render_highres_frame(
                    env.env.sim, cam_name=cams[0],
                    width=video_res, height=video_res)
                frames.append(frame)

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

            replan_count += 1

            # Execute action steps
            for a_idx in range(action.shape[1]):
                if t >= max_steps:
                    break
                a = action[0, a_idx]

                # Track actions
                action_log.append({
                    "t": t, "replan": replan_count - 1, "a_idx": a_idx,
                    "x": float(a[0]), "y": float(a[1]), "z": float(a[2]),
                    "gripper": float(a[9]) if len(a) > 9 else float(a[-1]),
                })


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

                # Capture frame at EVERY sim step for smooth video
                if frames is not None:
                    frame = render_highres_frame(
                        env.env.sim, cam_name=cams[0],
                        width=video_res, height=video_res)
                    frames.append(frame)

                if done:
                    break

        successes.append(float(max_reward))
        pbar.set_postfix(sr=f"{np.mean(successes)*100:.0f}%", succ=sum(1 for s in successes if s > 0.5))

        # Save video (local + wandb)
        if frames is not None and len(frames) > 0:
            try:
                import imageio
                status = "success" if max_reward > 0.5 else "fail"
                video_dir = os.path.join(output_dir, 'videos', f'seed{seed}')
                os.makedirs(video_dir, exist_ok=True)
                video_path = os.path.join(video_dir, f'ep{ep:02d}_{status}.mp4')
                imageio.mimsave(video_path, frames, fps=20)

                if use_wandb:
                    import wandb
                    wandb.log({
                        f"video/seed{seed}_ep{ep:02d}_{status}": wandb.Video(
                            video_path, fps=20, format="mp4"),
                    })
            except Exception as e:
                print(f"Failed to save video: {e}")

        # Log action trajectory plot to wandb
        if use_wandb and ep < video_episodes and len(action_log) > 0:
            import wandb
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ts = [a["t"] for a in action_log]
            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
            for ax, key, label in zip(axes,
                    ["x", "y", "z", "gripper"],
                    ["pos_x", "pos_y", "pos_z", "gripper"]):
                vals = [a[key] for a in action_log]
                ax.plot(ts, vals, linewidth=1)
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel("timestep")
            status = "success" if max_reward > 0.5 else "fail"
            axes[0].set_title(f"seed{seed} ep{ep} ({status}, replans={replan_count})")
            plt.tight_layout()
            wandb.log({f"actions/trajectory_seed{seed}_ep{ep:02d}": wandb.Image(fig)})
            plt.close(fig)

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
    parser.add_argument('--video_res', type=int, default=512,
                        help='Resolution for video rendering (default: 512)')
    parser.add_argument('--output_dir', type=str, default='eval_output')
    # wandb
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='equidiff-eval')
    parser.add_argument('--log_every_replan', type=int, default=1,
                        help='Log voxel obs to wandb every N replans (1=every replan)')
    args = parser.parse_args()

    # Init wandb
    if args.wandb:
        import wandb
        ckpt_name = os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint)))
        run_name = f"eval_{ckpt_name}_n{args.n_test}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

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
            video_res=args.video_res,
            use_wandb=args.wandb,
            log_every_replan=args.log_every_replan,
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

    # Log summary to wandb
    if args.wandb:
        import wandb
        log_data = {
            "summary/mean_success_rate": float(mean_rate),
            "summary/std_success_rate": float(std_rate),
        }
        for seed in args.seeds:
            log_data[f"summary/seed_{seed}"] = all_results[seed]['success_rate']
        wandb.log(log_data)
        wandb.finish()

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
