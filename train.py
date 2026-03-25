"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from equi_diffpo.workspace.base_workspace import BaseWorkspace

max_steps = {
    'stack_d1': 400,
    'stack_three_d1': 400,
    'square_d2': 400,
    'threading_d2': 400,
    'coffee_d2': 400,
    'three_piece_assembly_d2': 500,
    'three_piece_assembly_d0': 500,
    'hammer_cleanup_d1': 500,
    'mug_cleanup_d0': 500,
    'mug_cleanup_d1': 500,
    'kitchen_d1': 800,
    'nut_assembly_d0': 500,
    'pick_place_d0': 1000,
    'coffee_preparation_d1': 800,
    'tool_hang': 700,
    'can': 400,
    'lift': 400,
    'square': 400,
    'square_d0': 400,
}

def _resolve_task_variant(task_name: str, table: dict):
    """
    Map variant names like 'square_d0' to a known key (e.g. 'square').
    This keeps Hydra configs flexible while still failing loudly for truly unknown tasks.
    """
    if task_name in table:
        return task_name

    base = task_name
    d_suffix = None
    if '_' in task_name:
        maybe_base, maybe_diff = task_name.rsplit('_', 1)
        if maybe_diff.startswith('d') and maybe_diff[1:].isdigit():
            base = maybe_base
            d_suffix = maybe_diff

    candidates = []
    if d_suffix is not None:
        # Prefer the same difficulty if it exists, then common fallbacks, then base.
        candidates.extend([
            f"{base}_{d_suffix}",
            f"{base}_d2",
            f"{base}_d1",
            f"{base}_d0",
            base,
        ])
    else:
        candidates.append(base)

    for key in candidates:
        if key in table:
            return key
    return None


def get_max_steps(task_name: str) -> int:
    key = _resolve_task_variant(task_name, max_steps)
    if key is None:
        # Default for unseen MimicGen d0 tasks.
        if task_name.rsplit('_', 1)[-1] == 'd0':
            return 500
        raise KeyError(
            f"Unknown task_name '{task_name}' for max_steps resolver. "
            f"Known keys: {sorted(max_steps.keys())}"
        )
    return int(max_steps[key])


def get_ws_x_center(task_name):
    if task_name.startswith('kitchen_') or task_name.startswith('hammer_cleanup_'):
        return -0.2
    else:
        return 0.

def get_ws_y_center(task_name):
    return 0.

OmegaConf.register_new_resolver("get_max_steps", get_max_steps, replace=True)
OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
OmegaConf.register_new_resolver("get_ws_y_center", get_ws_y_center, replace=True)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'equi_diffpo','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
