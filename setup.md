# fine tune setup bimanual so 101 with gr00t repository

````bash
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
```


On hpc cluster
```bash
module load Python/3.10.8-GCCcore-12.2.0
cd /home/jxl2244/Isaac-GR00T
python -m venv gr00t
source /home/jxl2244/Isaac-GR00T/gr00t/bin/activate
pip install -U "huggingface_hub"
pip install --upgrade setuptools
pip install -e .[base]
pip install torch
pip install --no-build-isolation flash-attn==2.7.1.post4]
```
```bash
# Create logs directory
mkdir -p /home/jxl2244/Isaac-GR00T/logs

# Load modules
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.6.0  # Required for flash-attn compilation (must be >=12.4 for torch cu124)

# Create venv in home (stays persistent)
cd /home/jxl2244/Isaac-GR00T
python -m venv gr00t
source gr00t/bin/activate

# Install packages (one-time)
pip install -U "huggingface_hub"
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
```

### ~~Download demo datasets from Hugging Face~~ (SKIP THIS STEP)
**NOTE:** Datasets are now automatically downloaded to temporary storage during SLURM job execution to avoid home directory quota issues. The SLURM scripts handle dataset download automatically.

If you still want to download datasets manually to home directory (not recommended due to storage limits):
```bash
# OPTIONAL - Only if you want datasets in home directory
hf download --repo-type dataset YieumYoon/recode-bimanual-red-block-basket-v2.1 \
    --local-dir ./demo_data/recode-bimanual-red-block-basket-v2.1

hf download --repo-type dataset YieumYoon/recode-bimanual-red-block-basket-2-v2.1 \
    --local-dir ./demo_data/recode-bimanual-red-block-basket-2-v2.1

hf download --repo-type dataset YieumYoon/recode-bimanual-red-block-basket-4-v2.1 \
    --local-dir ./demo_data/recode-bimanual-red-block-basket-4-v2.1
```


### Creating Modality file

Use this once to make the datasets “GR00T-compatible” without changing any repo code.

1) Create modality.json for each dataset

```bash
# Datasets you downloaded
DATASETS=(
    recode-bimanual-red-block-basket-v2.1
    recode-bimanual-red-block-basket-2-v2.1
    recode-bimanual-red-block-basket-4-v2.1
)

# Write the same modality.json to each dataset's meta folder
for d in "${DATASETS[@]}"; do
    mkdir -p "./demo_data/$d/meta"
    cat > "./demo_data/$d/meta/modality.json" <<'JSON'
{
    "state": {
        "left_arm": { "start": 0, "end": 5 },
        "left_gripper": { "start": 5, "end": 6 },
        "right_arm": { "start": 6, "end": 11 },
        "right_gripper": { "start": 11, "end": 12 }
    },
    "action": {
        "left_arm": { "start": 0, "end": 5 },
        "left_gripper": { "start": 5, "end": 6 },
        "right_arm": { "start": 6, "end": 11 },
        "right_gripper": { "start": 11, "end": 12 }
    },
    "video": {
        "left_gripper":  { "original_key": "observation.images.left_gripper" },
        "right_gripper": { "original_key": "observation.images.right_gripper" },
        "top":           { "original_key": "observation.images.top" }
    },
    "annotation": {
        "human.task_description": { "original_key": "task_index" }
    }
}
JSON
    echo "Wrote modality.json for $d"
done
```

2) Fix info.json path placeholders (expected by the loader)

The loader uses episode_chunk as the placeholder. Update each dataset's meta/info.json.

```bash
python - <<'PY'
import json, pathlib
datasets = [
    'recode-bimanual-red-block-basket-v2.1',
    'recode-bimanual-red-block-basket-2-v2.1',
    'recode-bimanual-red-block-basket-4-v2.1',
]
for d in datasets:
        p = pathlib.Path('demo_data')/d/'meta'/'info.json'
        with p.open() as f: info = json.load(f)
        # Normalize placeholders expected by GR00T
        info['data_path']  = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
        info['video_path'] = 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'
        with p.open('w') as f: json.dump(info, f, indent=4)
        print('Updated', p)
PY
```

3) Force fresh stats computation (optional but recommended)

If your precomputed meta/stats.json causes an indexing error, back it up to let GR00T recompute it automatically.

```bash
for d in "${DATASETS[@]}"; do
    f="./demo_data/$d/meta/stats.json"
    if [ -f "$f" ]; then mv "$f" "$f.bak" && echo "Backed up $f"; fi
done
```

4) Quick validation (one dataset shown)

```bash
conda run -n gr00t \
    python scripts/load_dataset.py \
    --dataset-path ./demo_data/recode-bimanual-red-block-basket-v2.1 \
    --plot-state-action \
    --video-backend torchvision_av
```

Expected: shapes for state/action/video are printed, and a small plot is generated. Repeat for the other datasets if you like by changing --dataset-path.

**Common warnings you can ignore:**
- "Warning: Skipping left_hand / right_hand..." — The plotter looks for `hand` keys but your robot has `gripper` keys. This is normal and harmless.
- "Initialized dataset ... with EmbodimentTag.GR1" — By default datasets use a generic tag. If you're fine-tuning a new robot type (not GR1 humanoid), this is expected. The model will learn your robot's specific action space during training.

Note: If you used the download paths above, your fine-tune command should reference the -v2.1 directories (adjust the three --dataset-path entries accordingly).


### Create custom data config for SO-100

Since your SO-100 robot has different video keys and state/action structure than the built-in configs, create a custom data config:

```bash
cat > recode_data_config.py <<'PYTHON'
"""
Custom data configuration for bimanual SO-100 robot (recode datasets).
This config works with the modality.json files you created.
"""

from gr00t.experiment.data_config import BaseDataConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.video import (
    VideoToTensor,
    VideoCrop,
    VideoResize,
    VideoColorJitter,
    VideoToNumpy,
)
from gr00t.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.concat import ConcatTransform
from gr00t.model.transforms import GR00TTransform


class RecodeBimanualDataConfig(BaseDataConfig):
    """Data config for bimanual SO-100 with 3 cameras and joint position control."""

    # Video keys matching your modality.json
    video_keys = [
        "video.left_gripper",
        "video.right_gripper",
        "video.top",
    ]

    # State keys matching your modality.json (12D: 2 arms × (5 joints + 1 gripper))
    state_keys = [
        "state.left_arm",
        "state.left_gripper",
        "state.right_arm",
        "state.right_gripper",
    ]

    # Action keys matching your modality.json
    action_keys = [
        "action.left_arm",
        "action.left_gripper",
        "action.right_arm",
        "action.right_gripper",
    ]

    # Language/annotation keys
    language_keys = ["annotation.human.task_description"]

    # Observation and action horizons
    observation_indices = [0]  # Current observation only
    action_indices = list(range(16))  # 16 future action steps

    # Normalization: use min_max for all state/action (joint positions and gripper)
    state_normalization_modes = {
        "state.left_arm": "min_max",
        "state.left_gripper": "min_max",
        "state.right_arm": "min_max",
        "state.right_gripper": "min_max",
    }

    action_normalization_modes = {
        "action.left_arm": "min_max",
        "action.left_gripper": "min_max",
        "action.right_arm": "min_max",
        "action.right_gripper": "min_max",
    }

    def transform(self):
        """Define the transform pipeline for your robot."""
        transforms = [
            # Video transforms (resize to 224x224 for the model)
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),

            # State transforms (normalize joint positions)
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),

            # Action transforms (normalize joint positions)
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),

            # Concatenate all modalities
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),

            # Final GR00T transform (pad to model's expected dimensions)
            GR00TTransform(
                state_horizon=len(self.observation_indices),  # 1
                action_horizon=len(self.action_indices),      # 16
                max_state_dim=64,   # Pad state to 64 dims
                max_action_dim=32,  # Pad action to 32 dims
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
PYTHON

echo "Created recode_data_config.py"
```

This config:
- Matches your modality.json video keys (`left_gripper`, `right_gripper`, `top`)
- Handles your 12D joint position state/action (left/right arm + gripper)
- Applies standard video augmentation and normalization
- Pads to GR00T's expected dimensions (64 state, 32 action)


### fine tuning with the dataset
python scripts/load_dataset.py --dataset-path ./demo_data/recode-bimanual-red-block-basket-v2.1/ ./demo_data/recode-bimanual-red-block-basket-2-v2.1/ ./demo_data/recode-bimanual-red-block-basket-4-v2.1/ --plot-state-action --video-backend torchvision_av


Start fine-tuning with reduced batch size for 11GB GPU:

```bash
conda activate gr00t
cd /home/junsu-lee/Documents/github/Isaac-GR00T

python scripts/gr00t_finetune.py \
  --dataset-path ./demo_data/recode-bimanual-red-block-basket-v2.1/ ./demo_data/recode-bimanual-red-block-basket-2-v2.1/ ./demo_data/recode-bimanual-red-block-basket-4-v2.1/ \
  --num-gpus 1 \
  --output-dir ./so101-bimanual-checkpoints \
  --max-steps 10000 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --data-config recode_data_config:RecodeBimanualDataConfig \
  --video-backend torchvision_av \
  --no-tune_diffusion_model \
  --embodiment-tag new_embodiment
```

Parameters:

- `--batch-size 4`: Reduced for GPU memory constraints (default 32)
- `--gradient-accumulation-steps 8`: Effective batch size = 4 × 8 = 32
- `--no-tune_diffusion_model`: Freeze diffusion model to save memory
- `--embodiment-tag new_embodiment`: Explicitly set embodiment tag for your bimanual SO-100 (not GR1 humanoid)
- `--data-config recode_data_config:RecodeBimanualDataConfig`: Your custom config that matches SO-100's video keys and joint structure
- Total episodes: 30 + 25 + 45 = 100 episodes, 84,161 frames

Training progress will be logged to Weights & Biases (wandb).

Checkpoints will be saved every 1000 steps to `./recode-checkpoints/`.
````
