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

    # State keys matching modality.json structure
    state_keys = [
        "state.left_arm",
        "state.left_gripper", 
        "state.right_arm",
        "state.right_gripper",
    ]

    # Action keys matching modality.json structure
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

    # Normalization: use min_max for all state/action components
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
            VideoResize(apply_to=self.video_keys, height=224,
                        width=224, interpolation="linear"),
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
