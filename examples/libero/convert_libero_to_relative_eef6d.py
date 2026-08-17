"""Convert LIBERO controller commands to per-frame relative EEF6D increments.

The destination remains a standard LeRobot dataset. OpenPI's normal
``delta_timestamps`` loader assembles the per-frame 10D actions into chunks; the
LIBERO policy transform converts those increments into inclusive cumulative SE(3)
targets for training.

Usage:
    uv run examples/libero/convert_libero_to_relative_eef6d.py \
        --source-repo-id physical-intelligence/libero \
        --output-repo-id local/libero_relative_eef6d
"""

from __future__ import annotations

import copy

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm
import tyro

from openpi.policies import libero_policy


def convert_action(action: np.ndarray, *, terminal: bool) -> np.ndarray:
    """Convert one 7D LIBERO command; terminal motion is neutral for safe clamping."""
    converted = libero_policy.libero_actions_to_eef6d(np.asarray(action, dtype=np.float32)).copy()
    if terminal:
        converted[:3] = 0
        converted[3:9] = [1, 0, 0, 0, 1, 0]
    return converted


def convert_dataset(
    source_repo_id: str = "physical-intelligence/libero",
    output_repo_id: str = "local/libero_relative_eef6d",
    image_writer_threads: int = 10,
) -> None:
    """Copy a LIBERO dataset while replacing 7D actions with 10D EEF6D increments."""
    output_root = HF_LEROBOT_HOME / output_repo_id
    if output_root.exists():
        raise FileExistsError(f"Destination already exists: {output_root}")

    source = LeRobotDataset(source_repo_id)
    features = {
        "image": copy.deepcopy(source.features["image"]),
        "wrist_image": copy.deepcopy(source.features["wrist_image"]),
        "state": copy.deepcopy(source.features["state"]),
        "actions": {"dtype": "float32", "shape": (10,), "names": ["actions"]},
    }
    destination = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=source.fps,
        robot_type=source.meta.robot_type,
        features=features,
        use_videos=False,
        image_writer_threads=image_writer_threads,
    )

    try:
        for episode_index in tqdm(range(source.num_episodes), desc="Converting LIBERO episodes"):
            start = int(source.episode_data_index["from"][episode_index])
            end = int(source.episode_data_index["to"][episode_index])
            for index in range(start, end):
                frame = source[index]
                destination.add_frame(
                    {
                        "image": libero_policy.parse_image(frame["image"]),
                        "wrist_image": libero_policy.parse_image(frame["wrist_image"]),
                        "state": frame["state"],
                        "actions": convert_action(frame["actions"], terminal=index == end - 1),
                        "task": frame["task"],
                    }
                )
            destination.save_episode()
    finally:
        destination.stop_image_writer()

    print(f"Converted dataset written to {output_root}")


if __name__ == "__main__":
    tyro.cli(convert_dataset)
