import importlib.util
from pathlib import Path

from lerobot.common.datasets import lerobot_dataset
import numpy as np

_SCRIPT = Path(__file__).parents[3] / "examples/libero/convert_libero_to_relative_eef6d.py"
_SPEC = importlib.util.spec_from_file_location("convert_libero_to_relative_eef6d", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
convert_libero_to_relative_eef6d = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(convert_libero_to_relative_eef6d)


def test_converter_encodes_command_and_neutralizes_terminal_motion():
    action = np.array([0.1, -0.2, 0.3, 0, 0, np.pi / 2, 1], dtype=np.float32)

    converted = convert_libero_to_relative_eef6d.convert_action(action, terminal=False)
    terminal = convert_libero_to_relative_eef6d.convert_action(action, terminal=True)

    np.testing.assert_allclose(converted, [0.1, -0.2, 0.3, 0, -1, 0, 1, 0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(terminal, [0, 0, 0, 1, 0, 0, 0, 1, 0, 1], atol=1e-6)


def test_converter_writes_standard_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(lerobot_dataset, "HF_LEROBOT_HOME", tmp_path)
    monkeypatch.setattr(convert_libero_to_relative_eef6d, "HF_LEROBOT_HOME", tmp_path)
    features = {
        "image": {"dtype": "image", "shape": (4, 4, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (4, 4, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }
    source = lerobot_dataset.LeRobotDataset.create(
        "test/source", fps=10, robot_type="panda", features=features, use_videos=False
    )
    for index in range(2):
        source.add_frame(
            {
                "image": np.full((4, 4, 3), index, dtype=np.uint8),
                "wrist_image": np.full((4, 4, 3), index + 1, dtype=np.uint8),
                "state": np.full(8, index, dtype=np.float32),
                "actions": np.array([0.1, 0, 0, 0, 0, 0, -1 + 2 * index], dtype=np.float32),
                "task": "test task",
            }
        )
    source.save_episode()

    convert_libero_to_relative_eef6d.convert_dataset("test/source", "test/output", image_writer_threads=0)
    output = lerobot_dataset.LeRobotDataset("test/output")

    assert output.num_episodes == 1
    assert output.num_frames == 2
    assert output.fps == source.fps
    assert output.features["actions"]["shape"] == (10,)
    assert output[0]["task"] == "test task"
    np.testing.assert_array_equal(output[1]["state"], source[1]["state"])
    np.testing.assert_array_equal(output[1]["image"], source[1]["image"])
    np.testing.assert_array_equal(output[1]["wrist_image"], source[1]["wrist_image"])
    np.testing.assert_array_equal([output[index]["episode_index"] for index in range(2)], [0, 0])
    np.testing.assert_array_equal([output[index]["frame_index"] for index in range(2)], [0, 1])
    np.testing.assert_allclose(output[0]["actions"][:3], [0.1, 0, 0])
    np.testing.assert_allclose(output[1]["actions"][:9], [0, 0, 0, 1, 0, 0, 0, 1, 0])
    assert output[1]["actions"][9] == source[1]["actions"][6]
