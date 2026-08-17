import numpy as np

from openpi.models import model as _model
from openpi.policies import libero_policy

_IDENTITY_ROT6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


def test_libero_command_action_round_trip():
    actions = np.array(
        [
            [0.1, -0.2, 0.3, 0.0, 0.0, np.pi / 2, -1.0],
            [-0.4, 0.5, -0.6, 0.1, -0.2, 0.3, 1.0],
        ],
        dtype=np.float32,
    )

    increments = libero_policy.libero_actions_to_eef6d(actions)
    relative = libero_policy.eef6d_increments_to_relative(increments)
    decoded = libero_policy.relative_eef6d_to_libero_actions(relative)

    np.testing.assert_allclose(decoded, actions, atol=1e-5)
    np.testing.assert_allclose(increments[0, 3:9], [0, -1, 0, 1, 0, 0], atol=1e-6)
    np.testing.assert_array_equal(increments[:, 9], actions[:, 6])


def test_relative_eef6d_uses_inclusive_se3_prefix():
    increments = np.array(
        [
            [1, 0, 0, *_IDENTITY_ROT6D, -1],
            [0, 2, 0, *_IDENTITY_ROT6D, 1],
        ],
        dtype=np.float32,
    )

    relative = libero_policy.eef6d_increments_to_relative(increments)

    np.testing.assert_allclose(relative[:, :3], [[1, 0, 0], [1, 2, 0]], atol=1e-6)
    np.testing.assert_allclose(relative[:, 3:9], np.tile(_IDENTITY_ROT6D, (2, 1)), atol=1e-6)


def test_relative_eef6d_rotates_later_translation_during_composition():
    commands = np.array(
        [[1, 0, 0, 0, 0, np.pi / 2, -1], [1, 0, 0, 0, 0, 0, 1]],
        dtype=np.float32,
    )

    relative = libero_policy.eef6d_increments_to_relative(libero_policy.libero_actions_to_eef6d(commands))

    np.testing.assert_allclose(relative[:, :3], [[1, 0, 0], [1, 1, 0]], atol=1e-6)


def test_libero_relative_eef6d_policy_transforms_actions_only():
    raw_actions = np.array([[0.1, 0, 0, 0, 0, 0, -1], [0, 0.2, 0, 0, 0, 0, 1]], dtype=np.float32)
    stored_actions = libero_policy.libero_actions_to_eef6d(raw_actions)
    sample = {
        "observation/state": np.arange(8, dtype=np.float32),
        "observation/image": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "actions": stored_actions,
        "prompt": "move",
    }

    inputs = libero_policy.LiberoInputs(
        model_type=_model.ModelType.PI05,
        relative_eef6d_actions=True,
    )(sample)
    outputs = libero_policy.LiberoOutputs(relative_eef6d_actions=True)({"actions": inputs["actions"]})

    np.testing.assert_array_equal(inputs["state"], sample["observation/state"])
    np.testing.assert_allclose(outputs["actions"], raw_actions, atol=1e-6)


def test_relative_eef6d_decode_handles_degenerate_rotation():
    relative = np.zeros((2, 10), dtype=np.float32)
    relative[:, 9] = [-1, 1]

    decoded = libero_policy.relative_eef6d_to_libero_actions(relative)

    assert decoded.shape == (2, 7)
    assert np.isfinite(decoded).all()
