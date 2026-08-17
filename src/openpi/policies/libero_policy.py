import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model as _model

_LIBERO_ACTION_DIM = 7
_EEF6D_ACTION_DIM = 10


def _matrix_to_rotation6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix)[..., :2, :].reshape(*matrix.shape[:-2], 6)


def _rotation6d_to_matrix(rotation6d: np.ndarray) -> np.ndarray:
    """Decode first-two-row rotation6d, including finite fallback for model outputs."""
    rotation6d = np.asarray(rotation6d)
    first_raw, second_raw = rotation6d[..., :3], rotation6d[..., 3:6]

    first_norm = np.linalg.norm(first_raw, axis=-1, keepdims=True)
    first_valid = np.isfinite(first_norm) & (first_norm > 1e-8)
    first = np.broadcast_to(np.array([1, 0, 0], dtype=rotation6d.dtype), first_raw.shape).copy()
    np.divide(first_raw, first_norm, out=first, where=first_valid)

    second_orthogonal = second_raw - np.sum(first * second_raw, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)
    second_valid = np.isfinite(second_norm) & (second_norm > 1e-8)

    fallback_axis = np.eye(3, dtype=rotation6d.dtype)[np.argmin(np.abs(first), axis=-1)]
    fallback_second = fallback_axis - np.sum(first * fallback_axis, axis=-1, keepdims=True) * first
    fallback_second /= np.linalg.norm(fallback_second, axis=-1, keepdims=True)
    second = fallback_second.copy()
    np.divide(second_orthogonal, second_norm, out=second, where=second_valid)

    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-2)


def _pose9_to_matrix(pose9: np.ndarray) -> np.ndarray:
    pose9 = np.asarray(pose9)
    matrix = np.zeros((*pose9.shape[:-1], 4, 4), dtype=pose9.dtype)
    matrix[..., :3, :3] = _rotation6d_to_matrix(pose9[..., 3:9])
    matrix[..., :3, 3] = pose9[..., :3]
    matrix[..., 3, 3] = 1
    return matrix


def _matrix_to_pose9(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[..., :3, 3], _matrix_to_rotation6d(matrix[..., :3, :3])], axis=-1)


def libero_actions_to_eef6d(actions: np.ndarray) -> np.ndarray:
    """Encode LIBERO delta commands as command-space SE(3) increments with row rotation6d."""
    actions = np.asarray(actions)
    if actions.shape[-1] != _LIBERO_ACTION_DIM:
        raise ValueError(f"LIBERO actions must have width {_LIBERO_ACTION_DIM}, got {actions.shape[-1]}.")
    rotation = Rotation.from_rotvec(actions[..., 3:6]).as_matrix()
    return np.concatenate([actions[..., :3], _matrix_to_rotation6d(rotation), actions[..., 6:7]], axis=-1).astype(
        np.float32, copy=False
    )


def eef6d_increments_to_relative(increments: np.ndarray) -> np.ndarray:
    """Inclusive SE(3) prefix: ten stored commands produce ten model targets."""
    increments = np.asarray(increments)
    if increments.ndim < 2 or increments.shape[-1] != _EEF6D_ACTION_DIM:
        raise ValueError(
            f"EEF6D increment chunks must have shape (..., H, {_EEF6D_ACTION_DIM}), got {increments.shape}."
        )
    increment_matrices = _pose9_to_matrix(increments[..., :9])
    relative_matrices = np.empty_like(increment_matrices)
    accumulated = np.broadcast_to(
        np.eye(4, dtype=increment_matrices.dtype), increment_matrices.shape[:-3] + (4, 4)
    ).copy()
    for step in range(increment_matrices.shape[-3]):
        accumulated = accumulated @ increment_matrices[..., step, :, :]
        relative_matrices[..., step, :, :] = accumulated
    return np.concatenate([_matrix_to_pose9(relative_matrices), increments[..., 9:10]], axis=-1).astype(
        np.float32, copy=False
    )


def relative_eef6d_to_libero_actions(relative_actions: np.ndarray) -> np.ndarray:
    """Invert inclusive relative targets back to executable LIBERO delta commands."""
    relative_actions = np.asarray(relative_actions)
    if relative_actions.ndim < 2 or relative_actions.shape[-1] != _EEF6D_ACTION_DIM:
        raise ValueError(
            f"Relative EEF6D chunks must have shape (..., H, {_EEF6D_ACTION_DIM}), got {relative_actions.shape}."
        )
    relative_matrices = _pose9_to_matrix(relative_actions[..., :9])
    increment_matrices = np.empty_like(relative_matrices)
    increment_matrices[..., 0, :, :] = relative_matrices[..., 0, :, :]
    if relative_matrices.shape[-3] > 1:
        increment_matrices[..., 1:, :, :] = (
            np.linalg.inv(relative_matrices[..., :-1, :, :]) @ relative_matrices[..., 1:, :, :]
        )
    rotation_vectors = Rotation.from_matrix(increment_matrices[..., :3, :3].reshape(-1, 3, 3)).as_rotvec()
    rotation_vectors = rotation_vectors.reshape(*increment_matrices.shape[:-2], 3)
    return np.concatenate(
        [increment_matrices[..., :3, 3], rotation_vectors, relative_actions[..., 9:10]], axis=-1
    ).astype(np.float32, copy=False)


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    relative_eef6d_actions: bool = False

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = parse_image(data["observation/image"])
        wrist_image = parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = eef6d_increments_to_relative(actions) if self.relative_eef6d_actions else actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    relative_eef6d_actions: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        actions = np.asarray(data["actions"])
        if self.relative_eef6d_actions:
            return {"actions": relative_eef6d_to_libero_actions(actions[..., :_EEF6D_ACTION_DIM])}
        return {"actions": actions[..., :_LIBERO_ACTION_DIM]}
