"""RoboCasa365 observation adapter for the released XR-1 processor.

RLinf environments expose one current RGB frame per camera and a flat state
vector.  The released XR-1 checkpoint consumes three one-frame videos, a
four-frame state history, and a 60D state tensor.  This adapter keeps that
conversion in the rollout policy boundary and emits batch-first tensors so the
RLinf trajectory/replay code can split them safely.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
from PIL import Image


class XR1RoboCasa365ObservationAdapter:
    """Convert RLinf's standard RoboCasa365 observations to XR-1 inputs."""

    def __init__(
        self,
        processor: Any,
        *,
        robot_type: str = "robocasa365",
        state_dim: int = 60,
        state_length: int = 4,
        video_history: int = 1,
        history_interval: int = 2,
        crop_ratio: float = 0.95,
        action_horizon: int = 16,
    ) -> None:
        if state_dim <= 0 or state_length <= 0 or video_history <= 0:
            raise ValueError("state_dim, state_length, and video_history must be positive")
        self.processor = processor
        self.robot_type = str(robot_type)
        self.state_dim = int(state_dim)
        self.state_length = int(state_length)
        self.video_history = int(video_history)
        self.history_interval = int(history_interval)
        self.crop_ratio = float(crop_ratio)
        self._state_queues: list[deque[np.ndarray]] = []
        self._image_history: dict[str, list[deque[np.ndarray]]] = {}
        self._last_descriptions: list[str] = []
        self.action_horizon = int(action_horizon)

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _to_frame(value: Any) -> Image.Image:
        frame = np.asarray(value)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim != 3 or frame.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Expected an HWC image, got shape {frame.shape}")
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[..., :3]
        return Image.fromarray(np.ascontiguousarray(frame))

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        """Match evaluator: center crop, then resize back to source size."""
        if self.crop_ratio >= 1.0:
            return frame
        h, w = frame.shape[:2]
        ch, cw = max(1, int(h * self.crop_ratio)), max(1, int(w * self.crop_ratio))
        top, left = (h - ch) // 2, (w - cw) // 2
        cropped = Image.fromarray(np.ascontiguousarray(frame[top : top + ch, left : left + cw]))
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        return np.asarray(cropped.resize((w, h), resampling), dtype=np.uint8)

    @staticmethod
    def _sample_history(history: deque[np.ndarray], length: int, interval: int) -> list[np.ndarray]:
        items = list(history)
        return [items[max(0, len(items) - 1 - (length - 1 - i) * interval)] for i in range(length)]

    def _camera_frames(self, value: Any, index: int, *, extra_view: bool = False) -> list[Image.Image]:
        array = self._as_numpy(value)
        if array.ndim == 3:
            sample = array
        elif array.ndim == 4:
            # Standard RLinf camera tensors are [B,H,W,C].  A single-env
            # temporal tensor [T,H,W,C] is also accepted when B == 1.
            if extra_view:
                sample = array[index, 0] if array.shape[0] > index else array[0]
            else:
                sample = array[index]
        elif array.ndim == 5:
            # [B,T,H,W,C] for a temporal camera or [B,N,H,W,C] for views.
            sample = array[index]
            if extra_view:
                sample = sample[0]
        elif array.ndim == 6:
            # [B,T,N,H,W,C] temporal extra-camera views.
            sample = array[index, :, 0]
        else:
            raise ValueError(f"Unsupported camera tensor shape {array.shape}")

        if sample.ndim == 3:
            frames = [sample]
        elif sample.ndim == 4:
            frames = [sample[i] for i in range(sample.shape[0])]
        else:
            raise ValueError(f"Unsupported per-sample camera shape {sample.shape}")
        if not frames:
            raise ValueError("Camera history cannot be empty")

        return [self._to_frame(frame) for frame in frames]

    def _state_history(self, states: Any, batch_size: int) -> np.ndarray:
        array = self._as_numpy(states).astype(np.float32, copy=False)
        if array.ndim == 1:
            array = array[None]
        if array.ndim == 2:
            array = array[:, None, :]
        if array.ndim != 3 or array.shape[0] != batch_size:
            raise ValueError(
                "RoboCasa365 states must be [B,D] or [B,T,D], "
                f"got {array.shape} for batch_size={batch_size}"
            )

        width = min(int(array.shape[-1]), self.state_dim)
        padded = np.zeros(
            (batch_size, array.shape[1], self.state_dim), dtype=np.float32
        )
        padded[..., :width] = array[..., :width]
        if padded.shape[1] < self.state_length:
            padded = np.concatenate(
                [
                    np.repeat(padded[:, :1], self.state_length - padded.shape[1], axis=1),
                    padded,
                ],
                axis=1,
            )
        return np.ascontiguousarray(padded[:, -self.state_length :])

    @staticmethod
    def _pack_video_tensor(
        data: dict[str, Any],
        key: str,
        lens_key: str,
        batch_size: int,
        items_per_sample: int | None = None,
    ) -> None:
        tensor = data.get(key)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
            return
        if tensor.shape[0] % batch_size != 0:
            raise ValueError(
                f"Processor field '{key}' has leading dim {tensor.shape[0]}, "
                f"which is not divisible by batch size {batch_size}"
            )
        per_sample = (
            int(items_per_sample)
            if items_per_sample is not None
            else int(tensor.shape[0] // batch_size)
        )
        if tensor.shape[0] != batch_size * per_sample:
            raise ValueError(
                f"Cannot pack '{key}': leading dim {tensor.shape[0]} != "
                f"{batch_size}*{per_sample}"
            )
        data[key] = tensor.reshape(batch_size, per_sample, *tensor.shape[1:])
        data[lens_key] = torch.full(
            (batch_size,),
            per_sample,
            dtype=torch.int64,
            device=tensor.device,
        )

    def __call__(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        main_images = env_obs.get("main_images")
        wrist_images = env_obs.get("wrist_images")
        extra_images = env_obs.get("extra_view_images")
        states = env_obs.get("states")
        descriptions = env_obs.get("task_descriptions")
        if main_images is None or wrist_images is None or states is None:
            raise KeyError(
                "XR-1 RoboCasa365 adapter requires main_images, wrist_images, "
                "states, and task_descriptions"
            )

        main_array = self._as_numpy(main_images)
        if main_array.ndim == 3:
            main_array = main_array[None]
        batch_size = int(main_array.shape[0])
        if descriptions is None:
            descriptions = [""] * batch_size
        descriptions = [str(item) for item in list(descriptions)]
        if len(descriptions) != batch_size:
            raise ValueError(
                f"task_descriptions has {len(descriptions)} entries for batch {batch_size}"
            )

        if extra_images is None:
            raise KeyError(
                "XR-1 RoboCasa365 checkpoint expects the right-camera "
                "extra_view_images input"
            )

        # RLinf supplies one observation at each action-chunk boundary. Keep
        # the same temporal context as the official evaluator (4 frames,
        # interval 2) instead of repeatedly presenting only the current frame.
        if len(self._state_queues) != batch_size:
            self._state_queues = [deque(maxlen=(self.state_length - 1) * self.history_interval + 1) for _ in range(batch_size)]
            self._image_history = {key: [deque(maxlen=(self.video_history - 1) * self.history_interval + 1) for _ in range(batch_size)] for key in ("main", "right", "wrist")}
            self._last_descriptions = [""] * batch_size
        conversations: list[list[dict[str, Any]]] = []
        for index in range(batch_size):
            if descriptions[index] != self._last_descriptions[index]:
                self._state_queues[index].clear()
                for queue in self._image_history.values():
                    queue[index].clear()
                self._last_descriptions[index] = descriptions[index]
            main_frame = self._camera_frames(main_images, index)[-1]
            right_frame = self._camera_frames(extra_images, index, extra_view=True)[-1]
            wrist_frame = self._camera_frames(wrist_images, index)[-1]
            state_sample = self._as_numpy(states)[index]
            state_frames = state_sample if state_sample.ndim == 2 else [state_sample]
            for state_frame in state_frames:
                self._state_queues[index].append(np.asarray(state_frame, dtype=np.float32).copy())
            frame_values = {
                "main": self._camera_frames(main_images, index),
                "right": self._camera_frames(extra_images, index, extra_view=True),
                "wrist": self._camera_frames(wrist_images, index),
            }
            for key, frames in frame_values.items():
                for frame in frames:
                    self._image_history[key][index].append(np.asarray(frame))
            main_video = [self._to_frame(self._crop(x)) for x in self._sample_history(self._image_history["main"][index], self.video_history, self.history_interval)]
            right_video = [self._to_frame(self._crop(x)) for x in self._sample_history(self._image_history["right"][index], self.video_history, self.history_interval)]
            wrist_video = [self._to_frame(self._crop(x)) for x in self._sample_history(self._image_history["wrist"][index], self.video_history, self.history_interval)]
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Left camera: "},
                            {"type": "video", "video": main_video},
                            {"type": "text", "text": "\nRight camera: "},
                            {"type": "video", "video": right_video},
                            {"type": "text", "text": "\nWrist camera: "},
                            {"type": "video", "video": wrist_video},
                            {
                                "type": "text",
                                "text": f"\n\nGenerate robot actions for the task:\n{descriptions[index]} /no_cot",
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "<cot></cot>"}],
                    },
                ]
            )

        state = np.stack([
            np.stack(self._sample_history(self._state_queues[i], self.state_length, self.history_interval), axis=0)
            for i in range(batch_size)
        ], axis=0)
        width = min(int(state.shape[-1]), self.state_dim)
        state_padded = np.zeros((batch_size, self.state_length, self.state_dim), dtype=np.float32)
        state_padded[..., :width] = state[..., :width]
        state = state_padded
        processed = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            do_resize=False,
            state=state,
            robot_type=self.robot_type,
        )
        data = {
            key: value
            for key, value in dict(processed).items()
            if isinstance(value, torch.Tensor)
        }
        if "action_mask" not in data:
            raise KeyError("XR-1 processor did not return action_mask")
        action_mask = data["action_mask"]
        if action_mask.ndim == 2:
            action_mask = action_mask.unsqueeze(0)
        if action_mask.shape[0] == 1:
            action_mask = action_mask.expand(batch_size, -1, -1).clone()
        if action_mask.shape[0] != batch_size:
            raise ValueError(
                f"Processor action_mask batch {action_mask.shape[0]} != {batch_size}"
            )
        if action_mask.shape[1] != self.action_horizon:
            raise ValueError(
                f"XR-1 action horizon mismatch: processor returned {action_mask.shape[1]}, "
                f"configured {self.action_horizon}"
            )
        data["action_mask"] = action_mask
        # MiBotProcessor emits robot state in checkpoint precision (bf16), and
        # the deployment server applies the same floating-point conversion
        # before calling the model.
        data["state"] = torch.as_tensor(state, dtype=torch.bfloat16)
        data["action"] = torch.zeros(
            (batch_size, self.action_horizon, action_mask.shape[-1]),
            dtype=action_mask.dtype,
        )

        # The processor flattens all video patches across samples. Pack those
        # fields with explicit lengths so RLinf can stack/split trajectories;
        # the policy restores them immediately before the VLM forward.
        self._pack_video_tensor(
            data,
            "video_grid_thw",
            "video_grid_thw_lens",
            batch_size,
            items_per_sample=3,
        )
        self._pack_video_tensor(
            data,
            "pixel_values_videos",
            "pixel_values_videos_lens",
            batch_size,
        )
        return data


__all__ = ["XR1RoboCasa365ObservationAdapter"]
