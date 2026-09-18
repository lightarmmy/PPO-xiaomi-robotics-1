#!/usr/bin/env python
"""Direct XR-1 RoboCasa365 rollout without Ray/RLinf.

This is the control-path acceptance test for PPO integration.  It intentionally
uses the same processor, state construction, history, decode_action and
convert_action calls as eval_robocasa365, then executes the decoded action in a
single native RoboCasa environment.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def quat_xyzw_to_axis_angle(quaternion):
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.size != 4:
        return np.zeros(3, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    q = q / norm
    if q[3] < 0:
        q = -q
    xyz = q[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, np.clip(q[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)


def observation_to_state(obs):
    state = np.concatenate([
        np.asarray(obs["state.end_effector_position_relative"], dtype=np.float32).reshape(-1),
        quat_xyzw_to_axis_angle(obs["state.end_effector_rotation_relative"]),
        np.asarray(obs["state.gripper_qpos"], dtype=np.float32).reshape(-1),
        np.asarray(obs["state.base_position"], dtype=np.float32).reshape(-1),
        quat_xyzw_to_axis_angle(obs["state.base_rotation"]),
    ]).astype(np.float32)
    if state.shape != (14,):
        raise ValueError(f"expected official 14D state, got {state.shape}")
    return state


def center_crop(image, ratio=0.95):
    image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if ratio >= 1.0:
        return image
    width, height = image.size
    crop_width = max(1, int(width * ratio))
    crop_height = max(1, int(height * ratio))
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    image = image.crop((left, top, left + crop_width, top + crop_height))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return image.resize((width, height), resampling)


def build_messages(images, instruction):
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Left camera: "},
            {"type": "video", "video": [center_crop(x) for x in images[0]]},
            {"type": "text", "text": "\nRight camera: "},
            {"type": "video", "video": [center_crop(x) for x in images[1]]},
            {"type": "text", "text": "\nWrist camera: "},
            {"type": "video", "video": [center_crop(x) for x in images[2]]},
            {"type": "text", "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot"},
        ],
    }, {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]}]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", default="TurnOnMicrowave")
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--seed", type=int, default=87)
    parser.add_argument("--horizon", type=int, default=450)
    parser.add_argument("--replan-steps", type=int, default=16)
    parser.add_argument("--obs-history", type=int, default=4)
    parser.add_argument("--obs-interval", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from transformers import AutoModel, AutoProcessor
    import gymnasium as gym
    import robocasa  # noqa: F401 -- registers robocasa/* Gym environments
    from robocasa.utils.env_utils import convert_action

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", local_files_only=True,
    ).cuda().eval()
    env = gym.make(f"robocasa/{args.task}", split=args.split, seed=args.seed)
    obs = env.reset(seed=args.seed)
    if isinstance(obs, tuple):
        obs = obs[0]
    instruction = obs["annotation.human.task_description"]
    queues = [collections.deque(maxlen=(args.obs_history - 1) * args.obs_interval + 1) for _ in range(3)]
    state_queue = collections.deque(maxlen=(args.obs_history - 1) * args.obs_interval + 1)
    def current_images(o):
        return [np.ascontiguousarray(o[k], dtype=np.uint8) for k in (
            "video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")]
    for q, image in zip(queues, current_images(obs)):
        q.append(image)
    state_queue.append(observation_to_state(obs))
    action_plan = collections.deque()
    success = False
    records = []
    for step in range(args.horizon):
        if not action_plan:
            states = [list(state_queue)[max(0, len(state_queue)-1-(args.obs_history-1-i)*args.obs_interval)] for i in range(args.obs_history)]
            images = [[list(q)[max(0, len(q)-1-(args.obs_history-1-i)*args.obs_interval)] for i in range(args.obs_history)] for q in queues]
            state = np.zeros((1, args.obs_history, 60), dtype=np.float32)
            state[0, :, :14] = np.asarray(states)
            inputs = processor.apply_chat_template(
                build_messages(images, instruction), tokenize=True, return_dict=True,
                return_tensors="pt", padding=True, do_resize=False,
                state=state, robot_type="robocasa365",
            )
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in dict(inputs).items()}
            inputs["task_id"] = "robocasa365"
            output = model(**inputs)
            actions = getattr(output, "actions", output)
            actions = processor.decode_action(actions, robot_type="robocasa365")[0, :, :12]
            action_plan.extend(actions[: args.replan_steps].float().cpu().numpy())
        action = np.asarray(action_plan.popleft(), dtype=np.float32)
        obs, _, done, truncated, info = env.step(convert_action(action))
        success = bool(info.get("success", False))
        for q, image in zip(queues, current_images(obs)):
            q.append(image)
        state_queue.append(observation_to_state(obs))
        if step == 0 or success or done or truncated or step % 25 == 24:
            logging.info("step=%d success=%s done=%s truncated=%s", step + 1, success, done, truncated)
        records.append({"step": step + 1, "success": success, "done": bool(done), "truncated": bool(truncated)})
        if success or done or truncated:
            break
    result = {"task": args.task, "split": args.split, "seed": args.seed, "success": success, "steps": len(records), "records": records}
    Path(args.output).write_text(json.dumps(result, indent=2))
    env.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
