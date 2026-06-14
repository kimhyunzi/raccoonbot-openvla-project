#!/usr/bin/env python3
"""
Enhanced physical RaccoonBot OpenVLA client.

Separated from the MuJoCo-only enhanced client.
This file uses the same logging/visualization system but additionally sends actions to the physical RaccoonBot.

Default real-robot execution path:
OpenVLA 7D action -> env.execute_delta_action7() -> exec_info[target_xyz/gripper_cmd] -> original RealRaccoonController IK -> robot command

Optional direct path:
OpenVLA 7D action -> logging-grade 7D-to-4DOF mapper -> robot command
Use --direct_4dof_mapping only for short, careful experiments.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from raccoon_env import SyncSimRaccoonEnv
from openvla_multicolor_client_real_robot import RealRaccoonController
from openvla_multicolor_client_enhanced import (
    CYLINDER_BODY_BY_COLOR,
    CYLINDER_COLORS,
    DEFAULT_OBJECT_X_RANGE,
    DEFAULT_OBJECT_Y_RANGE,
    DEFAULT_MIN_OBJECT_DISTANCE,
    EXECUTION_PRESETS,
    TASKS,
    RaccoonKinematics,
    TraceLogger,
    request_action_with_retry,
    resolve_target_color_task_instruction,
    sample_object_specs,
    make_default_object_specs,
    reset_multicolor_scene,
    object_specs_to_meta,
    clear_existing_images,
    get_target_object_metrics,
    update_success_meta,
    validate_obs,
    print_step_log,
    maybe_tunnel_context,
    build_server_url,
)


def validate_exec_info_for_real(exec_info: Dict[str, Any]) -> None:
    missing = [key for key in ["target_xyz", "gripper_cmd"] if key not in exec_info]
    if missing:
        raise KeyError(
            f"env.execute_delta_action7() did not return required keys for real robot: {missing}. "
            f"Available keys: {list(exec_info.keys())}"
        )


def execute_direct_4dof_on_real_robot(
    real_robot: RealRaccoonController,
    direct_4dof: Dict[str, Any],
    speed: int,
) -> Dict[str, Any]:
    if not direct_4dof.get("ik_success", False):
        raise ValueError(f"direct 4DOF mapping IK failed: {direct_4dof}")

    joint_degrees = direct_4dof.get("joint_degrees_4dof")
    if not joint_degrees or len(joint_degrees) < 4:
        raise ValueError(f"invalid direct joint_degrees_4dof: {joint_degrees}")

    # The original controller exposes this internal method. We keep it isolated here.
    real_robot._send_joint_degrees(joint_degrees[:4], speed=speed)

    gripper_state = direct_4dof.get("gripper_state")
    if gripper_state == "close":
        real_robot.close_gripper()
    elif gripper_state == "open":
        real_robot.open_gripper()
    else:
        raise ValueError(f"invalid gripper_state from direct mapping: {gripper_state}")

    return {
        "execution_mode": "direct_4dof_mapping",
        "target_xyz_m": direct_4dof.get("target_xyz_m"),
        "target_xyz_cm": direct_4dof.get("target_xyz_cm"),
        "joint_degrees": [float(v) for v in joint_degrees[:4]],
        "gripper_state": gripper_state,
    }


def safe_close_real_robot(real_robot: Optional[RealRaccoonController], go_home: bool) -> None:
    if real_robot is None:
        return
    try:
        if go_home and getattr(real_robot, "connected", False):
            real_robot.go_home(wait_seconds=0.0)
    except Exception as exc:
        print(f"[REAL_ROBOT WARN] go_home on exit failed: {exc}")

    # Original close() is often a no-op. Try common cleanup names as best effort.
    for name in ["close", "dispose", "disconnect", "stop"]:
        try:
            fn = getattr(real_robot, name, None)
            if callable(fn):
                fn()
                break
        except Exception as exc:
            print(f"[REAL_ROBOT WARN] {name} failed: {exc}")


def rollout(args: argparse.Namespace, server_url: str) -> None:
    rng = np.random.default_rng(args.seed)
    target_color, task, instruction = resolve_target_color_task_instruction(
        args.instruction, args.target_color, args.task, args.instruction_variant, rng
    )

    mode_tag = "direct4dof" if args.direct_4dof_mapping else "envbridge"
    out_dir = Path(args.output_dir) / f"real_{task}_{target_color}_{args.execution_preset}_{mode_tag}_episode_{args.episode_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_existing_images(out_dir)
    logger = TraceLogger(out_dir)

    object_specs = sample_object_specs(rng, tuple(args.object_x_range), tuple(args.object_y_range), min_distance=args.min_object_distance) if args.randomize_objects else make_default_object_specs()
    env = SyncSimRaccoonEnv(xml_path=args.xml_path, image_size=(256, 256), camera_name=args.camera_name, use_viewer=args.use_viewer)
    real_robot: Optional[RealRaccoonController] = None

    meta = {
        "task": task,
        "instruction": instruction,
        "target_color": target_color,
        "target_body_name": CYLINDER_BODY_BY_COLOR[target_color],
        "execution_preset": args.execution_preset,
        "use_real_robot": True,
        "direct_4dof_mapping": args.direct_4dof_mapping,
        "speed": args.speed,
        "settle_seconds_per_action": args.settle_seconds_per_action,
        "real_settle_seconds": args.real_settle_seconds,
        "max_delta_xyz": args.max_delta_xyz,
        "delta_scale": args.delta_scale,
        "seed": args.seed,
        "randomize_objects": args.randomize_objects,
        "success_distance_threshold": args.success_distance_threshold,
        "success_displacement_threshold": args.success_displacement_threshold,
        "success_lift_height_threshold": args.success_lift_height_threshold,
        "all_object_init_poses": object_specs_to_meta(object_specs),
        "success": False,
        "final_status": "started",
    }
    (out_dir / "rollout_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        reset_multicolor_scene(env, object_specs, target_color)
        if hasattr(env, "lockh"):
            env.lockh()

        real_robot = RealRaccoonController(
            require_ready=not args.allow_sim_only_on_hw_fail,
            home_wait_seconds=args.real_initial_wait_seconds,
            beep_on_ready=not args.no_beep_on_ready,
        )

        if hasattr(env, "debug_check_current_ee_reachable"):
            env.debug_check_current_ee_reachable()
        if args.initial_settle_seconds > 0 and hasattr(env, "settle_steps"):
            env.settle_steps(seconds=args.initial_settle_seconds)

        print(
            f"[REAL SCENE] task={task} | instruction={instruction!r} | color={target_color} | "
            f"preset={args.execution_preset} | mode={mode_tag}"
        )

        obs = env.get_observation(); validate_obs(obs)

        for step_idx in range(args.max_steps):
            step_start = time.perf_counter()
            inference_time = sim_execution_time = real_execution_time = None
            action7 = None; direct_4dof = None; frame_name = None; real_info = None

            try:
                validate_obs(obs)
                t0 = time.perf_counter()
                action7, _ = request_action_with_retry(
                    server_url, instruction, obs["image"], args.unnorm_key,
                    args.request_timeout, args.max_infer_retries, args.infer_retry_sleep,
                )
                inference_time = time.perf_counter() - t0

                direct_4dof = RaccoonKinematics.map_openvla_7d_to_raccoon_4dof(
                    obs.get("ee_pose"), action7, args.delta_scale, args.max_delta_xyz,
                    args.gripper_close_threshold, args.invert_gripper,
                )

                # Always execute the MuJoCo path first because it has clipping / projection / fallback / retry.
                t1 = time.perf_counter()
                exec_info = env.execute_delta_action7(action=action7, speed=args.speed, delta_scale=args.delta_scale, max_delta_xyz=args.max_delta_xyz)
                sim_execution_time = time.perf_counter() - t1

                if real_robot is not None and getattr(real_robot, "connected", False):
                    t2 = time.perf_counter()
                    if args.direct_4dof_mapping:
                        real_info = execute_direct_4dof_on_real_robot(real_robot, direct_4dof, speed=args.speed)
                    else:
                        validate_exec_info_for_real(exec_info)
                        real_info = real_robot.execute_from_exec_info(exec_info, speed=args.speed)
                        real_info["execution_mode"] = "env_execute_delta_action7_bridge"
                    real_execution_time = time.perf_counter() - t2

                    wait_sec = args.real_settle_seconds if args.real_settle_seconds is not None else args.settle_seconds_per_action
                    if wait_sec > 0:
                        time.sleep(wait_sec)

                if args.settle_seconds_per_action > 0 and hasattr(env, "settle_steps"):
                    env.settle_steps(seconds=args.settle_seconds_per_action)

                obs = env.get_observation(); validate_obs(obs)
                frame_name = f"frame_{step_idx:06d}.png"
                Image.fromarray(obs["image"]).save(out_dir / frame_name)
                metrics = get_target_object_metrics(env, target_color, object_specs, obs)

                row = {
                    "step": step_idx,
                    "task": task,
                    "instruction": instruction,
                    "target_color": target_color,
                    "execution_preset": args.execution_preset,
                    "real_execution_mode": mode_tag,
                    "raw_action_7d": action7,
                    "direct_4dof_target_xyz_m": direct_4dof.get("target_xyz_m"),
                    "direct_4dof_joint_degrees": direct_4dof.get("joint_degrees_4dof"),
                    "direct_4dof_gripper_state": direct_4dof.get("gripper_state"),
                    "direct_4dof_ik_success": direct_4dof.get("ik_success"),
                    "direct_4dof_reason": direct_4dof.get("reason"),
                    "env_applied_delta_xyz": exec_info.get("applied_delta_xyz"),
                    "env_final_delta_xyz": exec_info.get("final_delta_xyz"),
                    "env_actual_move_xyz": exec_info.get("actual_move_xyz"),
                    "env_target_xyz": exec_info.get("target_xyz"),
                    "gripper_cmd": exec_info.get("gripper_cmd"),
                    "gripper_action": exec_info.get("gripper_action"),
                    "fallback_used": exec_info.get("fallback_used"),
                    "retry_count": exec_info.get("retry_count"),
                    "blocked_by_height": exec_info.get("gripper_blocked_by_height"),
                    "real_robot": real_info,
                    "inference_time_sec": inference_time,
                    "sim_execution_time_sec": sim_execution_time,
                    "real_execution_time_sec": real_execution_time,
                    "total_step_time_sec": time.perf_counter() - step_start,
                    **metrics,
                    "frame": frame_name,
                    "status": "ok",
                }
                logger.log_step(row); print_step_log(row)

            except Exception as exc:
                try:
                    fail_obs = env.get_observation()
                    if "image" in fail_obs:
                        frame_name = f"frame_{step_idx:06d}_failed.png"
                        Image.fromarray(fail_obs["image"]).save(out_dir / frame_name)
                        obs = fail_obs
                except Exception:
                    pass

                row = {
                    "step": step_idx,
                    "task": task,
                    "instruction": instruction,
                    "target_color": target_color,
                    "raw_action_7d": action7,
                    "direct_4dof": direct_4dof,
                    "real_robot": real_info,
                    "inference_time_sec": inference_time,
                    "sim_execution_time_sec": sim_execution_time,
                    "real_execution_time_sec": real_execution_time,
                    "total_step_time_sec": time.perf_counter() - step_start,
                    "error": repr(exc),
                    "frame": frame_name,
                    "status": "fail",
                }
                logger.log_step(row)
                print(f"[{step_idx:03d}] REAL FAIL | {exc}")
                if args.stop_on_fail:
                    meta["final_status"] = f"failed at step {step_idx}"
                    break

        if meta["final_status"] == "started":
            meta["final_status"] = "max_steps reached"
        update_success_meta(meta, logger, task, args)

    except KeyboardInterrupt:
        print("\n[STOP] interrupted by user")
        meta["final_status"] = "interrupted"
    finally:
        safe_close_real_robot(real_robot, args.real_go_home_on_exit)
        env.close()
        summary = logger.finalize(meta)
        print(f"[REAL SUMMARY] saved to {out_dir}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def apply_execution_preset(args: argparse.Namespace) -> None:
    if args.execution_preset == "custom":
        return
    p = EXECUTION_PRESETS[args.execution_preset]
    args.speed = p["speed"]
    args.settle_seconds_per_action = p["settle_seconds_per_action"]
    args.max_delta_xyz = p["max_delta_xyz"]
    args.delta_scale = p["delta_scale"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--xml_path", type=str, default="Raccoon_colored_cylinder.xml")
    p.add_argument("--server_url", type=str, default=None)
    p.add_argument("--unnorm_key", type=str, default="raccoon_pick_place")
    p.add_argument("--task", type=str, default="auto", choices=["auto", *TASKS])
    p.add_argument("--instruction", type=str, default=None)
    p.add_argument("--instruction_variant", type=int, default=-1)
    p.add_argument("--target_color", type=str, default="auto", choices=["auto", "random", *CYLINDER_COLORS])
    p.add_argument("--output_dir", type=str, default="rollout_outputs_real_enhanced")
    p.add_argument("--episode_id", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--camera_name", type=str, default="front_view")
    p.add_argument("--use_viewer", action="store_true")
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--execution_preset", type=str, default="safe", choices=["custom", "safe", "balanced", "fast"])
    p.add_argument("--speed", type=int, default=70)
    p.add_argument("--settle_seconds_per_action", type=float, default=0.8)
    p.add_argument("--initial_settle_seconds", type=float, default=0.3)
    p.add_argument("--delta_scale", type=float, default=1.0)
    p.add_argument("--max_delta_xyz", type=float, default=0.005)
    p.add_argument("--request_timeout", type=float, default=60.0)
    p.add_argument("--max_infer_retries", type=int, default=2)
    p.add_argument("--infer_retry_sleep", type=float, default=0.5)

    p.add_argument("--randomize_objects", action="store_true", default=True)
    p.add_argument("--no_randomize_objects", dest="randomize_objects", action="store_false")
    p.add_argument("--object_x_range", type=float, nargs=2, default=DEFAULT_OBJECT_X_RANGE)
    p.add_argument("--object_y_range", type=float, nargs=2, default=DEFAULT_OBJECT_Y_RANGE)
    p.add_argument("--min_object_distance", type=float, default=DEFAULT_MIN_OBJECT_DISTANCE)

    p.add_argument("--direct_4dof_mapping", action="store_true", help="Use direct logging-grade 7D-to-4DOF mapping for real robot. Default is safer env bridge.")
    p.add_argument("--allow_sim_only_on_hw_fail", action="store_true")
    p.add_argument("--real_initial_wait_seconds", type=float, default=5.0)
    p.add_argument("--real_settle_seconds", type=float, default=None)
    p.add_argument("--real_go_home_on_exit", action="store_true")
    p.add_argument("--no_beep_on_ready", action="store_true")
    p.add_argument("--gripper_close_threshold", type=float, default=0.5)
    p.add_argument("--invert_gripper", action="store_true")

    p.add_argument("--success_distance_threshold", type=float, default=0.035)
    p.add_argument("--success_displacement_threshold", type=float, default=0.030)
    p.add_argument("--success_lift_height_threshold", type=float, default=0.025)
    p.add_argument("--stop_on_fail", action="store_true")

    p.add_argument("--use_ssh_tunnel", action="store_true")
    p.add_argument("--ssh_host", type=str, default=os.environ.get("OPENVLA_SSH_HOST", ""))
    p.add_argument("--ssh_port", type=int, default=int(os.environ.get("OPENVLA_SSH_PORT", "22")))
    p.add_argument("--ssh_user", type=str, default=os.environ.get("OPENVLA_SSH_USER", ""))
    p.add_argument("--ssh_password", type=str, default=None)
    p.add_argument("--ssh_ask_password", action="store_true")
    p.add_argument("--remote_server_host", type=str, default=os.environ.get("OPENVLA_REMOTE_HOST", "127.0.0.1"))
    p.add_argument("--remote_server_port", type=int, default=int(os.environ.get("OPENVLA_REMOTE_PORT", "8000")))
    p.add_argument("--local_server_host", type=str, default="127.0.0.1")
    p.add_argument("--local_server_port", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_execution_preset(args)
    with maybe_tunnel_context(args) as tunnel:
        server_url = build_server_url(args, tunnel)
        if tunnel is not None:
            print(f"[SSH] {args.local_server_host}:{tunnel.local_bind_port} -> {args.remote_server_host}:{args.remote_server_port}")
        rollout(args, server_url)


if __name__ == "__main__":
    main()
