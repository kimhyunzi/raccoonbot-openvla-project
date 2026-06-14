#!/usr/bin/env python3
"""
Enhanced MuJoCo-only OpenVLA client for RaccoonBot.

This file is intentionally separated from the physical robot client.
It keeps the original MuJoCo execution path but adds:
- trained multi-task instruction templates
- robust inference retry
- logging-grade 7D -> 4DOF mapping evidence
- JSONL/CSV/summary logs
- timing and distance visualizations
- configurable execution-speed presets

Requires the original project file openvla_multicolor_client.py in the same directory.
"""

import argparse
import csv
import json
import math
import os
import time
from contextlib import nullcontext
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from sshtunnel import SSHTunnelForwarder

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from raccoon_env import SyncSimRaccoonEnv
from openvla_multicolor_client import (
    CYLINDER_BODY_BY_COLOR,
    CYLINDER_COLORS,
    DEFAULT_OBJECT_X_RANGE,
    DEFAULT_OBJECT_Y_RANGE,
    DEFAULT_MIN_OBJECT_DISTANCE,
    request_action,
    sample_object_specs,
    make_default_object_specs,
    reset_multicolor_scene,
    object_specs_to_meta,
    clear_existing_images,
)

INSTRUCTION_TEMPLATES_BY_TASK = {
    "grasp": [
        "grasp the {color} cylinder",
    ],
    "lift": [
        "lift the {color} cylinder",
        "pick up the {color} cylinder",
        "raise the {color} cylinder",
        "grasp and lift the {color} cylinder",
        "pick the {color} cylinder up from the table",
    ],
    "push": [
        "push the {color} cylinder forward",
        "move the {color} cylinder forward",
        "slide the {color} cylinder forward",
        "nudge the {color} cylinder away from the robot",
        "push the {color} object away from the robot",
    ],
    "pick_place": [
        "pick and place the {color} cylinder",
        "move the {color} cylinder to the side",
        "pick up the {color} cylinder and place it nearby",
        "relocate the {color} cylinder to the side",
        "grasp the {color} cylinder and put it down on the side",
    ],
}
TASKS = tuple(INSTRUCTION_TEMPLATES_BY_TASK.keys())

# These presets affect client-side motion execution only. They do not speed up OpenVLA inference.
EXECUTION_PRESETS = {
    "safe": {"speed": 70, "settle_seconds_per_action": 0.80, "max_delta_xyz": 0.005, "delta_scale": 1.0},
    "balanced": {"speed": 80, "settle_seconds_per_action": 0.45, "max_delta_xyz": 0.008, "delta_scale": 1.0},
    "fast": {"speed": 95, "settle_seconds_per_action": 0.20, "max_delta_xyz": 0.012, "delta_scale": 1.0},
}


def safe_float_list(values: Any, max_len: Optional[int] = None) -> List[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if max_len is not None:
        arr = arr[:max_len]
    return [float(v) for v in arr.tolist()]


def infer_color_from_instruction(instruction: Optional[str]) -> Optional[str]:
    if not instruction:
        return None
    text = instruction.lower()
    matches = [c for c in CYLINDER_COLORS if f" {c} " in f" {text} "]
    if len(matches) > 1:
        raise ValueError(f"instruction에 여러 색상이 들어 있습니다: {matches} | instruction={instruction!r}")
    return matches[0] if matches else None


def infer_task_from_instruction(instruction: Optional[str]) -> Optional[str]:
    if not instruction:
        return None
    text = instruction.lower()
    if any(k in text for k in ["pick and place", "place it", "place the", "relocate", "nearby", "to the side", "on the side"]):
        return "pick_place"
    if any(k in text for k in ["push", "slide", "nudge", "away from the robot", "forward"]):
        return "push"
    if any(k in text for k in ["lift", "raise", "up from the table", "grasp and lift"]):
        return "lift"
    if any(k in text for k in ["grasp", "grab", "pick up", "hold", "take"]):
        return "grasp"
    return None


def resolve_target_color_task_instruction(
    instruction: Optional[str],
    target_color_arg: str,
    task_arg: str,
    instruction_variant: int,
    rng: np.random.Generator,
) -> Tuple[str, str, str]:
    instruction_color = infer_color_from_instruction(instruction)

    if instruction_color is not None:
        target_color = instruction_color
        if target_color_arg in CYLINDER_COLORS and target_color_arg != instruction_color:
            raise ValueError(
                f"--instruction 색상({instruction_color})과 --target_color({target_color_arg})가 다릅니다."
            )
    elif target_color_arg in CYLINDER_COLORS:
        target_color = target_color_arg
    elif target_color_arg in ("auto", "random"):
        target_color = str(rng.choice(CYLINDER_COLORS))
    else:
        raise ValueError(f"지원하지 않는 --target_color 값입니다: {target_color_arg}")

    if task_arg == "auto":
        inferred = infer_task_from_instruction(instruction)
        if inferred is None and not instruction:
            raise ValueError("--task auto에서 --instruction이 없으면 task를 추론할 수 없습니다. --task를 명시하세요.")
        task = inferred or "lift"
    else:
        task = task_arg

    if instruction and instruction.strip():
        return target_color, task, instruction.strip()

    templates = INSTRUCTION_TEMPLATES_BY_TASK[task]
    template = templates[instruction_variant % len(templates)] if instruction_variant >= 0 else str(rng.choice(templates))
    return target_color, task, template.format(color=target_color)


def request_action_with_retry(
    server_url: str,
    instruction: str,
    image_rgb: np.ndarray,
    unnorm_key: Optional[str],
    timeout: float,
    max_retries: int,
    retry_sleep: float,
) -> Tuple[List[float], Dict[str, Any]]:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = request_action(server_url, instruction, image_rgb, unnorm_key, timeout=timeout)
            if "action" not in response:
                raise KeyError(f"server response has no 'action': {response.keys()}")
            action7 = safe_float_list(response["action"])
            if len(action7) < 7:
                raise ValueError(f"server returned {len(action7)} action dims, expected >=7")
            return action7[:7], response
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                print(f"[INFER RETRY] {attempt + 1}/{max_retries} | {exc}")
                time.sleep(retry_sleep)
    assert last_exc is not None
    raise last_exc


class RaccoonKinematics:
    """Logging-grade explicit OpenVLA 7D -> RaccoonBot 4DOF mapping."""

    L1, L2, L3, L4 = 8.25, 10.0, 10.0, 8.0
    TH1_LIMIT = (-120.0, 120.0)
    TH2_LIMIT = (-90.0, 30.0)
    TH3_LIMIT = (-150.0, 0.0)
    TH4_LIMIT = (-180.0, 180.0)

    @classmethod
    def calculate_4dof_ik_from_xyz_m(cls, target_xyz_m: Sequence[float]) -> Optional[List[float]]:
        tx, ty, tz = [float(v) for v in target_xyz_m[:3]]
        return cls.calculate_4dof_ik_from_xyz_cm(tx * 100.0, ty * 100.0, tz * 100.0)

    @classmethod
    def calculate_4dof_ik_from_xyz_cm(cls, x_cm: float, y_cm: float, z_cm: float) -> Optional[List[float]]:
        if not ((-28.0 <= x_cm <= 28.0) and (-15.0 <= y_cm <= 28.0) and (0.0 <= z_cm <= 36.25)):
            return None

        # Same convention as the current real_robot client. Treat as logging-grade until physically validated.
        x, y, z = y_cm, -x_cm, z_cm
        th1 = math.atan2(y, x)
        c1, s1 = math.cos(th1), math.sin(th1)
        wx = x - cls.L4 * c1
        wy = y - cls.L4 * s1
        wz = z - cls.L1

        c3 = (wx * wx + wy * wy + wz * wz - cls.L2 ** 2 - cls.L3 ** 2) / (2.0 * cls.L2 * cls.L3)
        if c3 < -1.0001 or c3 > 1.0001:
            return None
        c3 = float(np.clip(c3, -1.0, 1.0))
        s3_abs = math.sqrt(max(0.0, 1.0 - c3 * c3))
        th1_deg = math.degrees(th1)

        for s3 in (-s3_abs, s3_abs):
            th3 = math.atan2(s3, c3)
            m1 = c3 * cls.L3 + cls.L2
            m2 = wz
            m3 = s3 * cls.L3
            m4 = c1 * wx + s1 * wy
            c2 = m1 * m2 - m3 * m4
            s2 = -m2 * m3 - m1 * m4
            th2 = math.atan2(s2, c2)
            th2_deg = math.degrees(th2)
            th3_deg = math.degrees(th3)
            th4_deg = -(th2_deg + th3_deg) - 90.0

            if not (cls.TH1_LIMIT[0] <= th1_deg <= cls.TH1_LIMIT[1]):
                continue
            if not (cls.TH2_LIMIT[0] <= th2_deg <= cls.TH2_LIMIT[1]):
                continue
            if not (cls.TH3_LIMIT[0] <= th3_deg <= cls.TH3_LIMIT[1]):
                continue
            if not (cls.TH4_LIMIT[0] <= th4_deg <= cls.TH4_LIMIT[1]):
                continue
            return [float(th1_deg), float(th2_deg), float(th3_deg), float(th4_deg)]
        return None

    @classmethod
    def map_openvla_7d_to_raccoon_4dof(
        cls,
        current_ee_xyz_m: Optional[Sequence[float]],
        action7: Sequence[float],
        delta_scale: float,
        max_delta_xyz: float,
        gripper_close_threshold: float = 0.5,
        invert_gripper: bool = False,
    ) -> Dict[str, Any]:
        action = np.asarray(action7, dtype=np.float64).reshape(-1)
        if action.shape[0] < 7:
            raise ValueError(f"OpenVLA action must have 7 dims, got {action.shape[0]}")

        if current_ee_xyz_m is None:
            return {
                "raw_action_7d": safe_float_list(action, 7),
                "current_ee_xyz_m": None,
                "target_xyz_m": None,
                "joint_degrees_4dof": None,
                "gripper_cmd": float(action[6]),
                "gripper_state": None,
                "ik_success": False,
                "reason": "missing_ee_pose",
            }

        current = np.asarray(current_ee_xyz_m[:3], dtype=np.float64)
        raw_delta = action[:3] * float(delta_scale)
        clipped_delta = np.clip(raw_delta, -float(max_delta_xyz), float(max_delta_xyz))
        target_xyz_m = current + clipped_delta
        gripper_cmd = float(action[6])
        close = gripper_cmd >= float(gripper_close_threshold)
        if invert_gripper:
            close = not close
        gripper_state = "close" if close else "open"
        joints = cls.calculate_4dof_ik_from_xyz_m(target_xyz_m)

        return {
            "raw_action_7d": safe_float_list(action, 7),
            "current_ee_xyz_m": safe_float_list(current, 3),
            "raw_delta_xyz_m": safe_float_list(raw_delta, 3),
            "clipped_delta_xyz_m": safe_float_list(clipped_delta, 3),
            "target_xyz_m": safe_float_list(target_xyz_m, 3),
            "target_xyz_cm": safe_float_list(target_xyz_m * 100.0, 3),
            "joint_degrees_4dof": joints,
            "gripper_cmd": gripper_cmd,
            "gripper_state": gripper_state,
            "ik_success": joints is not None,
            "reason": "ok" if joints is not None else "ik_failed",
        }


class TraceLogger:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.rows: List[Dict[str, Any]] = []
        self.trace_jsonl = out_dir / "action_trace.jsonl"
        self.trace_csv = out_dir / "action_trace.csv"
        self.summary_json = out_dir / "summary.json"
        self.summary_md = out_dir / "summary.md"
        self.trace_jsonl.write_text("", encoding="utf-8")

    def log_step(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        with open(self.trace_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _flatten_for_csv(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in row.items():
            out[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
        return out

    def _write_csv(self) -> None:
        flat_rows = [self._flatten_for_csv(r) for r in self.rows]
        fieldnames: List[str] = []
        for row in flat_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(self.trace_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat_rows)

    def finalize(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        self._write_csv()
        summary: Dict[str, Any] = {"meta": meta, "num_steps": len(self.rows), "success": bool(meta.get("success", False))}

        for key in [
            "inference_time_sec", "sim_execution_time_sec", "total_step_time_sec",
            "ee_to_object_distance_m", "object_xy_displacement_m", "object_lift_height_m",
        ]:
            vals = [float(r[key]) for r in self.rows if r.get(key) is not None]
            if vals:
                summary[f"{key}_mean"] = float(np.mean(vals))
                summary[f"{key}_min"] = float(np.min(vals))
                summary[f"{key}_max"] = float(np.max(vals))
                summary[f"{key}_last"] = float(vals[-1])
        dists = [float(r["ee_to_object_distance_m"]) for r in self.rows if r.get("ee_to_object_distance_m") is not None]
        lifts = [float(r["object_lift_height_m"]) for r in self.rows if r.get("object_lift_height_m") is not None]
        ik = [bool(r["direct_4dof_ik_success"]) for r in self.rows if r.get("direct_4dof_ik_success") is not None]
        if dists:
            summary["min_ee_to_object_distance_m"] = float(min(dists))
        if lifts:
            summary["max_object_lift_height_m"] = float(max(lifts))
        if ik:
            summary["direct_4dof_ik_success_rate"] = float(np.mean([1.0 if x else 0.0 for x in ik]))
        summary["total_inference_time_sec"] = float(sum(float(r.get("inference_time_sec") or 0.0) for r in self.rows))
        summary["total_sim_execution_time_sec"] = float(sum(float(r.get("sim_execution_time_sec") or 0.0) for r in self.rows))
        summary["total_logged_step_time_sec"] = float(sum(float(r.get("total_step_time_sec") or 0.0) for r in self.rows))

        with open(self.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self._write_summary_md(summary)
        self._write_plots()
        return summary

    def _write_summary_md(self, summary: Dict[str, Any]) -> None:
        meta = summary.get("meta", {})
        lines = ["# Rollout Summary", "", "## Meta"]
        for key in ["task", "instruction", "target_color", "execution_preset", "speed", "settle_seconds_per_action", "max_delta_xyz", "success"]:
            if key in meta:
                lines.append(f"- **{key}**: `{meta[key]}`")
        lines += ["", "## Timing"]
        for key in ["inference_time_sec_mean", "sim_execution_time_sec_mean", "total_step_time_sec_mean", "total_inference_time_sec", "total_sim_execution_time_sec"]:
            if key in summary:
                lines.append(f"- **{key}**: `{summary[key]:.4f}` sec")
        lines += ["", "## Distance / Motion"]
        for key in ["min_ee_to_object_distance_m", "object_xy_displacement_m_last", "max_object_lift_height_m", "direct_4dof_ik_success_rate"]:
            if key in summary:
                val = summary[key]
                lines.append(f"- **{key}**: `{val:.4f}`" if isinstance(val, float) else f"- **{key}**: `{val}`")
        self.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_plots(self) -> None:
        if plt is None or not self.rows:
            return
        steps = [int(r["step"]) for r in self.rows]
        plt.figure()
        for key in ["inference_time_sec", "sim_execution_time_sec", "total_step_time_sec"]:
            vals = [r.get(key) for r in self.rows]
            if any(v is not None for v in vals):
                plt.plot(steps, [float(v) if v is not None else np.nan for v in vals], label=key)
        plt.xlabel("Step"); plt.ylabel("Time (sec)"); plt.title("Timing per Step"); plt.legend(); plt.tight_layout()
        plt.savefig(self.out_dir / "timing_trace.png", dpi=160); plt.close()

        plt.figure()
        for key in ["ee_to_object_distance_m", "object_xy_displacement_m", "object_lift_height_m"]:
            vals = [r.get(key) for r in self.rows]
            if any(v is not None for v in vals):
                plt.plot(steps, [float(v) if v is not None else np.nan for v in vals], label=key)
        plt.xlabel("Step"); plt.ylabel("Distance (m)"); plt.title("Distance / Object Motion Trace"); plt.legend(); plt.tight_layout()
        plt.savefig(self.out_dir / "distance_trace.png", dpi=160); plt.close()


def validate_obs(obs: Dict[str, Any]) -> None:
    if "image" not in obs:
        raise KeyError("env.get_observation() returned no 'image' key")


def get_target_object_metrics(env: SyncSimRaccoonEnv, target_color: str, object_specs: Dict[str, Dict[str, float]], obs: Dict[str, Any]) -> Dict[str, Any]:
    body_name = CYLINDER_BODY_BY_COLOR[target_color]
    obj_pose = env.get_object_pose(body_name)
    obj_xyz = safe_float_list(obj_pose, 3)
    initial_xy = np.array([object_specs[target_color]["x"], object_specs[target_color]["y"]], dtype=np.float64)
    current_xy = np.array(obj_xyz[:2], dtype=np.float64)
    ee_pose = obs.get("ee_pose")
    ee_dist = None
    if ee_pose is not None:
        ee_dist = float(np.linalg.norm(np.asarray(ee_pose[:3], dtype=np.float64) - np.asarray(obj_xyz[:3], dtype=np.float64)))
    return {
        "object_xyz_m": obj_xyz,
        "object_initial_xy_m": safe_float_list(initial_xy, 2),
        "object_xy_displacement_m": float(np.linalg.norm(current_xy - initial_xy)),
        "object_lift_height_m": float(obj_xyz[2] - 0.02) if len(obj_xyz) >= 3 else None,
        "ee_pose_m": safe_float_list(ee_pose, 3) if ee_pose is not None else None,
        "ee_to_object_distance_m": ee_dist,
    }


def update_success_meta(meta: Dict[str, Any], logger: TraceLogger, task: str, args: argparse.Namespace) -> None:
    dists = [r["ee_to_object_distance_m"] for r in logger.rows if r.get("ee_to_object_distance_m") is not None]
    disps = [r["object_xy_displacement_m"] for r in logger.rows if r.get("object_xy_displacement_m") is not None]
    lifts = [r["object_lift_height_m"] for r in logger.rows if r.get("object_lift_height_m") is not None]
    if dists:
        meta["min_ee_to_object_distance_m"] = float(min(dists))
    if disps:
        meta["final_object_xy_displacement_m"] = float(disps[-1])
    if lifts:
        meta["max_object_lift_height_m"] = float(max(lifts))
    if task == "lift":
        meta["success"] = bool(lifts and max(lifts) > args.success_lift_height_threshold)
    elif task in ("push", "pick_place"):
        meta["success"] = bool(disps and disps[-1] > args.success_displacement_threshold)
    else:
        meta["success"] = bool(dists and min(dists) < args.success_distance_threshold)


def print_step_log(row: Dict[str, Any]) -> None:
    def fmt(v: Any, d: int = 4) -> str:
        if v is None:
            return "-"
        try:
            return f"{float(v):.{d}f}"
        except Exception:
            return str(v)
    print(
        f"[{row['step']:03d}] OK | task={row['task']} | color={row['target_color']} | "
        f"infer={fmt(row.get('inference_time_sec'),3)}s | sim={fmt(row.get('sim_execution_time_sec'),3)}s | "
        f"ee_obj={fmt(row.get('ee_to_object_distance_m'))}m | disp={fmt(row.get('object_xy_displacement_m'))}m | "
        f"lift={fmt(row.get('object_lift_height_m'))}m | grip={row.get('gripper_action')} | "
        f"fallback={row.get('fallback_used')} | retry={row.get('retry_count')} | ik={row.get('direct_4dof_ik_success')}"
    )


def rollout(args: argparse.Namespace, server_url: str) -> None:
    rng = np.random.default_rng(args.seed)
    target_color, task, instruction = resolve_target_color_task_instruction(
        args.instruction, args.target_color, args.task, args.instruction_variant, rng
    )
    out_dir = Path(args.output_dir) / f"{task}_{target_color}_{args.execution_preset}_episode_{args.episode_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_existing_images(out_dir)
    logger = TraceLogger(out_dir)

    object_specs = sample_object_specs(rng, tuple(args.object_x_range), tuple(args.object_y_range), min_distance=args.min_object_distance) if args.randomize_objects else make_default_object_specs()
    env = SyncSimRaccoonEnv(xml_path=args.xml_path, image_size=(256, 256), camera_name=args.camera_name, use_viewer=args.use_viewer)

    meta = {
        "task": task, "instruction": instruction, "target_color": target_color,
        "target_body_name": CYLINDER_BODY_BY_COLOR[target_color], "execution_preset": args.execution_preset,
        "use_real_robot": False, "speed": args.speed, "settle_seconds_per_action": args.settle_seconds_per_action,
        "max_delta_xyz": args.max_delta_xyz, "delta_scale": args.delta_scale, "seed": args.seed,
        "randomize_objects": args.randomize_objects, "success_distance_threshold": args.success_distance_threshold,
        "success_displacement_threshold": args.success_displacement_threshold,
        "success_lift_height_threshold": args.success_lift_height_threshold,
        "all_object_init_poses": object_specs_to_meta(object_specs), "success": False, "final_status": "started",
    }
    (out_dir / "rollout_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        reset_multicolor_scene(env, object_specs, target_color)
        if hasattr(env, "lockh"):
            env.lockh()
        if hasattr(env, "debug_check_current_ee_reachable"):
            env.debug_check_current_ee_reachable()
        if args.initial_settle_seconds > 0 and hasattr(env, "settle_steps"):
            env.settle_steps(seconds=args.initial_settle_seconds)

        print(f"[SCENE] task={task} | instruction={instruction!r} | color={target_color} | preset={args.execution_preset}")
        obs = env.get_observation(); validate_obs(obs)

        for step_idx in range(args.max_steps):
            step_start = time.perf_counter()
            inference_time = sim_execution_time = None
            action7 = None; direct_4dof = None; frame_name = None
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
                t1 = time.perf_counter()
                exec_info = env.execute_delta_action7(action=action7, speed=args.speed, delta_scale=args.delta_scale, max_delta_xyz=args.max_delta_xyz)
                sim_execution_time = time.perf_counter() - t1
                if args.settle_seconds_per_action > 0 and hasattr(env, "settle_steps"):
                    env.settle_steps(seconds=args.settle_seconds_per_action)
                obs = env.get_observation(); validate_obs(obs)
                frame_name = f"frame_{step_idx:06d}.png"
                Image.fromarray(obs["image"]).save(out_dir / frame_name)
                metrics = get_target_object_metrics(env, target_color, object_specs, obs)
                row = {
                    "step": step_idx, "task": task, "instruction": instruction, "target_color": target_color,
                    "execution_preset": args.execution_preset, "raw_action_7d": action7,
                    "direct_4dof_target_xyz_m": direct_4dof.get("target_xyz_m"),
                    "direct_4dof_joint_degrees": direct_4dof.get("joint_degrees_4dof"),
                    "direct_4dof_gripper_state": direct_4dof.get("gripper_state"),
                    "direct_4dof_ik_success": direct_4dof.get("ik_success"),
                    "direct_4dof_reason": direct_4dof.get("reason"),
                    "env_applied_delta_xyz": exec_info.get("applied_delta_xyz"),
                    "env_final_delta_xyz": exec_info.get("final_delta_xyz"),
                    "env_actual_move_xyz": exec_info.get("actual_move_xyz"),
                    "env_target_xyz": exec_info.get("target_xyz"),
                    "env_raw_rotation_rpy": exec_info.get("raw_rotation_rpy"),
                    "gripper_cmd": exec_info.get("gripper_cmd"), "gripper_action": exec_info.get("gripper_action"),
                    "fallback_used": exec_info.get("fallback_used"), "retry_count": exec_info.get("retry_count"),
                    "blocked_by_height": exec_info.get("gripper_blocked_by_height"),
                    "inference_time_sec": inference_time, "sim_execution_time_sec": sim_execution_time,
                    "total_step_time_sec": time.perf_counter() - step_start, **metrics,
                    "frame": frame_name, "status": "ok",
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
                    "step": step_idx, "task": task, "instruction": instruction, "target_color": target_color,
                    "raw_action_7d": action7, "direct_4dof": direct_4dof,
                    "inference_time_sec": inference_time, "sim_execution_time_sec": sim_execution_time,
                    "total_step_time_sec": time.perf_counter() - step_start,
                    "error": repr(exc), "frame": frame_name, "status": "fail",
                }
                logger.log_step(row)
                print(f"[{step_idx:03d}] FAIL | {exc}")
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
        env.close()
        summary = logger.finalize(meta)
        print(f"[SUMMARY] saved to {out_dir}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def resolve_ssh_password(args: argparse.Namespace) -> Optional[str]:
    if args.ssh_password:
        return args.ssh_password
    if os.environ.get("OPENVLA_SSH_PASSWORD"):
        return os.environ["OPENVLA_SSH_PASSWORD"]
    if args.use_ssh_tunnel and args.ssh_ask_password:
        return getpass("SSH password: ")
    return None


def maybe_tunnel_context(args: argparse.Namespace):
    if not args.use_ssh_tunnel:
        return nullcontext(None)
    if not args.ssh_host or not args.ssh_user:
        raise ValueError("--use_ssh_tunnel requires --ssh_host and --ssh_user or env vars.")
    return SSHTunnelForwarder(
        ssh_address_or_host=(args.ssh_host, args.ssh_port), ssh_username=args.ssh_user,
        ssh_password=resolve_ssh_password(args), remote_bind_address=(args.remote_server_host, args.remote_server_port),
        local_bind_address=(args.local_server_host, args.local_server_port),
    )


def build_server_url(args: argparse.Namespace, tunnel: Optional[SSHTunnelForwarder]) -> str:
    if tunnel is not None:
        return f"http://{args.local_server_host}:{tunnel.local_bind_port}"
    if not args.server_url:
        raise ValueError("--server_url is required when --use_ssh_tunnel is not enabled.")
    return args.server_url


def apply_execution_preset(args: argparse.Namespace) -> None:
    if args.execution_preset == "custom":
        return
    p = EXECUTION_PRESETS[args.execution_preset]
    args.speed = p["speed"]; args.settle_seconds_per_action = p["settle_seconds_per_action"]
    args.max_delta_xyz = p["max_delta_xyz"]; args.delta_scale = p["delta_scale"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--xml_path", type=str, default="Raccoon_colored_cylinder.xml")
    p.add_argument("--server_url", type=str, default=None)
    p.add_argument("--unnorm_key", type=str, default="raccoon_pick_place")
    p.add_argument("--task", type=str, default="auto", choices=["auto", *TASKS])
    p.add_argument("--instruction", type=str, default=None)
    p.add_argument("--instruction_variant", type=int, default=-1)
    p.add_argument("--target_color", type=str, default="auto", choices=["auto", "random", *CYLINDER_COLORS])
    p.add_argument("--output_dir", type=str, default="rollout_outputs_enhanced")
    p.add_argument("--episode_id", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--camera_name", type=str, default="front_view")
    p.add_argument("--use_viewer", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--execution_preset", type=str, default="custom", choices=["custom", "safe", "balanced", "fast"])
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
    args = parse_args(); apply_execution_preset(args)
    with maybe_tunnel_context(args) as tunnel:
        server_url = build_server_url(args, tunnel)
        if tunnel is not None:
            print(f"[SSH] {args.local_server_host}:{tunnel.local_bind_port} -> {args.remote_server_host}:{args.remote_server_port}")
        rollout(args, server_url)


if __name__ == "__main__":
    main()
