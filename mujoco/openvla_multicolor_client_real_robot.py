import argparse
import base64
import io
import json
import math
import os
import re
import time
from contextlib import nullcontext
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np
import requests
from PIL import Image
from sshtunnel import SSHTunnelForwarder

from raccoon_env import SyncSimRaccoonEnv

try:
    from roboid import Raccoon
except ImportError:
    Raccoon = None


CYLINDER_BODY_BY_COLOR = {
    "red": "target_object",
    "blue": "target_object_blue",
    "green": "target_object_green",
    "yellow": "target_object_yellow",
}
CYLINDER_COLORS = tuple(CYLINDER_BODY_BY_COLOR.keys())

# Dataset collection code와 동일한 기본 배치 조건.
# 이전 단일 object range였던 x=(-0.18, 0.18), y=(0.10, 0.18)보다
# x는 좁게, y는 조금 더 앞으로 제한한다.
DEFAULT_OBJECT_X_RANGE = (-0.10, 0.10)
DEFAULT_OBJECT_Y_RANGE = (0.16, 0.25)
DEFAULT_MIN_OBJECT_DISTANCE = 0.035
DEFAULT_YAW_RANGE = (-math.pi / 4, math.pi / 4)
DEFAULT_INSTRUCTION_TEMPLATE = "grasp the {color} cylinder"


def image_to_b64(image_rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def request_action(
    server_url: str,
    instruction: str,
    image_rgb: np.ndarray,
    unnorm_key: Optional[str],
    timeout: float = 60.0,
) -> Dict[str, Any]:
    payload = {
        "instruction": instruction,
        "image_b64": image_to_b64(image_rgb),
        "unnorm_key": unnorm_key,
        "do_sample": False,
    }
    response = requests.post(f"{server_url.rstrip('/')}/predict", json=payload, timeout=timeout)
    if not response.ok:
        print(f"[SERVER ERROR] {response.status_code} | {response.text}")
        response.raise_for_status()
    return response.json()


def resolve_ssh_password(args: argparse.Namespace) -> Optional[str]:
    if args.ssh_password:
        return args.ssh_password
    env_password = os.environ.get("OPENVLA_SSH_PASSWORD")
    if env_password:
        return env_password
    if args.use_ssh_tunnel and args.ssh_ask_password:
        return getpass("SSH password: ")
    return None


def open_ssh_tunnel(args: argparse.Namespace) -> SSHTunnelForwarder:
    ssh_password = resolve_ssh_password(args)
    tunnel = SSHTunnelForwarder(
        ssh_address_or_host=(args.ssh_host, args.ssh_port),
        ssh_username=args.ssh_user,
        ssh_password=ssh_password,
        remote_bind_address=(args.remote_server_host, args.remote_server_port),
        local_bind_address=(args.local_server_host, args.local_server_port),
    )
    tunnel.start()
    return tunnel


def build_server_url(args: argparse.Namespace, tunnel: Optional[SSHTunnelForwarder]) -> str:
    if tunnel is not None:
        return f"http://{args.local_server_host}:{tunnel.local_bind_port}"
    if not args.server_url:
        raise ValueError("--server_url is required when --use_ssh_tunnel is not enabled.")
    return args.server_url


def maybe_tunnel_context(args: argparse.Namespace):
    if args.use_ssh_tunnel:
        return open_ssh_tunnel(args)
    return nullcontext(None)



class RealRaccoonController:
    """
    실제 라쿤봇 하드웨어 제어 어댑터.

    서버 이미지는 기존 코드 그대로 MuJoCo obs["image"]를 사용하고,
    서버에서 받은 action은 먼저 SyncSimRaccoonEnv.execute_delta_action7()로
    clipping/IK/retry가 적용된다. 그 결과 exec_info["target_xyz"]를
    같은 IK 기준으로 실제 라쿤봇 관절 각도로 변환해 전송한다.
    """

    L1, L2, L3, L4 = 8.25, 10.0, 10.0, 8.0
    HOME_DEGREES = (0.0, -10.0, -140.0, 60.0)

    def __init__(
        self,
        require_ready: bool = True,
        home_wait_seconds: float = 5.0,
        beep_on_ready: bool = True,
    ) -> None:
        if Raccoon is None:
            raise ImportError(
                "roboid 패키지를 import할 수 없습니다. 실제 라쿤봇 제어 환경에서 실행하거나 "
                "--use_real_robot 옵션을 끄세요."
            )

        self.hw = Raccoon()
        ready = bool(getattr(getattr(self.hw, "_roboid", None), "_ready", False))
        if not ready:
            msg = "라쿤봇 하드웨어 연결에 실패했습니다. USB/Bluetooth 연결과 전원을 확인하세요."
            if require_ready:
                raise RuntimeError(msg)
            print(f"[REAL_ROBOT WARN] {msg} 시뮬레이션 명령만 계속합니다.")
            self.hw = None
            return

        self.go_home(wait_seconds=home_wait_seconds)
        self.lockh()
        self.open_gripper()

        if beep_on_ready:
            try:
                self.hw.beep()
            except Exception as exc:
                print(f"[REAL_ROBOT WARN] beep 실패: {exc}")

        print("[REAL_ROBOT] 하드웨어 연결 성공")

    @property
    def connected(self) -> bool:
        return self.hw is not None

    def _try_call(self, fn_name: str, *candidate_args: Sequence[Any]) -> bool:
        """roboid 버전별 API 차이를 흡수하기 위한 작은 wrapper."""
        if not self.connected:
            return False
        fn = getattr(self.hw, fn_name, None)
        if fn is None:
            return False

        last_error: Optional[Exception] = None
        for args in candidate_args:
            try:
                fn(*args)
                return True
            except TypeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return False

    def _send_joint_degrees(self, degrees: Sequence[float], speed: int = 70) -> None:
        degrees = [float(v) for v in degrees[:4]]
        speed = int(speed)

        # 일부 roboid 버전: set_degree(j, deg, speed) 또는 set_degree(j1, j2, j3, j4)
        if self._try_call("set_degree", (*degrees, speed), tuple(degrees)):
            return

        # 현재 설치된 패키지에서 주로 쓰이는 이름: degree_to(...)
        if self._try_call("degree_to", (*degrees, speed), tuple(degrees), ([1, 2, 3, 4], degrees, speed)):
            return

        # per-joint API만 있는 경우 fallback
        per_joint_ok = True
        for joint_id, degree in enumerate(degrees, start=1):
            if not (
                self._try_call("set_degree", (joint_id, degree, speed), (joint_id, degree))
                or self._try_call("degree_to", (joint_id, degree, speed), (joint_id, degree))
            ):
                per_joint_ok = False
                break
        if per_joint_ok:
            return

        raise AttributeError("Raccoon 객체에서 set_degree/degree_to 관절 제어 API를 찾지 못했습니다.")

    def _calc_inv_kinematics(self, x_cm: float, y_cm: float, z_cm: float) -> Optional[list[float]]:
        if not (
            isinstance(x_cm, (int, float))
            and isinstance(y_cm, (int, float))
            and isinstance(z_cm, (int, float))
        ):
            return None

        if not ((-28.0 <= x_cm <= 28.0) and (-15.0 <= y_cm <= 28.0) and (0.0 <= z_cm <= 36.25)):
            return None

        x, y, z = y_cm, -x_cm, z_cm
        th1 = math.atan2(y, x)
        c1 = math.cos(th1)
        s1 = math.sin(th1)

        wx = x - self.L4 * c1
        wy = y - self.L4 * s1
        wz = z - self.L1

        c3 = (wx * wx + wy * wy + wz * wz - self.L2 * self.L2 - self.L3 * self.L3) / (2.0 * self.L2 * self.L3)
        if c3 < -1.0001 or c3 > 1.0001:
            return None
        c3 = float(np.clip(c3, -1.0, 1.0))

        s3_abs = math.sqrt(max(0.0, 1.0 - c3 * c3))
        th1_deg = math.degrees(th1)

        for s3 in (-s3_abs, s3_abs):
            th3 = math.atan2(s3, c3)

            m1 = c3 * self.L3 + self.L2
            m2 = wz
            m3 = s3 * self.L3
            m4 = c1 * wx + s1 * wy

            c2 = m1 * m2 - m3 * m4
            s2 = -m2 * m3 - m1 * m4
            th2 = math.atan2(s2, c2)

            th2_deg = math.degrees(th2)
            th3_deg = math.degrees(th3)
            th4_deg = -(th2_deg + th3_deg) - 90.0

            if th1_deg < -120.0 or th1_deg > 120.0:
                continue
            if th2_deg < -90.0 or th2_deg > 30.0:
                continue
            if th3_deg < -150.0 or th3_deg > 0.0:
                continue

            return [th1_deg, th2_deg, th3_deg, th4_deg]

        return None

    def move_to(self, x_cm: float, y_cm: float, z_cm: float, speed: int = 70) -> list[float]:
        angles = self._calc_inv_kinematics(float(x_cm), float(y_cm), float(z_cm))
        if angles is None:
            raise ValueError(f"[REAL_ROBOT] IK fail: ({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f}) cm")
        self._send_joint_degrees(angles[:4], speed=speed)
        return angles

    def open_gripper(self) -> None:
        if self.connected:
            if not self._try_call("open_gripper", tuple()):
                print("[REAL_ROBOT WARN] open_gripper API를 찾지 못했습니다.")

    def close_gripper(self) -> None:
        if self.connected:
            if not self._try_call("close_gripper", tuple()):
                print("[REAL_ROBOT WARN] close_gripper API를 찾지 못했습니다.")

    def lockh(self) -> None:
        if self.connected:
            if not (self._try_call("lock_horz", tuple()) or self._try_call("lockh", tuple())):
                print("[REAL_ROBOT WARN] gripper horizontal lock API를 찾지 못했습니다.")

    def lockv(self) -> None:
        if self.connected:
            if not (self._try_call("lock_vert", tuple()) or self._try_call("lockv", tuple())):
                print("[REAL_ROBOT WARN] gripper vertical lock API를 찾지 못했습니다.")

    def unlock(self) -> None:
        if self.connected:
            self._try_call("unlock", tuple())

    def go_home(self, wait_seconds: float = 0.0) -> None:
        if self.connected:
            self._send_joint_degrees(self.HOME_DEGREES, speed=50)
            if wait_seconds > 0:
                time.sleep(wait_seconds)

    def execute_from_exec_info(self, exec_info: Dict[str, Any], speed: int = 70) -> Dict[str, Any]:
        tx, ty, tz = [float(v) for v in exec_info["target_xyz"]]
        gripper = float(exec_info["gripper_cmd"])

        angles = self.move_to(tx * 100.0, ty * 100.0, tz * 100.0, speed=speed)

        if gripper >= 0.5:
            self.close_gripper()
            gripper_state = "close"
        else:
            self.open_gripper()
            gripper_state = "open"

        real_info = {
            "target_xyz_m": [tx, ty, tz],
            "target_xyz_cm": [tx * 100.0, ty * 100.0, tz * 100.0],
            "joint_degrees": [float(v) for v in angles[:4]],
            "gripper_state": gripper_state,
        }
        print(
            f"[REAL_ROBOT] target_cm={[round(v, 2) for v in real_info['target_xyz_cm']]} | "
            f"joint_deg={[round(v, 2) for v in real_info['joint_degrees']]} | "
            f"gripper={gripper_state}"
        )
        return real_info

    def close(self) -> None:
        # roboid.Raccoon에는 명시적 close API가 없는 버전이 많아서 no-op 처리.
        pass

def print_success_log(step_idx: int, exec_info: Dict[str, Any]) -> None:
    final_delta_xyz = [round(float(v), 4) for v in exec_info["final_delta_xyz"]]
    move_xyz = [round(float(v), 4) for v in exec_info["actual_move_xyz"]]
    target_xyz = [round(float(v), 4) for v in exec_info["target_xyz"]]
    gripper = float(exec_info["gripper_cmd"])
    retries = int(exec_info["retry_count"])

    raw_rot = exec_info.get("raw_rotation_rpy", [0.0, 0.0, 0.0])
    raw_rot = [round(float(v), 4) for v in raw_rot]

    applied_delta = exec_info.get("applied_delta_xyz", final_delta_xyz)
    applied_delta = [round(float(v), 4) for v in applied_delta]

    projection_info = exec_info.get("projection_info", {})
    rot_proj = projection_info.get("rotation_projection_xyz", [0.0, 0.0, 0.0])
    rot_proj = [round(float(v), 5) for v in rot_proj]

    fallback_used = exec_info.get("fallback_used", "none")
    gripper_action = exec_info.get("gripper_action", "close" if gripper >= 0.5 else "open")
    blocked = bool(exec_info.get("gripper_blocked_by_height", False))

    print(
        f"[{step_idx:03d}] OK | "
        f"applied={applied_delta} | "
        f"rot={raw_rot} | rot_proj={rot_proj} | "
        f"final_delta={final_delta_xyz} | move={move_xyz} | target={target_xyz} | "
        f"gripper_cmd={gripper:.1f} -> {gripper_action} | "
        f"blocked_by_height={blocked} | fallback={fallback_used} | retries={retries}"
    )



def print_fail_log(step_idx: int, exc: Exception) -> None:
    print(f"[{step_idx:03d}] FAIL | {exc}")


def infer_color_from_instruction(instruction: Optional[str]) -> Optional[str]:
    """Return the single color word found in an instruction, or None."""
    if not instruction:
        return None

    text = instruction.lower()
    matches = []
    for color in CYLINDER_COLORS:
        if re.search(rf"\b{re.escape(color)}\b", text):
            matches.append(color)

    if len(matches) > 1:
        raise ValueError(f"instruction에 여러 색상이 들어 있습니다: {matches} | instruction={instruction!r}")
    return matches[0] if matches else None


def resolve_target_color_and_instruction(
    instruction: Optional[str],
    target_color_arg: str,
    rng: np.random.Generator,
    instruction_template: str,
) -> Tuple[str, str]:
    """
    Keep the OpenVLA prompt and the physical target color synchronized.

    Priority:
      1. If instruction already contains exactly one color, use that color.
      2. Else if --target_color is one of red/blue/green/yellow, use it.
      3. Else choose a random color and generate instruction from template.
    """
    instruction_color = infer_color_from_instruction(instruction)

    if instruction_color is not None:
        target_color = instruction_color
        if target_color_arg in CYLINDER_COLORS and target_color_arg != instruction_color:
            raise ValueError(
                f"--instruction 색상({instruction_color})과 --target_color({target_color_arg})가 다릅니다. "
                "OpenVLA prompt와 실제 target이 어긋나지 않도록 둘 중 하나를 수정하세요."
            )
    elif target_color_arg in CYLINDER_COLORS:
        target_color = target_color_arg
    elif target_color_arg in ("auto", "random"):
        target_color = str(rng.choice(CYLINDER_COLORS))
    else:
        raise ValueError(f"지원하지 않는 --target_color 값입니다: {target_color_arg}")

    if instruction is None or instruction.strip() == "":
        instruction = instruction_template.format(color=target_color)

    return target_color, instruction


def make_default_object_specs() -> Dict[str, Dict[str, float]]:
    """Deterministic fallback used when randomization is disabled."""
    x_values = np.linspace(
        DEFAULT_OBJECT_X_RANGE[0] * 0.75,
        DEFAULT_OBJECT_X_RANGE[1] * 0.75,
        len(CYLINDER_COLORS),
    )
    y_center = float(sum(DEFAULT_OBJECT_Y_RANGE) / 2.0)
    return {
        color: {
            "body_name": CYLINDER_BODY_BY_COLOR[color],
            "x": float(x_values[idx]),
            "y": y_center,
            "yaw": 0.0,
        }
        for idx, color in enumerate(CYLINDER_COLORS)
    }


def sample_object_specs(
    rng: np.random.Generator,
    x_range: Tuple[float, float] = DEFAULT_OBJECT_X_RANGE,
    y_range: Tuple[float, float] = DEFAULT_OBJECT_Y_RANGE,
    yaw_range: Tuple[float, float] = DEFAULT_YAW_RANGE,
    min_distance: float = DEFAULT_MIN_OBJECT_DISTANCE,
    max_tries: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """
    Dataset collection code와 동일한 조건으로 4개 색상 cylinder를 모두 배치한다.

    Defaults:
      - x_range=(-0.10, 0.10)
      - y_range=(0.16, 0.20)
      - min_object_distance=0.035
      - yaw_range=(-pi/4, pi/4)
    """
    if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
        raise ValueError(f"잘못된 spawn range입니다: x_range={x_range}, y_range={y_range}")

    specs: Dict[str, Dict[str, float]] = {}
    placed_xy = []

    # 특정 색상이 항상 먼저 배치되어 유리/불리해지는 bias를 줄인다.
    placement_order = list(CYLINDER_COLORS)
    rng.shuffle(placement_order)

    for color in placement_order:
        for _ in range(max_tries):
            x = float(rng.uniform(x_range[0], x_range[1]))
            y = float(rng.uniform(y_range[0], y_range[1]))
            xy = np.array([x, y], dtype=np.float64)

            if all(np.linalg.norm(xy - other_xy) >= min_distance for other_xy in placed_xy):
                specs[color] = {
                    "body_name": CYLINDER_BODY_BY_COLOR[color],
                    "x": x,
                    "y": y,
                    "yaw": float(rng.uniform(yaw_range[0], yaw_range[1])),
                }
                placed_xy.append(xy)
                break
        else:
            raise RuntimeError(
                "색상 cylinder 4개를 겹치지 않게 배치하지 못했습니다. "
                f"x_range={x_range}, y_range={y_range}, min_distance={min_distance}를 확인하세요."
            )

    return {color: specs[color] for color in CYLINDER_COLORS}


def reset_freejoint_body_pose(env: SyncSimRaccoonEnv, body_name: str, x: float, y: float, z: float, yaw: float) -> None:
    """Set a MuJoCo freejoint body pose directly through env.model/env.data."""
    if not hasattr(env, "model") or not hasattr(env, "data"):
        raise AttributeError("SyncSimRaccoonEnv에 model/data 속성이 필요합니다.")

    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"body not found: {body_name}. XML이 Raccoon_colored_cylinder.xml인지 확인하세요.")

    jnt_adr = int(env.model.body_jntadr[body_id])
    jnt_num = int(env.model.body_jntnum[body_id])
    if jnt_num < 1:
        raise ValueError(f"{body_name} has no joint")

    joint_id = jnt_adr
    qpos_adr = int(env.model.jnt_qposadr[joint_id])

    # freejoint qpos = [x, y, z, qw, qx, qy, qz]
    qw = math.cos(yaw / 2.0)
    qz = math.sin(yaw / 2.0)
    env.data.qpos[qpos_adr:qpos_adr + 7] = np.array([x, y, z, qw, 0.0, 0.0, qz], dtype=np.float64)

    qvel_adr = int(env.model.jnt_dofadr[joint_id])
    env.data.qvel[qvel_adr:qvel_adr + 6] = 0.0


def reset_multicolor_scene(
    env: SyncSimRaccoonEnv,
    object_specs: Dict[str, Dict[str, float]],
    target_color: str,
) -> None:
    """
    Reset the robot using the existing env.reset_episode(), then place all four
    colored cylinders in the scene. The prompted color is stored as env.active_object_body_name
    when the env supports that attribute, but inference only needs the rendered image.
    """
    if target_color not in object_specs:
        raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")

    target_spec = object_specs[target_color]

    # Existing raccoon_env expects a single target pose for reset_episode().
    # We use the prompted target pose to reset the robot/home state, then override
    # all four cylinder poses below.
    env.reset_episode(float(target_spec["x"]), float(target_spec["y"]), float(target_spec["yaw"]))

    for color, spec in object_specs.items():
        reset_freejoint_body_pose(
            env=env,
            body_name=str(spec["body_name"]),
            x=float(spec["x"]),
            y=float(spec["y"]),
            z=0.02,
            yaw=float(spec["yaw"]),
        )

    target_body_name = str(target_spec["body_name"])
    if hasattr(env, "active_object_body_name"):
        env.active_object_body_name = target_body_name
    if hasattr(env, "target_body_name"):
        env.target_body_name = target_body_name

    mujoco.mj_forward(env.model, env.data)


def object_specs_to_meta(object_specs: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    return {
        color: {
            "body_name": str(spec["body_name"]),
            "xy": [float(spec["x"]), float(spec["y"])],
            "yaw": float(spec["yaw"]),
        }
        for color, spec in object_specs.items()
    }


def write_rollout_meta(
    out_dir: Path,
    instruction: str,
    target_color: str,
    object_specs: Dict[str, Dict[str, float]],
    args: Dict[str, Any],
) -> None:
    meta = {
        "instruction": instruction,
        "target_color": target_color,
        "target_body_name": CYLINDER_BODY_BY_COLOR[target_color],
        "all_object_init_poses": object_specs_to_meta(object_specs),
        "args": args,
    }
    with open(out_dir / "rollout_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def rollout(
    xml_path: str,
    server_url: str,
    instruction: Optional[str],
    unnorm_key: str,
    output_dir: str,
    episode_id: int = 1,
    max_steps: int = 1000000,
    use_viewer: bool = True,
    camera_name: str = "front_view",
    speed: int = 70,
    settle_seconds_per_action: float = 0.8,
    initial_settle_seconds: float = 0.3,
    delta_scale: float = 1.0,
    randomize_objects: bool = True,
    request_timeout: float = 60.0,
    max_delta_xyz: float = 0.005,
    target_color_arg: str = "auto",
    instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE,
    seed: Optional[int] = None,
    object_x_range: Tuple[float, float] = DEFAULT_OBJECT_X_RANGE,
    object_y_range: Tuple[float, float] = DEFAULT_OBJECT_Y_RANGE,
    min_object_distance: float = DEFAULT_MIN_OBJECT_DISTANCE,
    use_real_robot: bool = False,
    allow_sim_only_on_hw_fail: bool = False,
    real_initial_wait_seconds: float = 5.0,
    real_settle_seconds: Optional[float] = None,
    real_go_home_on_exit: bool = False,
) -> None:
    out_dir = Path(output_dir) / f"episode_{episode_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 기존 이미지 삭제 후 새로 저장 시작
    clear_existing_images(out_dir)

    rng = np.random.default_rng(seed)
    target_color, instruction = resolve_target_color_and_instruction(
        instruction=instruction,
        target_color_arg=target_color_arg,
        rng=rng,
        instruction_template=instruction_template,
    )

    if randomize_objects:
        object_specs = sample_object_specs(
            rng=rng,
            x_range=object_x_range,
            y_range=object_y_range,
            min_distance=min_object_distance,
        )
    else:
        object_specs = make_default_object_specs()

    env = SyncSimRaccoonEnv(
        xml_path=xml_path,
        image_size=(256, 256),
        camera_name=camera_name,
        use_viewer=use_viewer,
    )
    real_robot: Optional[RealRaccoonController] = None

    try:
        reset_multicolor_scene(
            env=env,
            object_specs=object_specs,
            target_color=target_color,
        )

        env.lockh()
        if use_real_robot:
            real_robot = RealRaccoonController(
                require_ready=not allow_sim_only_on_hw_fail,
                home_wait_seconds=real_initial_wait_seconds,
            )

        env.debug_check_current_ee_reachable()

        # Dataset collector와 동일하게 첫 observation 전에 free-joint cylinder를 안정화한다.
        if initial_settle_seconds > 0:
            env.settle_steps(seconds=initial_settle_seconds)

        write_rollout_meta(
            out_dir=out_dir,
            instruction=instruction,
            target_color=target_color,
            object_specs=object_specs,
            args={
                "xml_path": xml_path,
                "unnorm_key": unnorm_key,
                "camera_name": camera_name,
                "speed": speed,
                "settle_seconds_per_action": settle_seconds_per_action,
                "initial_settle_seconds": initial_settle_seconds,
                "delta_scale": delta_scale,
                "max_delta_xyz": max_delta_xyz,
                "seed": seed,
                "object_x_range": list(object_x_range),
                "object_y_range": list(object_y_range),
                "min_object_distance": min_object_distance,
                "use_real_robot": use_real_robot,
                "allow_sim_only_on_hw_fail": allow_sim_only_on_hw_fail,
                "real_initial_wait_seconds": real_initial_wait_seconds,
                "real_settle_seconds": real_settle_seconds,
                "real_go_home_on_exit": real_go_home_on_exit,
            },
        )

        print(
            f"[SCENE] instruction={instruction!r} | target_color={target_color!r} | "
            f"target_xy=({object_specs[target_color]['x']:.3f}, {object_specs[target_color]['y']:.3f}) | "
            f"objects={object_specs_to_meta(object_specs)}"
        )

        obs = env.get_observation()
        step_idx = 0

        while True:
            response = request_action(
                server_url=server_url,
                instruction=instruction,
                image_rgb=obs["image"],
                unnorm_key=unnorm_key,
                timeout=request_timeout,
            )
            action = response["action"]

            try:
                exec_info = env.execute_delta_action7(
                    action=action,
                    speed=speed,
                    delta_scale=delta_scale,
                    max_delta_xyz=max_delta_xyz,
                )

                if real_robot is not None and real_robot.connected:
                    exec_info["real_robot"] = real_robot.execute_from_exec_info(exec_info, speed=speed)
                    wait_seconds = settle_seconds_per_action if real_settle_seconds is None else real_settle_seconds
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)

                print_success_log(step_idx, exec_info)

                env.settle_steps(seconds=settle_seconds_per_action)
                obs = env.get_observation()

                frame_name = f"frame_{step_idx:06d}.png"
                Image.fromarray(obs["image"]).save(out_dir / frame_name)

            except Exception as exc:
                print_fail_log(step_idx, exc)
                obs = env.get_observation()

                frame_name = f"frame_{step_idx:06d}_skipped.png"
                Image.fromarray(obs["image"]).save(out_dir / frame_name)

                step_idx += 1
                if step_idx >= max_steps:
                    print("[STOP] max_steps reached")
                    break
                continue

            step_idx += 1
            if step_idx >= max_steps:
                print("[STOP] max_steps reached")
                break

    except KeyboardInterrupt:
        print("\n[STOP] interrupted by user")

    finally:
        if real_robot is not None:
            if real_go_home_on_exit and real_robot.connected:
                real_robot.go_home(wait_seconds=0.0)
            real_robot.close()
        env.close()


def clear_existing_images(out_dir: Path) -> None:
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    deleted_count = 0
    for file_path in out_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_exts:
            file_path.unlink()
            deleted_count += 1

    print(f"[CLEANUP] removed {deleted_count} existing image files from {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, default="Raccoon_colored_cylinder.xml")
    parser.add_argument("--server_url", type=str, default=None, help="Direct HTTP URL, e.g. http://127.0.0.1:8000")
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="OpenVLA prompt. If omitted, generated as 'grasp the {color} cylinder'.",
    )
    parser.add_argument(
        "--target_color",
        type=str,
        default="auto",
        choices=["auto", "random", *CYLINDER_COLORS],
        help="Target color. 'auto' uses the color in --instruction, or random if instruction has no color.",
    )
    parser.add_argument("--instruction_template", type=str, default=DEFAULT_INSTRUCTION_TEMPLATE)
    parser.add_argument("--unnorm_key", type=str, default="raccoon_pick_place")
    parser.add_argument("--output_dir", type=str, default="rollout_outputs")
    parser.add_argument("--episode_id", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--speed", type=int, default=70)
    parser.add_argument("--settle_seconds_per_action", type=float, default=0.8)
    parser.add_argument("--initial_settle_seconds", type=float, default=0.3)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--max_delta_xyz", type=float, default=0.005)
    parser.add_argument("--request_timeout", type=float, default=60.0)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--camera_name", type=str, default="front_view")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--object_x_range", type=float, nargs=2, default=DEFAULT_OBJECT_X_RANGE)
    parser.add_argument("--object_y_range", type=float, nargs=2, default=DEFAULT_OBJECT_Y_RANGE)
    parser.add_argument("--min_object_distance", type=float, default=DEFAULT_MIN_OBJECT_DISTANCE)
    parser.add_argument("--use_real_robot", action="store_true", help="서버 action을 실제 라쿤봇 하드웨어에도 전송합니다.")
    parser.add_argument(
        "--allow_sim_only_on_hw_fail",
        action="store_true",
        help="--use_real_robot 상태에서 하드웨어 연결 실패 시 종료하지 않고 MuJoCo만 계속합니다.",
    )
    parser.add_argument("--real_initial_wait_seconds", type=float, default=5.0, help="실제 로봇 home 이동 후 대기 시간")
    parser.add_argument(
        "--real_settle_seconds",
        type=float,
        default=None,
        help="실제 로봇 action 전송 후 대기 시간. 생략하면 --settle_seconds_per_action 값을 사용합니다.",
    )
    parser.add_argument("--real_go_home_on_exit", action="store_true", help="종료 시 실제 로봇을 home 자세로 보냅니다.")
    parser.add_argument(
        "--no_randomize_box",
        action="store_true",
        help="Legacy name. Disables randomization for all four colored cylinders.",
    )
    parser.add_argument(
        "--no_randomize_objects",
        action="store_true",
        help="Disables randomization for all four colored cylinders.",
    )

    parser.add_argument("--use_ssh_tunnel", action="store_true", help="Connect to the inference server through SSH local port forwarding")
    parser.add_argument("--ssh_host", type=str, default="qlak315.iptime.org")
    parser.add_argument("--ssh_port", type=int, default=24100)
    parser.add_argument("--ssh_user", type=str, default="root")
    parser.add_argument("--ssh_password", type=str, default=None, help="Prefer OPENVLA_SSH_PASSWORD or --ssh_ask_password")
    parser.add_argument("--ssh_ask_password", action="store_true", help="Prompt for the SSH password interactively")
    parser.add_argument("--remote_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--remote_server_port", type=int, default=8000)
    parser.add_argument("--local_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--local_server_port", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with maybe_tunnel_context(args) as tunnel:
        server_url = build_server_url(args, tunnel)

        if tunnel is not None:
            print(
                f"[SSH] {args.local_server_host}:{tunnel.local_bind_port} -> "
                f"{args.remote_server_host}:{args.remote_server_port}"
            )

        rollout(
            xml_path=args.xml_path,
            server_url=server_url,
            instruction=args.instruction,
            unnorm_key=args.unnorm_key,
            output_dir=args.output_dir,
            episode_id=args.episode_id,
            max_steps=args.max_steps,
            use_viewer=args.use_viewer,
            camera_name=args.camera_name,
            speed=args.speed,
            settle_seconds_per_action=args.settle_seconds_per_action,
            initial_settle_seconds=args.initial_settle_seconds,
            delta_scale=args.delta_scale,
            randomize_objects=not (args.no_randomize_box or args.no_randomize_objects),
            request_timeout=args.request_timeout,
            max_delta_xyz=args.max_delta_xyz,
            target_color_arg=args.target_color,
            instruction_template=args.instruction_template,
            seed=args.seed,
            object_x_range=tuple(args.object_x_range),
            object_y_range=tuple(args.object_y_range),
            min_object_distance=args.min_object_distance,
            use_real_robot=args.use_real_robot,
            allow_sim_only_on_hw_fail=args.allow_sim_only_on_hw_fail,
            real_initial_wait_seconds=args.real_initial_wait_seconds,
            real_settle_seconds=args.real_settle_seconds,
            real_go_home_on_exit=args.real_go_home_on_exit,
        )


if __name__ == "__main__":
    main()
