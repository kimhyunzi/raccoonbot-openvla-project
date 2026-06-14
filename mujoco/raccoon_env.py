import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image


@dataclass
class WorkspaceBounds:
    # Conservative workspace for stable IK on this robot.
    x_min: float = -0.16
    x_max: float = 0.16
    y_min: float = 0.11
    y_max: float = 0.18
    z_min: float = 0.02
    z_max: float = 0.10

    def clip_xyz(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return (
            float(np.clip(x, self.x_min, self.x_max)),
            float(np.clip(y, self.y_min, self.y_max)),
            float(np.clip(z, self.z_min, self.z_max)),
        )


class SyncSimRaccoonEnv:
    """
    MuJoCo rollout environment for OpenVLA delta-action testing.

    Supported actions:
      - absolute 4D waypoint action: [x, y, z, gripper]
      - OpenVLA 7D delta action: [dx, dy, dz, droll, dpitch, dyaw, gripper]

    Notes:
      - This robot only uses xyz + gripper for execution.
      - Rotation deltas are ignored because the 4-axis structure does not support
        full 6D end-effector control.
      - IK failures are reduced with conservative workspace clipping and fallback targets.
    """

    MAX_SPEEDS = [2.2, 2.3, 2.3, 2.3]
    GRIPPER_SPEED = 15.0

    # IK link lengths; matches the user's original code convention (cm-scale values).
    L1, L2, L3, L4 = 8.25, 10.0, 10.0, 8.0

    MODE_POSITION = 0
    MODE_VELOCITY = 1

    GRIP_OPEN = 0.15701
    GRIP_CLOSE = -0.85

    GRIP_MODE_FREE = 0
    GRIP_MODE_HORZ = 1
    GRIP_MODE_VERT = 2

    def __init__(
        self,
        xml_path: str,
        image_size: Tuple[int, int] = (256, 256),
        camera_name: Optional[str] = "front_view",
        use_viewer: bool = False,
        workspace: Optional[WorkspaceBounds] = None,
    ) -> None:
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"xml 파일을 찾을 수 없습니다: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=image_size[1], width=image_size[0])
        self.camera_name = camera_name
        self.use_viewer = use_viewer
        self.workspace = workspace or WorkspaceBounds()

        self.viewer = None
        if self.use_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.target_angles = [0.0] * 4
        self.current_setpoints = [0.0] * 5
        self.joint_velocities = [0.0] * 4
        self.joint_control_mode = [self.MODE_POSITION] * 4
        self.gripper_target = self.GRIP_OPEN
        self.gripper_mode = self.GRIP_MODE_FREE

        for i in range(4):
            self.joint_velocities[i] = self.MAX_SPEEDS[i] * 0.7

        self.reset_episode(0.15, 0.15, 0.0)

    # ---------- kinematics / commands ----------

    def _calc_inv_kinematics(self, x: float, y: float, z: float) -> Optional[List[float]]:
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float))):
            return None

        if not ((-28.0 <= x <= 28.0) and (-15.0 <= y <= 28.0) and (0.0 <= z <= 36.25)):
            return None

        # Original convention
        x, y = y, -x

        th1 = math.atan2(y, x)
        c1 = math.cos(th1)
        s1 = math.sin(th1)

        # Wrist center
        wx = x - self.L4 * c1
        wy = y - self.L4 * s1
        wz = z - self.L1

        c3 = (wx * wx + wy * wy + wz * wz - self.L2 * self.L2 - self.L3 * self.L3) / (2.0 * self.L2 * self.L3)

        # reject clearly invalid points, clamp near-boundary numeric noise
        if c3 < -1.0001 or c3 > 1.0001:
            return None
        c3 = float(np.clip(c3, -1.0, 1.0))

        s3_abs = math.sqrt(max(0.0, 1.0 - c3 * c3))
        s3_candidates = [-s3_abs, s3_abs]

        th1_deg = math.degrees(th1)

        for s3 in s3_candidates:
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

    def degree_to(self, joints: Sequence[int], degrees: Sequence[float], speed: int = 70) -> None:
        j_list = list(joints) if isinstance(joints, (list, tuple)) else [joints]
        d_list = list(degrees) if isinstance(degrees, (list, tuple)) else [degrees]
        if len(d_list) == 1 and len(j_list) > 1:
            d_list = d_list * len(j_list)

        for j, deg in zip(j_list, d_list):
            idx = j - 1
            if 0 <= idx < 4:
                self.joint_control_mode[idx] = self.MODE_POSITION
                self.target_angles[idx] = np.radians(deg)
                percent = max(float(speed), 0.0)       
                self.joint_velocities[idx] = (percent / 100.0) * self.MAX_SPEEDS[idx]

    def move_to(self, x_cm: float, y_cm: float, z_cm: float, speed: int = 70) -> None:
        angles = self._calc_inv_kinematics(x_cm, y_cm, z_cm)
        if angles is None:
            raise ValueError(f"도달할 수 없는 좌표입니다: ({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f}) cm")
        self.degree_to([1, 2, 3, 4], angles[:4], speed)

    def open_gripper(self) -> None:
        self.gripper_target = self.GRIP_OPEN

    def close_gripper(self) -> None:
        self.gripper_target = self.GRIP_CLOSE

    def lockh(self) -> None:
        self.gripper_mode = self.GRIP_MODE_HORZ

    def lockv(self) -> None:
        self.gripper_mode = self.GRIP_MODE_VERT

    def unlock(self) -> None:
        if self.gripper_mode != self.GRIP_MODE_FREE:
            self.target_angles[3] = self.data.qpos[3]
            self.gripper_mode = self.GRIP_MODE_FREE

    def execute_absolute_action4(self, action: Sequence[float], speed: int = 70) -> None:
        target_x, target_y, target_z, gripper = action
        self.move_to(float(target_x) * 100.0, float(target_y) * 100.0, float(target_z) * 100.0, speed=speed)
        if float(gripper) >= 0.5:
            self.close_gripper()
        else:
            self.open_gripper()

    def execute_delta_action7(
        self,
        action: Sequence[float],
        speed: int = 70,
        max_delta_xyz: float = 0.01,
        delta_scale: float = 1.0,
        shrink_ratio: float = 0.15,
        max_retries: int = 3,
        use_rotation_projection: bool = True,
        rot_to_trans: Tuple[float, float, float] = (0.15, 0.10, 0.10),
        enable_safe_height_fallback: bool = True,
        safe_height_z: float = 0.08,
        min_close_height_z: float = 0.035,
    ) -> Dict[str, object]:
        """
        OpenVLA 7D delta action -> RaccoonBot 4DOF 실행.

        개선점 (Rotation-aware 7D-to-4DOF Projection with IK-aware Safe Fallback):
          - use_rotation_projection=True: 회전 성분(droll,dpitch,dyaw)을 폐기하지 않고
            작은 Cartesian 보정으로 projection (정책 의도 보존용 heuristic; 운동학적 변환 아님)
          - use_rotation_projection=False: 기존 baseline과 정확히 동일 (회전 폐기, 비교 실험용)
          - IK 실패 시: shrink retry -> safe-height fallback -> ValueError 순으로 단계적 fallback
          - gripper: EE가 너무 높으면 (target_z > min_close_height_z) close를 막고 open 유지

        하위호환: 기존 인자는 그대로, optional 인자만 뒤에 추가.
        기존 client가 쓰던 key (final_delta_xyz / actual_move_xyz / target_xyz /
        gripper_cmd / retry_count) 는 모두 유지.
        """
        if len(action) < 7:
            raise ValueError(f"action 길이가 부족합니다: len={len(action)}, action={action}")

        dx, dy, dz, droll, dpitch, dyaw, gripper = [float(v) for v in action[:7]]
        raw_delta_xyz = [dx, dy, dz]
        raw_rotation_rpy = [droll, dpitch, dyaw]

        # ---- Stage 1: rotation-aware projected delta ----
        # helper가 scaling(delta_scale) + rotation projection + clip 까지 모두 처리한다.
        proj_dx, proj_dy, proj_dz, projection_info = self._project_action7_to_raccoon4(
            dx=dx, dy=dy, dz=dz,
            droll=droll, dpitch=dpitch, dyaw=dyaw,
            max_delta_xyz=max_delta_xyz,
            delta_scale=delta_scale,
            use_rotation_projection=use_rotation_projection,
            rot_to_trans=rot_to_trans,
        )

        ee_x, ee_y, ee_z = self.get_ee_pose()

        # ---- Stage 2: workspace clipping bounds ----
        safe_x_min, safe_x_max = -0.18, 0.18
        safe_y_min, safe_y_max = 0.05, 0.25
        safe_z_min, safe_z_max = 0.02, 0.11

        cur_dx, cur_dy, cur_dz = proj_dx, proj_dy, proj_dz
        tried_results = []
        chosen_target = None
        chosen_delta = None
        fallback_used = "none"
        last_exc = None

        # ---- Stage 3+4: IK 시도 + delta shrink retry ----
        for retry_idx in range(max_retries + 1):
            tx = float(np.clip(ee_x + cur_dx, safe_x_min, safe_x_max))
            ty = float(np.clip(ee_y + cur_dy, safe_y_min, safe_y_max))
            tz = float(np.clip(ee_z + cur_dz, safe_z_min, safe_z_max))
            try:
                self.move_to(tx * 100.0, ty * 100.0, tz * 100.0, speed=speed)
                chosen_target = (tx, ty, tz)
                chosen_delta = (cur_dx, cur_dy, cur_dz)
                tried_results.append({
                    "retry_index": retry_idx, "stage": "shrink_retry",
                    "delta_xyz": [cur_dx, cur_dy, cur_dz],
                    "target_xyz": [tx, ty, tz], "ok": True, "error": None,
                })
                break
            except Exception as exc:
                last_exc = exc
                tried_results.append({
                    "retry_index": retry_idx, "stage": "shrink_retry",
                    "delta_xyz": [cur_dx, cur_dy, cur_dz],
                    "target_xyz": [tx, ty, tz], "ok": False, "error": str(exc),
                })
            cur_dx *= (1.0 - shrink_ratio)
            cur_dy *= (1.0 - shrink_ratio)
            cur_dz *= (1.0 - shrink_ratio)

        # ---- Stage 5: safe-height fallback ----
        # shrink retry가 전부 실패하면, z를 안전 높이로 올리고 xy 이동 위주로 재시도.
        # 낮은 z 에서의 IK 특이점/도달불가를 회피하기 위한 안정화 단계.
        if chosen_target is None and enable_safe_height_fallback:
            fb_tz = float(np.clip(safe_height_z, safe_z_min, safe_z_max))
            # xy는 projected delta의 절반만 적용해 보수적으로 이동
            fb_tx = float(np.clip(ee_x + proj_dx * 0.5, safe_x_min, safe_x_max))
            fb_ty = float(np.clip(ee_y + proj_dy * 0.5, safe_y_min, safe_y_max))
            try:
                self.move_to(fb_tx * 100.0, fb_ty * 100.0, fb_tz * 100.0, speed=speed)
                chosen_target = (fb_tx, fb_ty, fb_tz)
                chosen_delta = (fb_tx - ee_x, fb_ty - ee_y, fb_tz - ee_z)
                fallback_used = "safe_height_xy"
                tried_results.append({
                    "retry_index": len(tried_results), "stage": "safe_height_fallback",
                    "delta_xyz": list(chosen_delta),
                    "target_xyz": [fb_tx, fb_ty, fb_tz], "ok": True, "error": None,
                })
            except Exception as exc:
                last_exc = exc
                tried_results.append({
                    "retry_index": len(tried_results), "stage": "safe_height_fallback",
                    "delta_xyz": [fb_tx - ee_x, fb_ty - ee_y, fb_tz - ee_z],
                    "target_xyz": [fb_tx, fb_ty, fb_tz], "ok": False, "error": str(exc),
                })

        # ---- Stage 6: 그래도 실패하면 명확한 ValueError ----
        if chosen_target is None:
            raise ValueError(
                f"IK fail | ee=({ee_x:.4f},{ee_y:.4f},{ee_z:.4f}) | "
                f"raw_delta=({raw_delta_xyz[0]:.4f},{raw_delta_xyz[1]:.4f},{raw_delta_xyz[2]:.4f}) | "
                f"projected_delta=({proj_dx:.4f},{proj_dy:.4f},{proj_dz:.4f}) | "
                f"retries={max_retries} | safe_height_fallback={enable_safe_height_fallback} | "
                f"last_error={last_exc}"
            )

        tx, ty, tz = chosen_target
        final_dx, final_dy, final_dz = chosen_delta

        # ---- gripper: height-aware close 차단 ----
        gripper_blocked_by_height = False
        if gripper >= 0.5:
            if tz > min_close_height_z:
                # 너무 높은 위치에서의 close는 위험/무의미 -> open 유지
                self.open_gripper()
                gripper_action = "open"
                gripper_blocked_by_height = True
            else:
                self.close_gripper()
                gripper_action = "close"
        else:
            self.open_gripper()
            gripper_action = "open"

        actual_move = [tx - ee_x, ty - ee_y, tz - ee_z]

        return {
            "success": True,
            "ee_pose_before": [ee_x, ee_y, ee_z],
            "raw_action": [float(v) for v in action[:7]],
            "raw_delta_xyz": raw_delta_xyz,
            "raw_rotation_rpy": raw_rotation_rpy,
            "applied_delta_xyz": [proj_dx, proj_dy, proj_dz],
            "final_delta_xyz": [final_dx, final_dy, final_dz],
            "target_xyz": [tx, ty, tz],
            "actual_move_xyz": actual_move,
            "projection_info": projection_info,
            "fallback_used": fallback_used,
            "gripper_cmd": gripper,
            "gripper_action": gripper_action,
            "gripper_blocked_by_height": gripper_blocked_by_height,
            "tried_results": tried_results,
            "retry_count": len(tried_results) - 1,
        }

    # ---------- synchronous stepping ----------

    def _apply_controls_once(self) -> None:
        dt = self.model.opt.timestep
        for i in range(4):
            if i == 3 and self.gripper_mode != self.GRIP_MODE_FREE:
                base_angle = -(self.current_setpoints[1] + self.current_setpoints[2])
                if self.gripper_mode == self.GRIP_MODE_HORZ:
                    desired = base_angle - np.radians(90)
                else:
                    desired = base_angle - np.radians(180)
                error = desired - self.current_setpoints[i]
                speed_rad_s = self.MAX_SPEEDS[i]
                limit_step = speed_rad_s * dt
                step = np.clip(error, -limit_step, limit_step)
                self.current_setpoints[i] += step
            else:
                if self.joint_control_mode[i] == self.MODE_VELOCITY:
                    self.current_setpoints[i] += self.joint_velocities[i] * dt
                else:
                    error = self.target_angles[i] - self.current_setpoints[i]
                    if abs(error) > 1e-4:
                        max_step = abs(self.joint_velocities[i]) * dt
                        step_val = np.clip(error, -max_step, max_step)
                        self.current_setpoints[i] += step_val

            joint_id = self.model.actuator_trnid[i, 0]
            rng = self.model.jnt_range[joint_id]
            self.current_setpoints[i] = np.clip(self.current_setpoints[i], rng[0], rng[1])
            self.data.ctrl[i] = self.current_setpoints[i]

        try:
            touch_l = self.data.sensor("sensor_L").data[0]
            touch_r = self.data.sensor("sensor_R").data[0]
            is_touched = (touch_l > 0.1) and (touch_r > 0.1)
        except Exception:
            is_touched = False

        if self.gripper_target == self.GRIP_CLOSE and is_touched:
            self.gripper_target = self.data.qpos[4] - 0.028

        g_err = self.gripper_target - self.current_setpoints[4]
        if abs(g_err) > 1e-4:
            g_step = self.GRIPPER_SPEED * dt
            g_move = np.clip(g_err, -g_step, g_step)
            self.current_setpoints[4] += g_move

        self.data.ctrl[4] = self.current_setpoints[4]

    def step_n(self, n_steps: int) -> None:
        for _ in range(int(n_steps)):
            self._apply_controls_once()
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

    def steps_for_seconds(self, seconds: float) -> int:
        return max(1, int(round(seconds / self.model.opt.timestep)))

    def settle_steps(self, seconds: float = 2.0) -> None:
        self.step_n(self.steps_for_seconds(seconds))

    # ---------- rendering / state ----------

    def get_robot_state(self) -> Dict[str, List[float]]:
        joint_angles = [float(self.data.qpos[i]) for i in range(4)]
        gripper_state = float(self.data.qpos[4])
        return {"joint_angles": joint_angles, "gripper_state": gripper_state}

    def get_object_pose(self, body_name: str = "target_object") -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")
        pos = self.data.xpos[body_id].copy()
        xmat = self.data.xmat[body_id].reshape(3, 3).copy()
        yaw = math.atan2(xmat[1, 0], xmat[0, 0])
        return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float32)

    def render_rgb(self) -> np.ndarray:
        cam_id = self.camera_name if self.camera_name is not None else -1
        self.renderer.update_scene(self.data, camera=cam_id)
        image = self.renderer.render()
        return image.copy()

    def get_ee_pose(self, body_name: str = "Link4") -> Tuple[float, float, float]:
        """
        Return EE pose in meters, using forward kinematics that matches _calc_inv_kinematics().

        Important:
        - Do NOT use MuJoCo body xpos of Link4 here.
        - This computes the same endpoint convention assumed by move_to()/IK.
        """

        # Current joint angles in radians
        th1 = float(self.data.qpos[0])
        th2 = float(self.data.qpos[1])
        th3 = float(self.data.qpos[2])

        # Internal planar coordinates consistent with the IK derivation
        # r = wrist-center radial distance in the rotated base plane
        r = -self.L2 * math.sin(th2) - self.L3 * math.sin(th2 + th3)
        z = self.L1 + self.L2 * math.cos(th2) + self.L3 * math.cos(th2 + th3)

        # Add L4 offset exactly the same way IK assumes it
        r_tip = r + self.L4

        # Convert internal coordinates back to the external/world convention
        x_cm = -math.sin(th1) * r_tip
        y_cm =  math.cos(th1) * r_tip
        z_cm = z

        # return in meters (because the rest of your pipeline uses meters for ee_pose)
        return x_cm / 100.0, y_cm / 100.0, z_cm / 100.0

    def get_observation(self) -> Dict[str, object]:
        rs = self.get_robot_state()
        obj = self.get_object_pose()
        img = self.render_rgb()
        ee_pose = list(self.get_ee_pose())
        return {
            "image": img,
            "joint_angles": rs["joint_angles"],
            "gripper_state": rs["gripper_state"],
            "object_pose": obj,
            "ee_pose": ee_pose,
        }

    def save_current_frame(self, path: str) -> None:
        Image.fromarray(self.render_rgb()).save(path)

    # ---------- reset / success ----------

    def reset_object_pose(
        self,
        body_name: str = "target_object",
        x: float = 0.15,
        y: float = 0.15,
        z: float = 0.02,
        yaw: float = 0.0,
    ) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")

        jnt_adr = self.model.body_jntadr[body_id]
        jnt_num = self.model.body_jntnum[body_id]
        if jnt_num < 1:
            raise ValueError(f"{body_name} has no joint")

        joint_id = jnt_adr
        qpos_adr = self.model.jnt_qposadr[joint_id]
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        self.data.qpos[qpos_adr:qpos_adr + 7] = np.array([x, y, z, qw, 0.0, 0.0, qz], dtype=np.float64)

        qvel_adr = self.model.jnt_dofadr[joint_id]
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    def reset_episode(self, box_x: float, box_y: float, box_yaw: float) -> None:
        home = np.radians([0.0, -10.0, -140.0, 60.0])

        for i in range(4):
            self.data.qpos[i] = home[i]
            self.data.ctrl[i] = home[i]
            self.current_setpoints[i] = home[i]
            self.target_angles[i] = home[i]
            self.joint_control_mode[i] = self.MODE_POSITION

        self.data.qvel[:] = 0.0
        self.data.qpos[4] = self.GRIP_OPEN
        self.data.ctrl[4] = self.GRIP_OPEN
        self.current_setpoints[4] = self.GRIP_OPEN
        self.gripper_target = self.GRIP_OPEN
        self.gripper_mode = self.GRIP_MODE_FREE

        self.reset_object_pose("target_object", x=box_x, y=box_y, z=0.02, yaw=box_yaw)
        mujoco.mj_forward(self.model, self.data)
        self.step_n(20)

    def randomize_object_pose(self) -> Tuple[float, float, float]:
        box_x = float(np.random.uniform(-0.16, 0.16))
        box_y = float(np.random.uniform(0.12, 0.18))
        box_yaw = float(np.random.uniform(-np.pi / 4, np.pi / 4))
        self.reset_episode(box_x, box_y, box_yaw)
        return box_x, box_y, box_yaw

    def is_success(self, goal_x: float, goal_y: float, tolerance: float = 0.03) -> bool:
        object_pose = self.get_object_pose()
        obj_xy = np.array(object_pose[:2], dtype=np.float32)
        goal_xy = np.array([goal_x, goal_y], dtype=np.float32)
        dist = np.linalg.norm(obj_xy - goal_xy)
        return bool(dist < tolerance)

    def parse_instruction_to_goal(self, instruction: str) -> Tuple[float, float]:
        text = instruction.lower().strip()
        if "left" in text:
            return -0.15, 0.15
        if "right" in text:
            return 0.15, 0.15
        if "forward" in text or "front" in text:
            return 0.15, 0.15
        if "backward" in text:
            return -0.10, -0.10
        if "center" in text:
            return 0.0, 0.15
        return -0.15, 0.15

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def debug_check_current_ee_reachable(self):
        ee_x, ee_y, ee_z = self.get_ee_pose()
        x_cm, y_cm, z_cm = ee_x * 100.0, ee_y * 100.0, ee_z * 100.0
        print(f"[DEBUG] current ee from get_ee_pose = ({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f}) cm")
        angles = self._calc_inv_kinematics(x_cm, y_cm, z_cm)
        print(f"[DEBUG] IK(current ee) = {angles}")
        return angles
    def _project_action7_to_raccoon4(
        self,
        dx: float, dy: float, dz: float,
        droll: float, dpitch: float, dyaw: float,
        max_delta_xyz: float,
        delta_scale: float,
        use_rotation_projection: bool,
        rot_to_trans: Tuple[float, float, float],
    ):
        """
        OpenVLA 7D delta -> RaccoonBot 실행가능 3D translation delta로 projection.

        반환: (proj_dx, proj_dy, proj_dz, projection_info)

        처리 순서:
          1. translation 성분에 delta_scale 적용  (기존 baseline과 동일하게 유지)
          2. use_rotation_projection=True 이면 회전 성분을 작은 Cartesian 보정으로 추가
               dx' += alpha * dyaw  * max_delta_xyz
               dy' += beta  * droll * max_delta_xyz
               dz' += gamma * dpitch* max_delta_xyz
             (운동학적 변환이 아니라, 폐기되던 회전 정보를 약하게 보존하는 heuristic)
          3. [-max_delta_xyz, max_delta_xyz] 로 clip

        use_rotation_projection=False 이면 rotation 보정 = 0 이므로
        결과가 기존 baseline (회전 폐기 + scale + clip) 과 정확히 동일하다.
        """
        alpha, beta, gamma = (float(rot_to_trans[0]), float(rot_to_trans[1]), float(rot_to_trans[2]))

        # 1. translation scaling
        sdx = dx * delta_scale
        sdy = dy * delta_scale
        sdz = dz * delta_scale

        # 2. rotation -> translation 보정
        if use_rotation_projection:
            rot_dx = alpha * dyaw * max_delta_xyz
            rot_dy = beta * droll * max_delta_xyz
            rot_dz = gamma * dpitch * max_delta_xyz
        else:
            rot_dx = rot_dy = rot_dz = 0.0

        before_clip = [sdx + rot_dx, sdy + rot_dy, sdz + rot_dz]

        # 3. clip
        proj_dx = float(np.clip(before_clip[0], -max_delta_xyz, max_delta_xyz))
        proj_dy = float(np.clip(before_clip[1], -max_delta_xyz, max_delta_xyz))
        proj_dz = float(np.clip(before_clip[2], -max_delta_xyz, max_delta_xyz))

        projection_info = {
            "raw_delta_xyz": [dx, dy, dz],
            "raw_rotation_rpy": [droll, dpitch, dyaw],
            "scaled_delta_xyz": [sdx, sdy, sdz],
            "rotation_projection_xyz": [rot_dx, rot_dy, rot_dz],
            "projected_delta_xyz_before_clip": list(before_clip),
            "projected_delta_xyz": [proj_dx, proj_dy, proj_dz],
            "rot_to_trans": [alpha, beta, gamma],
            "use_rotation_projection": bool(use_rotation_projection),
            "max_delta_xyz": float(max_delta_xyz),
            "delta_scale": float(delta_scale),
        }
        return proj_dx, proj_dy, proj_dz, projection_info
