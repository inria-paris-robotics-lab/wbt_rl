#!/usr/bin/env python3
"""Bridge node: unitree_simulation (PyBullet) ↔ holosoma inference policy via ROS2.

Translates between the unitree_control_interface (27 actuated DOF) and
the holosoma policy format (27 or 29 DOF), bridging these ROS2 topics:

  Simulation → Bridge → holosoma policy:
    /lowstate  ──►  /holosoma/low_state  (sensor_msgs/JointState)  joint pos & vel
    /lowstate  ──►  /holosoma/imu        (sensor_msgs/Imu)          quaternion & gyro

  holosoma policy → Bridge → Simulation:
    /holosoma/low_cmd   ──►  /lowcmd  (via G1ControlInterface)
    /holosoma/pd_gains  ──►  /lowcmd  (via G1ControlInterface)

The bridge auto-detects the policy DOF mode from the first /holosoma/low_cmd
message:
  - 27 DOF (G1 base): direct pass-through, no conversion needed.
  - 29 DOF (G1 pro):  waist_roll (idx 13) and waist_pitch (idx 14) are dropped.

Joint mapping (G1 27-DOF hardware ↔ 29-DOF policy):
  unitree[0:13]  ↔  holosoma[0:13]   left leg + right leg + waist_yaw
  0.0 (fixed)    ↔  holosoma[13]     waist_roll  (locked in mode_machine=6)
  0.0 (fixed)    ↔  holosoma[14]     waist_pitch (locked in mode_machine=6)
  unitree[13:27] ↔  holosoma[15:29]  left arm + right arm
"""

import argparse
import threading

import numpy as np
import rclpy
from loguru import logger
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from std_msgs.msg import Empty, Bool
from sensor_msgs.msg import Imu, JointState
from unitree_hg.msg import LowState

try:
    import zmq
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyzmq"])
    import zmq

from unitree_control_interface_py import G1ControlInterface

# ── Joint dimension constants ──────────────────────────────────────────────────
N_HOLOSOMA = 29        # canonical full G1 layout (waist_yaw=12, waist_roll=13, waist_pitch=14)
_LOCKED_WAIST = [13, 14]   # waist_roll, waist_pitch — locked in the 27-DOF variant
# Actuated unitree DOF (27 or 29). Overwritten from the --dof CLI arg in main();
# the 27 default preserves the previous behaviour. Threaded from deploy.py --g1-dof.
N_UNITREE  = 27

# Joint limits in 27-DOF unitree order (from g1_default_limits.yaml) — for logging only.
_Q_MIN = np.array([-2.5307,-0.5236,-2.7576,-0.087267,-0.88,-0.2618,
                   -2.5307,-2.9671,-2.7576,-0.087267,-0.88,-0.2618,
                   -2.618,-3.0892,-1.5882,-2.618,-1.0472,-1.972222054,-1.614429558,-1.614429558,
                   -3.0892,-2.2515,-2.618,-1.0472,-1.972222054,-1.614429558,-1.614429558])
_Q_MAX = np.array([2.8798,2.9671,2.7576,2.8798,0.53,0.2618,
                   2.8798,0.5236,2.7576,2.8798,0.53,0.2618,
                   2.618,2.6704,2.2515,2.618,2.0944,1.972222054,1.614429558,1.614429558,
                   2.6704,1.5882,2.618,2.0944,1.972222054,1.614429558,1.614429558])
_JOINT_NAMES_27 = [
    "left_hip_pitch","left_hip_roll","left_hip_yaw","left_knee","left_ankle_pitch","left_ankle_roll",
    "right_hip_pitch","right_hip_roll","right_hip_yaw","right_knee","right_ankle_pitch","right_ankle_roll",
    "waist_yaw",
    "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow","left_wrist_roll","left_wrist_pitch","left_wrist_yaw",
    "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow","right_wrist_roll","right_wrist_pitch","right_wrist_yaw",
]

# Safe fallback for goto_config before the policy's startup_pose is received.
# Matches G1ControlInterface.Kp_static so there is no gain discontinuity.
_G1_SAFE_Q  = [
    -0.5, 0.0, 0.0, 1.0, -0.5, 0.0,   # left  leg
    -0.5, 0.0, 0.0, 1.0, -0.5, 0.0,   # right leg
     0.0,                               # waist_yaw
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left  arm
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # right arm
]


def _to_canon(arr: np.ndarray, n: int) -> np.ndarray:
    """Expand an n-DOF array (27 or 29) to the 29-DOF canonical layout.

    For 27 DOF the locked waist (idx 13, 14) is filled with 0.0.
    """
    arr = np.asarray(arr, dtype=float)
    if n >= N_HOLOSOMA:
        return arr[:N_HOLOSOMA].copy()
    out = np.zeros(N_HOLOSOMA)
    out[:13] = arr[:13]    # left leg + right leg + waist_yaw
    out[15:] = arr[13:n]   # left arm + right arm (waist_roll/pitch left at 0)
    return out


def _from_canon(arr29: np.ndarray, n: int) -> np.ndarray:
    """Contract the 29-DOF canonical layout to an n-DOF array (27 or 29)."""
    arr29 = np.asarray(arr29, dtype=float)
    if n >= N_HOLOSOMA:
        return arr29.copy()
    return np.delete(arr29, _LOCKED_WAIST)


def _unitree_to_holosoma(arr: np.ndarray, n_policy: int) -> np.ndarray:
    """Map a unitree array (N_UNITREE DOF) → policy DOF format (27 or 29)."""
    return _from_canon(_to_canon(arr, N_UNITREE), n_policy)


def _holosoma_to_unitree(arr: np.ndarray, n_policy: int) -> np.ndarray:
    """Map a policy DOF array (27 or 29) → unitree DOF (N_UNITREE).

    Locked waist positions/torques are held at 0 (see _holosoma_to_unitree_gains
    for the gains, which instead reuse the waist_yaw value so the joints stay rigid).
    """
    return _from_canon(_to_canon(arr, n_policy), N_UNITREE)


def _holosoma_to_unitree_gains(arr: np.ndarray, n_policy: int) -> np.ndarray:
    """Like _holosoma_to_unitree but for kp/kd: when a 27-DOF policy drives a
    29-DOF robot, the locked waist joints (idx 13, 14) inherit the waist_yaw gains
    (idx 12) so they are held rigid at 0 instead of going limp (kp = 0)."""
    canon = _to_canon(arr, n_policy)
    if n_policy < N_HOLOSOMA:            # policy provides no waist_roll/pitch gains
        canon[13] = canon[14] = canon[12]
    return _from_canon(canon, N_UNITREE)


class UnitreePybulletBridgeNode(Node):
    """ROS2 bridge connecting unitree_simulation (PyBullet) to a holosoma inference policy.

    Startup sequence:
      1. Robot moves to standing configuration (via G1ControlInterface start routine, ~5 s).
      2. Bridge auto-unlocks with a 1 s transition.
      3. Policy commands arriving on /holosoma/low_cmd are forwarded to the simulation.
    """

    # holosoma G1 joint names in 29-DOF order (with _joint suffix used by holosoma configs)
    _JOINT_NAMES_29 = [
        "left_hip_pitch_joint",       "left_hip_roll_joint",       "left_hip_yaw_joint",
        "left_knee_joint",            "left_ankle_pitch_joint",    "left_ankle_roll_joint",
        "right_hip_pitch_joint",      "right_hip_roll_joint",      "right_hip_yaw_joint",
        "right_knee_joint",           "right_ankle_pitch_joint",   "right_ankle_roll_joint",
        "waist_yaw_joint",            "waist_roll_joint",          "waist_pitch_joint",
        "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",  "left_shoulder_yaw_joint",
        "left_elbow_joint",           "left_wrist_roll_joint",     "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint",          "right_wrist_roll_joint",    "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    # holosoma G1 joint names in 27-DOF order (waist_roll + waist_pitch removed)
    _JOINT_NAMES_27 = [
        n for n in _JOINT_NAMES_29 if n not in ("waist_roll_joint", "waist_pitch_joint")
    ]

    def __init__(self):
        super().__init__("unitree_pybullet_bridge")

        # ── holosoma publishers ──────────────────────────────────────────────
        self._state_pub = self.create_publisher(JointState, "/holosoma/low_state", 10)
        self._imu_pub   = self.create_publisher(Imu,        "/holosoma/imu",       10)
        self._unlock_pub = self.create_publisher(Empty,     "/unlock_base",        10)

        # ── IMU buffer — subscribe to /lowstate to access imu_state directly ─
        # (G1ControlInterface callback only exposes joint positions/velocities)
        self._imu_lock = threading.Lock()
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self._gyro = np.zeros(3)
        self.create_subscription(LowState, "/lowstate", self._lowstate_imu_cb, 10)

        # ── command buffers (populated by holosoma policy callbacks) ─────────
        self._cmd_lock     = threading.Lock()
        # ── Command buffers — initialised with safe fallback, overwritten by startup_pose ─
        # 27-DOF: _cmd_q was np.array(_G1_SAFE_Q) (27). Expanded to the actuated DOF
        # (waist held at 0 when 29).
        self._cmd_q        = _holosoma_to_unitree(np.array(_G1_SAFE_Q), 27)
        self._cmd_dq       = np.zeros(N_UNITREE)
        self._cmd_tau      = np.zeros(N_UNITREE)
        self._kp           = np.full(N_UNITREE, 75.0)   # matches Kp_static
        self._kd           = np.full(N_UNITREE,  1.0)   # matches Kd_static
        self._cmd_received = False
        self._n_policy: int = 0

        # Watchdog limits & joint names in unitree order, sized to the actuated DOF
        # (logging only). 27-DOF: were the module-level _Q_MIN/_Q_MAX/_JOINT_NAMES_27.
        # For 29-DOF, insert waist_roll/pitch (±0.52) at idx 13, 14.
        if N_UNITREE == N_HOLOSOMA:
            self._q_min = np.insert(_Q_MIN, 13, [-0.52, -0.52])
            self._q_max = np.insert(_Q_MAX, 13, [0.52, 0.52])
            self._names = _JOINT_NAMES_27[:13] + ["waist_roll", "waist_pitch"] + _JOINT_NAMES_27[13:]
        else:
            self._q_min, self._q_max, self._names = _Q_MIN, _Q_MAX, _JOINT_NAMES_27

        self.create_subscription(JointState, "/holosoma/low_cmd",  self._low_cmd_cb,  10)
        self.create_subscription(JointState, "/holosoma/pd_gains", self._pd_gains_cb, 10)

        # ── Startup pose from policy (TRANSIENT_LOCAL — received even if published before us) ─
        self._startup_q:  np.ndarray | None = None
        self._startup_kp: np.ndarray | None = None
        self._startup_kd: np.ndarray | None = None
        _startup_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(JointState, "/holosoma/startup_pose", self._startup_pose_cb, _startup_qos)

        # ── Watchdog QoS Patch ───────────────────────────────────────────────
        watchdog_qos = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, "/watchdog/is_safe", self._watchdog_patch_cb, watchdog_qos)

        # ── G1ControlInterface (handles watchdog, safety, start routine) ─────
        self._robot_if = G1ControlInterface(self, dof=N_UNITREE)
        self._robot_if.register_callback(self._joint_state_cb)
        self._unlocked = False

        # ── ZMQ clock publisher — feeds sim time to WBT policy ───────────────
        self._zmq_context = zmq.Context()
        self._zmq_clock_socket = self._zmq_context.socket(zmq.PUB)
        self._zmq_clock_socket.bind("tcp://*:5555")

        # ── Watchdog guard during goto_config ────────────────────────────────
        self._goto_config_done = False
        _orig_safety_cb = self._robot_if._watchdog_subscription.callback

        def _guarded_safety_cb(msg: Bool):
            if self._goto_config_done:
                _orig_safety_cb(msg)

        self._robot_if._watchdog_subscription.callback = _guarded_safety_cb

        # ── Wait for startup_pose then launch goto_config ────────────────────
        # Spin here until the policy publishes its startup pose (TRANSIENT_LOCAL
        # guarantees delivery even if published before this subscription).
        logger.info("Bridge waiting for /holosoma/startup_pose from policy...")
        self._startup_timer = self.create_timer(0.05, self._check_startup_ready)

    def _watchdog_patch_cb(self, msg: Bool):
        """Sync watchdog safety state — but only after goto_config is done.

        During goto_config the watchdog sees joints moving through the transition
        and can fire on bound violations. We hold is_safe=True until the robot is
        stable, then re-arm the watchdog and start honouring its signals.
        """
        if not hasattr(self, "_robot_if"):
            return
        if self._goto_config_done:
            self._robot_if._is_safe = msg.data
            if not msg.data:
                logger.warning("Watchdog fired — e-stop active.")

    # ── G1ControlInterface callback (~1 kHz) ──────────────────────────────────

    def _joint_state_cb(self, t: float, q, dq, ddq):
        """Receive joint state (27 DOF, URDF order); publish to holosoma and forward commands."""
        try:
            self._zmq_clock_socket.send_string(str(int(t * 1000)), zmq.NOBLOCK)
        except zmq.Again:
            pass

        # Log actual joint positions that exceed watchdog limits
        for i, (q_i, qmin, qmax, name) in enumerate(zip(q, self._q_min, self._q_max, self._names)):
            if q_i < qmin or q_i > qmax:
                logger.warning(f"Actual q out of limits: {name}[{i}] = {q_i:.4f} (limits [{qmin:.4f}, {qmax:.4f}])")

        q_np  = np.asarray(q)
        dq_np = np.asarray(dq)

        # Robot reached standing config: re-arm watchdog and start honouring it.
        if not self._goto_config_done and self._robot_if.can_be_unlocked():
            self._goto_config_done = True
            arm_msg = Bool()
            arm_msg.data = True
            self._robot_if._watchdog_publisher.publish(arm_msg)
            logger.info("Standing config reached — watchdog armed, monitoring active.")

        # Auto-unlock once the robot has reached the standing configuration
        if not self._unlocked and self._robot_if.can_be_unlocked():
            self._robot_if.unlock(transition_duration=1.0)
            self._unlocked = True
            logger.info(
                "Standing configuration reached — unlocked with 1 s transition. "
                "Waiting for policy commands on /holosoma/low_cmd ..."
            )
            self._unlock_pub.publish(Empty())

        # Publish joint state to holosoma (expand to policy DOF if needed)
        n_pol = self._n_policy if self._n_policy else N_HOLOSOMA  # default 29 before detection
        now = self.get_clock().now().to_msg()

        state_msg = JointState()
        state_msg.header.stamp = now
        # Names follow the POLICY DOF (n_pol), independent of the unitree DOF.
        state_msg.name         = self._JOINT_NAMES_29 if n_pol >= N_HOLOSOMA else self._JOINT_NAMES_27
        state_msg.position     = _unitree_to_holosoma(q_np, n_pol).tolist()
        state_msg.velocity     = _unitree_to_holosoma(dq_np, n_pol).tolist()
        self._state_pub.publish(state_msg)

        # Publish IMU data (latest values from _lowstate_imu_cb)
        with self._imu_lock:
            quat = self._quat.copy()
            gyro = self._gyro.copy()

        imu_msg = Imu()
        imu_msg.header.stamp       = now
        imu_msg.header.frame_id    = "base_link"
        # unitree quaternion is [w, x, y, z]; holosoma ros2_interface expects
        # sensor_msgs/Imu with (x, y, z, w) and internally converts to (w, x, y, z)
        imu_msg.orientation.w      = float(quat[0])
        imu_msg.orientation.x      = float(quat[1])
        imu_msg.orientation.y      = float(quat[2])
        imu_msg.orientation.z      = float(quat[3])
        imu_msg.angular_velocity.x = float(gyro[0])
        imu_msg.angular_velocity.y = float(gyro[1])
        imu_msg.angular_velocity.z = float(gyro[2])
        self._imu_pub.publish(imu_msg)

        # Hold standing pose immediately after unlock, even before the policy sends
        # its first command — avoids the gap where the robot has no PD control.
        if self._robot_if.can_be_controlled():
            with self._cmd_lock:
                q_cmd   = self._cmd_q.copy()
                dq_cmd  = self._cmd_dq.copy()
                tau_cmd = self._cmd_tau.copy()
                kp      = self._kp.copy()
                kd      = self._kd.copy()
            self._robot_if.send_command(
                q_cmd.tolist(), dq_cmd.tolist(), tau_cmd.tolist(),
                kp.tolist(), kd.tolist(),
            )

    def _lowstate_imu_cb(self, msg: LowState):
        """Buffer IMU data from the raw unitree LowState message.

        unitree imu_state.quaternion: [w, x, y, z]
        unitree imu_state.gyroscope:  angular velocity in base (local) frame [rad/s]
        """
        with self._imu_lock:
            q = msg.imu_state.quaternion
            self._quat[:] = [q[0], q[1], q[2], q[3]]
            g = msg.imu_state.gyroscope
            self._gyro[:] = [g[0], g[1], g[2]]

    # ── holosoma policy callbacks ─────────────────────────────────────────────

    def _detect_policy_dof(self, n_values: int) -> int:
        """Auto-detect policy DOF from the first message and log it once."""
        if self._n_policy:
            return self._n_policy
        # Policy DOF (27 base / 29 pro) is detected from the message length and is
        # independent of the unitree actuated DOF (N_UNITREE).
        self._n_policy = N_HOLOSOMA if n_values >= N_HOLOSOMA else 27
        logger.info(
            f"Policy DOF auto-detected: {self._n_policy} "
            f"({'G1 pro 29-DOF' if self._n_policy >= N_HOLOSOMA else 'G1 base 27-DOF'})"
        )
        return self._n_policy

    def _low_cmd_cb(self, msg: JointState):
        """Receive joint commands from holosoma policy (27 or 29 DOF) and map to 27 DOF."""
        with self._cmd_lock:
            n_pol = self._detect_policy_dof(len(msg.position))

            h_q = np.zeros(n_pol)
            n = min(len(msg.position), n_pol)
            h_q[:n] = msg.position[:n]
            self._cmd_q = _holosoma_to_unitree(h_q, n_pol)

            # Log any q_target that exceeds watchdog limits
            for i, (q, qmin, qmax, name) in enumerate(zip(self._cmd_q, self._q_min, self._q_max, self._names)):
                if q < qmin or q > qmax:
                    logger.warning(f"Policy q_target out of limits: {name}[{i}] = {q:.4f} (limits [{qmin:.4f}, {qmax:.4f}])")

            if msg.velocity:
                h_dq = np.zeros(n_pol)
                nv = min(len(msg.velocity), n_pol)
                h_dq[:nv] = msg.velocity[:nv]
                self._cmd_dq = _holosoma_to_unitree(h_dq, n_pol)

            if msg.effort:
                h_tau = np.zeros(n_pol)
                nt = min(len(msg.effort), n_pol)
                h_tau[:nt] = msg.effort[:nt]
                self._cmd_tau = _holosoma_to_unitree(h_tau, n_pol)

            self._cmd_received = True

    def _startup_pose_cb(self, msg: JointState):
        """Receive stiff startup configuration from the policy (TRANSIENT_LOCAL)."""
        if self._startup_q is not None:
            return
        n = len(msg.position)
        if n == 0:
            return
        n_pol = N_HOLOSOMA if n >= N_HOLOSOMA else 27
        if not self._n_policy:
            self._n_policy = n_pol
        h_pos = np.array(msg.position[:n_pol])
        h_kp  = np.array(msg.velocity[:n_pol]) if msg.velocity else np.full(n_pol, 75.0)
        h_kd  = np.array(msg.effort[:n_pol])   if msg.effort   else np.full(n_pol,  1.0)
        self._startup_q  = _holosoma_to_unitree(h_pos, n_pol)
        self._startup_kp = _holosoma_to_unitree_gains(h_kp, n_pol)
        self._startup_kd = _holosoma_to_unitree_gains(h_kd, n_pol)
        logger.info(
            f"Startup pose received ({n_pol} DOF): "
            f"q_legs={self._startup_q[:6].tolist()}, "
            f"kp_legs={self._startup_kp[:6].tolist()}, "
            f"kd_legs={self._startup_kd[:6].tolist()}"
        )

    def _check_startup_ready(self):
        """Wait for startup_pose from policy, then launch goto_config."""
        if self._startup_q is None:
            return
        self._startup_timer.cancel()
        # Try to patch goto_config gains so there is no discontinuity at unlock.
        try:
            self._robot_if.start_routine.kp = self._startup_kp.tolist()
            self._robot_if.start_routine.kd = self._startup_kd.tolist()
            logger.info("goto_config gains patched with policy stiff_startup gains")
        except AttributeError:
            logger.warning(
                "Cannot patch goto_config gains (start_routine not accessible) — "
                "gain jump may occur at unlock"
            )
        with self._cmd_lock:
            self._cmd_q = self._startup_q.copy()
            self._kp    = self._startup_kp.copy()
            self._kd    = self._startup_kd.copy()
        self._robot_if._is_safe = True
        self._robot_if.start_async(self._startup_q.tolist())
        logger.info(f"Starting goto_config → target q_legs={self._startup_q[:6].tolist()}")

    def _pd_gains_cb(self, msg: JointState):
        """Receive PD gains from holosoma policy (27 or 29 DOF) and map to 27 DOF."""
        with self._cmd_lock:
            n_pol = self._n_policy if self._n_policy else N_HOLOSOMA
            if msg.position:
                h_kp = np.zeros(n_pol)
                n = min(len(msg.position), n_pol)
                h_kp[:n] = msg.position[:n]
                self._kp = _holosoma_to_unitree_gains(h_kp, n_pol)
            if msg.velocity:
                h_kd = np.zeros(n_pol)
                nv = min(len(msg.velocity), n_pol)
                h_kd[:nv] = msg.velocity[:nv]
                self._kd = _holosoma_to_unitree_gains(h_kd, n_pol)


def main():
    parser = argparse.ArgumentParser(description="unitree_simulation <-> holosoma inference bridge")
    parser.add_argument(
        "--dof", type=int, choices=[27, 29], default=27,
        help="Actuated unitree DOF: 27 (waist_roll/pitch locked, mode_machine=6) or 29 (mode 5).",
    )
    args, _ = parser.parse_known_args()
    global N_UNITREE
    N_UNITREE = args.dof

    rclpy.init()
    node = UnitreePybulletBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
