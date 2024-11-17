#!/usr/bin/env python3
import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import threading

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState, Imu
from srobot_msg.msg import MotorCmds, MotorStates, MotorCmd, MotorState


class MojocoSim(Node):
    def __init__(self, xml_path, control_type="position"):
        super().__init__("mujoco_sim")

        self.get_logger().info('xml_path: "%s"' % xml_path)

        # Simulation timestep in seconds.
        self.dt: float = 0.002
        self.pause_flag: bool = True
        self.joint_state = JointState()
        self.joint_state.name = [
            "RF_Roll_Joint",
            "RF_Hip_Joint",
            "RF_Knee_Joint",
            "RF_Wheel_Joint",
            "LF_Roll_Joint",
            "LF_Hip_Joint",
            "LF_Knee_Joint",
            "LF_Wheel_Joint",
            "RH_Roll_Joint",
            "RH_Hip_Joint",
            "RH_Knee_Joint",
            "RH_Wheel_Joint",
            "LH_Roll_Joint",
            "LH_Hip_Joint",
            "LH_Knee_Joint",
            "LH_Wheel_Joint",
        ]

        # load mojoco model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.dt

        # for simulation data storage
        self.leg_datas = []
        self.ctrl = [0] * 16

        # # 更改actuator中kp的值
        # kp = 15.0
        # for i in range(model.nu):
        #     model.actuator_gainprm[i, 0] = kp  # 修改所有执行器的kp值
        #     model.actuator_biasprm[i, 1] = -kp

        # kp = 15.0
        # kd = 0.8
        # for i in range(model.nu):
        #     model.actuator_gainprm[i, 0] = kp  # 修改所有执行器的kp值
        #     model.actuator_biasprm[i, 1] = -kp
        #     model.actuator_biasprm[i, 2] = -kd
        # model.actuator_biasprm[-1, 2] = -0.2
        self.key_id = self.model.key("home").id

        self.joint_pos_noise_range = 0.005
        self.joint_vel_noise_range = 0.05
        self.quat_noise_range = 0.02
        self.gyro_noise_range = 0.06
        self.acc_noise_range = 0.0
        self.apply_noise = False

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=True, show_right_ui=True, key_callback=self.key_callback)

        # Reset the simulation to the initial keyframe.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        # Initialize the camera view to track the base link.
        mujoco.mjv_defaultCamera(self.viewer.cam)
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = self.model.body("Base_Link").id

        total_mass = sum(self.model.body_mass)
        print("total mass: ", total_mass)
        # * Set subscriber and publisher
        self.pub_joints = self.create_publisher(MotorStates, "/motor_states", 10)
        self.pub_rviz = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_imu = self.create_publisher(Imu, "/srobot_imu", 10)
        self.pub_sim_state = self.create_publisher(Bool, "/pause_flag", 10)

        if control_type == "position":
            self.subscription = self.create_subscription(MotorCmds, "/motor_cmds", self.control_position_callback, 10)
        elif control_type == "torque":
            self.subscription = self.create_subscription(MotorCmds, "/motor_cmds", self.control_torque_callback, 10)

        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

        self.thread1 = threading.Thread(target=self.sync_loop)
        self.thread1.start()

        timer_period = 1.0 / 500.0  # seconds
        self.timer = self.create_timer(timer_period, self.sim_step)
        self.i = 0
        self.last_time = time.time()

    def normalize_quaternion(self, quat):
        norm = np.linalg.norm(quat)
        return quat / norm

    def publish_msg(self):
        # * Publish joint positions and velocities
        qp = self.data.qpos[-16:].copy()
        qv = self.data.qvel[-16:].copy()

        if self.apply_noise:
            qp += np.random.normal(0.0, 1.0, 16) * self.joint_pos_noise_range
            qv += np.random.normal(0.0, 1.0, 16) * self.joint_vel_noise_range

        motor_state_msg = MotorStates()
        for i in range(16):
            obj = MotorState()
            obj.tau = 0.0
            obj.q = qp[i]
            obj.dq = qv[i]
            motor_state_msg.states.append(obj)
        self.pub_joints.publish(motor_state_msg)

        self.joint_state.position = 16 * [float(0.0)]
        self.joint_state.velocity = 16 * [float(0.0)]

        for i in range(16):
            self.joint_state.position[i] = float(qp[i])
            self.joint_state.velocity[i] = float(qv[i])
        t = self.get_clock().now()
        self.joint_state.header.stamp = t.to_msg()
        self.pub_rviz.publish(self.joint_state)

        # Publish body imu
        imu_msg = Imu()
        imu_msg.header.stamp = t.to_msg()
        quat = self.data.sensor("BodyQuat").data.copy()
        # print(quat)
        if self.apply_noise:
            quat += np.random.normal(0.0, 1.0, 4) * self.quat_noise_range
            quat = self.normalize_quaternion(quat)
        imu_msg.orientation.w = quat[0]
        imu_msg.orientation.x = quat[1]
        imu_msg.orientation.y = quat[2]
        imu_msg.orientation.z = quat[3]
        # print(imu_msg.orientation)
        gyro = self.data.sensor("BodyGyro").data.copy()
        if self.apply_noise:
            gyro += np.random.normal(0.0, 1.0, 3) * self.gyro_noise_range
        imu_msg.angular_velocity.x = gyro[0]
        imu_msg.angular_velocity.y = gyro[1]
        imu_msg.angular_velocity.z = gyro[2]
        acc = self.data.sensor("BodyAcc").data.copy()
        if self.apply_noise:
            acc += np.random.normal(0.0, 1.0, 3) * self.acc_noise_range
        imu_msg.linear_acceleration.x = acc[0]
        imu_msg.linear_acceleration.y = acc[1]
        imu_msg.linear_acceleration.z = acc[2]
        self.pub_imu.publish(imu_msg)

    def key_callback(self, keycode):
        self.viewer.lock()
        if chr(keycode) == " ":
            self.pause_flag = not self.pause_flag
            simState = Bool()
            simState.data = self.pause_flag
            self.pub_sim_state.publish(simState)

    def control_position_callback(self, data_msg):
        for i in range(16):
            self.ctrl[i] = data_msg.cmds[i].q
        self.data.ctrl[:] = self.ctrl

    def control_torque_callback(self, data_msg):
        for i in range(16):
            self.ctrl[i] = data_msg.cmds[i].tau
        self.data.ctrl[:] = self.ctrl

    def sync_loop(self):
        while self.viewer.is_running():
            self.viewer.sync()
            time.sleep(0.010)

    def sim_step(self):
        start_time = time.perf_counter()
        with self.viewer.lock():
            if not self.pause_flag:
                mujoco.mj_step(self.model, self.data)

                self.publish_msg()
            else:
                mujoco.mj_forward(self.model, self.data)

    def shutdown(self):
        # self.viewer.sync()
        self.thread1.join()

    def simulate(self):
        while self.viewer.is_running():
            start_time = time.perf_counter()
            with self.viewer.lock():
                if not self.pause_flag:
                    mujoco.mj_step(self.model, self.data)

                    self.publish_msg()
                else:
                    mujoco.mj_forward(self.model, self.data)
                # rclpy.spin_once(self)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            # viewer.sync()
            end_time = start_time + self.dt
            while time.perf_counter() < end_time:
                pass

        self.thread1.join()


def main(args=None):
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."
    rclpy.init(args=args)
    # Load the model and data.
    srobot_desc_path = get_package_share_directory("srobot_description")
    xml_path = os.path.join(srobot_desc_path, "mjcf/srobot_wq.xml")

    sim = MojocoSim(xml_path)

    # sim.simulate()
    rclpy.spin(sim)

    # Destroy the node explicitly
    sim.destroy_node()
    sim.shutdown()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
