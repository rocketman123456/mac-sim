#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import time
import threading

import rospy
import rospkg

from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Pose, Twist
from srobot_msg.msg import motor_cmds, motor_states, motor_cmd, motor_state

# Simulation timestep in seconds.
dt: float = 0.002
pause_flag: bool = True
joint_state = JointState()
joint_state.name = [
    "LeftHipYaw_Joint",
    "LeftHipRoll_Joint",
    "LeftHipPitch_Joint",
    "LeftKnee_Joint",
    "RightHipYaw_Joint",
    "RightHipRoll_Joint",
    "RightHipPitch_Joint",
    "RightKnee_Joint",
]


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."
    rospy.init_node("srobot_sim", anonymous=True)
    # Load the model and data.
    rospack = rospkg.RosPack()
    rospack.list()
    hector_desc_path = rospack.get_path("srobot_description")
    xml_path = hector_desc_path + "/mjcf/srobot_p.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    data.qpos[-8:] = np.array([0.0, 0.0, -0.5, 1.0, 0.0, 0.0, -0.5, 1.0])
    global dt
    model.opt.timestep = dt

    key_id = model.key("home").id

    joint_pos_noise_range = 0.02
    joint_vel_noise_range = 0.15
    quat_noise_range = 0.1
    gyro_noise_range = 0.1
    acc_noise_range = 0.2
    apply_noise = False

    def normalize_quaternion(quat):
        norm = np.linalg.norm(quat)
        return quat / norm

    def publishMsg():
        # * Publish joint positions and velocities
        # get last 8 element of qpos and qvel
        qp = data.qpos[-8:].copy()
        qv = data.qvel[-8:].copy()

        if apply_noise:
            qp += (2.0 * np.random.normal(0.0, 1.0, 8) - 1.0) * joint_pos_noise_range
            qv += (2.0 * np.random.normal(0.0, 1.0, 8) - 1.0) * joint_vel_noise_range

        motor_state_msg = motor_states()
        for i in range(8):
            obj = motor_state()
            obj.tau = 0.0
            obj.q = qp[i]
            obj.dq = qv[i]
            motor_state_msg.states.append(obj)
        pubJoints.publish(motor_state_msg)

        joint_state.position = qp.copy()
        joint_state.velocity = qv.copy()
        joint_state.header.stamp = rospy.get_rostime()
        pubRviz.publish(joint_state)

        # Publish body imu
        imu_msg = Imu()
        imu_msg.header.stamp = rospy.get_rostime()
        quat = data.sensor("BodyQuat").data.copy()
        # print(quat)
        if apply_noise:
            quat += (2.0 * np.random.normal(0.0, 1.0, 4) - 1.0) * quat_noise_range
            quat = normalize_quaternion(quat)
        imu_msg.orientation.w = quat[0]
        imu_msg.orientation.x = quat[1]
        imu_msg.orientation.y = quat[2]
        imu_msg.orientation.z = quat[3]
        # print(imu_msg.orientation)
        gyro = data.sensor("BodyGyro").data.copy()
        if apply_noise:
            gyro += (2.0 * np.random.normal(0.0, 1.0, 3) - 1.0) * gyro_noise_range
        imu_msg.angular_velocity.x = gyro[0]
        imu_msg.angular_velocity.y = gyro[1]
        imu_msg.angular_velocity.z = gyro[2]
        acc = data.sensor("BodyAcc").data.copy()
        if apply_noise:
            acc += (2.0 * np.random.normal(0.0, 1.0, 3) - 1.0) * acc_noise_range
        imu_msg.linear_acceleration.x = acc[0]
        imu_msg.linear_acceleration.y = acc[1]
        imu_msg.linear_acceleration.z = acc[2]
        pubImu.publish(imu_msg)

    def calculateControl():
        qp = data.qpos[-8:].copy()
        qv = data.qvel[-8:].copy()
        for i in range(8):
            # tau_p[i] = kp[i] * (q[i] - qp[i])
            # tau_d[i] = kd[i] * (dq[i] - qv[i])
            # ctrl[i] = tau[i] + tau_p[i] + tau_d[i]
            ctrl[i] = q[i]
        data.ctrl[:] = ctrl

    def key_callback(keycode):
        viewer.lock()
        if chr(keycode) == " ":
            global pause_flag
            pause_flag = not pause_flag
            simState = Bool()
            simState.data = pause_flag
            pubSimState.publish(simState)

    def controlCallback(data_msg):
        # data.ctrl[:] = d
        qp = data.qpos[-8:].copy()
        qv = data.qvel[-8:].copy()

        # tau = [0] * 8
        # tau_p = [0] * 8
        # tau_d = [0] * 8
        # kp = [0] * 8
        # kd = [0] * 8
        # q = [0] * 8
        # dq = [0] * 8
        # ctrl = [0] * 8
        for i in range(8):
            # kp[i] = data_msg.cmds[i].kp
            # kd[i] = data_msg.cmds[i].kd
            # q[i] = data_msg.cmds[i].q
            # dq[i] = data_msg.cmds[i].dq
            # tau[i] = data_msg.cmds[i].tau
            # tau_p[i] = kp[i] * (q[i] - qp[i])
            # tau_d[i] = kd[i] * (dq[i] - qv[i])
            # ctrl[i] = tau[i] + tau_p[i] + tau_d[i]
            ctrl[i] = data_msg.cmds[i].q
        # print(kp)
        # print(kd)
        data.ctrl[:] = ctrl

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=True, show_right_ui=True, key_callback=key_callback
    ) as viewer:
        # Reset the simulation to the initial keyframe.
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        # Initialize the camera view to track the base link.
        mujoco.mjv_defaultCamera(viewer.cam)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("Base_Link").id

        totalMass = sum(model.body_mass)
        print("total mass: ", totalMass)
        # * Set subscriber and publisher
        pubJoints = rospy.Publisher("/motor_states", motor_states, queue_size=10)
        pubRviz = rospy.Publisher("/joint_states", JointState, queue_size=10)
        pubImu = rospy.Publisher("/srobot_imu", Imu, queue_size=10)
        pubSimState = rospy.Publisher("/pause_flag", Bool, queue_size=10)
        rospy.Subscriber("/motor_cmds", motor_cmds, controlCallback)

        tau = [0] * 8
        tau_p = [0] * 8
        tau_d = [0] * 8
        kp = [0] * 8
        kd = [0] * 8
        q = [0] * 8
        dq = [0] * 8
        ctrl = [0] * 8

        mujoco.mj_forward(model, data)
        viewer.sync()

        def sync_loop():
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.010)

        thread1 = threading.Thread(target=sync_loop)
        thread1.start()

        while viewer.is_running():

            start_time = time.perf_counter()
            with viewer.lock():
                if not pause_flag:
                    mujoco.mj_step(model, data)
                    # calculateControl()
                    publishMsg()
                else:
                    mujoco.mj_forward(model, data)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            # viewer.sync()
            end_time = start_time + dt
            while time.perf_counter() < end_time:
                pass

        thread1.join()


if __name__ == "__main__":
    main()
