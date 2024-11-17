#!/usr/bin/env python3
import os
import mujoco

import rclpy
from ament_index_python.packages import get_package_share_directory

from srobot_sim.mujoco_sim import MojocoSim


def main(args=None):
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."
    rclpy.init(args=args)
    # Load the model and data.
    srobot_desc_path = get_package_share_directory("srobot_description")
    xml_path = os.path.join(srobot_desc_path, "mjcf/srobot_mpc.xml")

    sim = MojocoSim(xml_path, "torque")

    # sim.simulate()
    rclpy.spin(sim)

    # Destroy the node explicitly
    sim.destroy_node()
    sim.shutdown()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
