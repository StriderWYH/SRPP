# generated from catkin/cmake/template/pkg.context.pc.in
CATKIN_PACKAGE_PREFIX = ""
PROJECT_PKG_CONFIG_INCLUDE_DIRS = "${prefix}/include;/opt/ros/noetic/include;/home/ur3/catkin_FrankaPanda/devel/share/orocos_kdl/cmake/../../../include;/usr/include/eigen3".split(';') if "${prefix}/include;/opt/ros/noetic/include;/home/ur3/catkin_FrankaPanda/devel/share/orocos_kdl/cmake/../../../include;/usr/include/eigen3" != "" else []
PROJECT_CATKIN_DEPENDS = "roscpp;std_msgs;gazebo_ros_control;kdl_parser;boost_sml;controller_manager;controller_interface;control_toolbox;pluginlib;hardware_interface;transmission_interface;joint_limits_interface;eigen_conversions;urdf;angles;franka_hw;franka_msgs;franka_gripper;franka_example_controllers".replace(';', ' ')
PKG_CONFIG_LIBRARIES_WITH_PREFIX = "-lfranka_hw_sim;-lfranka_gripper_sim;/opt/ros/noetic/lib/x86_64-linux-gnu/libfranka.so.0.9.2;/home/ur3/catkin_FrankaPanda/devel/lib/liborocos-kdl.so.1.4.0".split(';') if "-lfranka_hw_sim;-lfranka_gripper_sim;/opt/ros/noetic/lib/x86_64-linux-gnu/libfranka.so.0.9.2;/home/ur3/catkin_FrankaPanda/devel/lib/liborocos-kdl.so.1.4.0" != "" else []
PROJECT_NAME = "franka_gazebo"
PROJECT_SPACE_DIR = "/home/ur3/catkin_FrankaPanda/install"
PROJECT_VERSION = "0.10.1"
