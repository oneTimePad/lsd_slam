cmake_minimum_required(VERSION 2.8.3)
project(depth_pkg)

find_package(catkin  REQUIRED COMPONENTS message_generation std_msgs)

add_message_files(DIRECTORY src/depth_predictor FILES Depth.msg)
catkin_python_setup()
generate_messages(DEPENDENCIES std_msgs)

catkin_package(CATKIN_DEPENDS message_runtime std_msgs)



catkin_install_python(PROGRAMS
	bin/predict
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
	depth_predictor.launch
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}/depth_predictor)
