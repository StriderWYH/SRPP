# Install script for directory: /home/ur3/catkin_FrankaPanda/src/panda_simulator/panda_sim_custom_action_server

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/ur3/catkin_FrankaPanda/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
        file(MAKE_DIRECTORY "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
      endif()
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin")
        file(WRITE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin" "")
      endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/ur3/catkin_FrankaPanda/install/_setup_util.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/ur3/catkin_FrankaPanda/install" TYPE PROGRAM FILES "/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/catkin_generated/installspace/_setup_util.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/ur3/catkin_FrankaPanda/install/env.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/ur3/catkin_FrankaPanda/install" TYPE PROGRAM FILES "/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/catkin_generated/installspace/env.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/ur3/catkin_FrankaPanda/install/setup.bash;/home/ur3/catkin_FrankaPanda/install/local_setup.bash")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/ur3/catkin_FrankaPanda/install" TYPE FILE FILES
    "/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/catkin_generated/installspace/setup.bash"
    "/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/catkin_generated/installspace/local_setup.bash"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/ur3/catkin_FrankaPanda/install/setup.sh;/home/ur3/catkin_FrankaPanda/install/local_setup.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/ur3/catkin_FrankaPanda/install" TYPE FILE FILES
    "/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/catkin_generated/installspace/setup.sh"
    "/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/catkin_generated/installspace/local_setup.sh"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/ur3/catkin_FrankaPanda/install/setup.zsh;/home/ur3/catkin_FrankaPanda/install/local_setup.zsh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/ur3/catkin_FrankaPanda/install" TYPE FILE FILES
    "/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/catkin_generated/installspace/setup.zsh"
    "/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/catkin_generated/installspace/local_setup.zsh"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/ur3/catkin_FrankaPanda/install/.rosinstall")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/ur3/catkin_FrankaPanda/install" TYPE FILE FILES "/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/catkin_generated/installspace/.rosinstall")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/catkin_generated/safe_execute_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/panda_sim_custom_action_server" TYPE FILE FILES "/home/ur3/catkin_FrankaPanda/devel/.private/panda_sim_custom_action_server/include/panda_sim_custom_action_server/PandaPositionFFJointTrajectoryActionServerConfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python3" -m compileall "/home/ur3/catkin_FrankaPanda/devel/.private/panda_sim_custom_action_server/lib/python3/dist-packages/panda_sim_custom_action_server/cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages/panda_sim_custom_action_server" TYPE DIRECTORY FILES "/home/ur3/catkin_FrankaPanda/devel/.private/panda_sim_custom_action_server/lib/python3/dist-packages/panda_sim_custom_action_server/cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/catkin_generated/installspace/panda_sim_custom_action_server.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/panda_sim_custom_action_server/cmake" TYPE FILE FILES
    "/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/catkin_generated/installspace/panda_sim_custom_action_serverConfig.cmake"
    "/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/catkin_generated/installspace/panda_sim_custom_action_serverConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/panda_sim_custom_action_server" TYPE FILE FILES "/home/ur3/catkin_FrankaPanda/src/panda_simulator/panda_sim_custom_action_server/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/panda_sim_custom_action_server" TYPE PROGRAM FILES "/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/catkin_generated/installspace/start_gripper_action_server.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/panda_sim_custom_action_server" TYPE PROGRAM FILES "/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/catkin_generated/installspace/start_joint_trajectory_server.py")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/gtest/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/ur3/catkin_FrankaPanda/build/panda_sim_custom_action_server/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
