cmake_minimum_required(VERSION 3.0.2)
project(MonoCamera)

#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

#eigen
find_package(Eigen3 REQUIRED)
#include_directories("/usr/include/eigen3")

find_package(catkin REQUIRED COMPONENTS 
    roscpp
    tf
    std_msgs
    message_generation
    #sensor_msgs
    #mavros_msgs
    geometry_msgs
)

add_message_files(
   FILES
   object.msg
 )

 generate_messages(
   DEPENDENCIES
   geometry_msgs
   std_msgs
 )

catkin_package(
    #INCLUDE_DIRS include
    CATKIN_DEPENDS 
    message_runtime
)

include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(MonoCamera src/MonoCamera.cpp)
add_dependencies(MonoCamera MonoCamera_generate_messages_cpp)
target_link_libraries(MonoCamera ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(listener src/listener.cpp)
add_dependencies(listener MonoCamera_generate_messages_cpp)
target_link_libraries(listener ${catkin_LIBRARIES} )
