
#include <iostream>

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>

#include <MonoCamera/object.h>

using namespace std;

void callback(const MonoCamera::object::ConstPtr& msg)
{
    cout<<"目标数字:"<<msg->object_number<<endl;
    cout<<"目标位置:("<<msg->object_position.x<<","<<msg->object_position.y<<","<<msg->object_position.z<<")"<<endl;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "listener");
    ros::NodeHandle nh;
    ros::Subscriber sub;
    sub = nh.subscribe<MonoCamera::object>("object_pub",10,callback);
    ros::spin();
    return 0;
}
