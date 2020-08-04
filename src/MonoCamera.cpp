#include <cmath>
#include <iostream>
#include <ros/ros.h>
#include <exception> 
#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>
#include <MonoCamera/object.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <geometry_msgs/PoseStamped.h>

//Jacobi
#define J_N 4
//#define Epsilon 35
#define Max 15
const int N2 = 2 * J_N;
double x0[J_N] = {300,300,300,300};
//牛顿迭代的初值(深度Z的初值)
double x_start[J_N] = {300,300,300,300};

const int max_value_H1 = 180/2-1;
const int max_value_H2 = 360/2;//因为是以8位存储，所以除以2
const int max_value = 255;//H：0~180, S和V：0-255

const cv::String window_capture_name = "Video Capture";
const cv::String window_detection_name = "Object Detection";

int low_H1 = 0, low_H2 = max_value_H1+1, low_S = 0, low_V = 0;
int high_H1= max_value_H1, high_H2= max_value_H2, high_S= max_value, high_V = max_value;

//匹配数字
 int serise_num = 1;

//相机坐标
Eigen::Vector3d Pc1,Pc2,Pc3, Pc4, Pc_central;

//旋转矩阵
Eigen::Matrix3d R_Euler;
//平移矩阵
Eigen::Vector3d t_Euler;
//顶点及中心点的世界坐标
Eigen::Vector3d P_world1, P_world2, P_world3, P_world4, P_worldcentral;

/////////////////////////////滑动条///////////////////////////
static void on_low_H1_thresh_trackbar(int, void *)
{
    low_H1 = std::min(high_H1-1, low_H1);//防止最大值小于最小值
	cv::setTrackbarPos("Low H1", window_detection_name, low_H1);
}
static void on_high_H1_thresh_trackbar(int, void *)
{
    high_H1 = std::max(high_H1, low_H1+1);
	setTrackbarPos("High H1", window_detection_name, high_H1);
}

static void on_low_H2_thresh_trackbar(int, void *)
{
    low_H2 = std::min(high_H2-1, low_H2);
	cv::setTrackbarPos("Low H2", window_detection_name, low_H2);
}
static void on_high_H2_thresh_trackbar(int, void *)
{
    high_H2 = std::max(high_H2, low_H2+1);
	setTrackbarPos("High H2", window_detection_name, high_H2);
}

static void on_low_S_thresh_trackbar(int, void *)
{
    low_S = std::min(high_S-1, low_S);
	setTrackbarPos("Low S",window_detection_name,low_S);
}
static void on_high_S_thresh_trackbar(int, void *)
{
    high_S = std::max(high_S,low_S+1);
	setTrackbarPos("High S",window_detection_name,high_S);
}

static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = std::min(high_V-1, low_V);
	setTrackbarPos("Low V",window_detection_name,low_V);
}
static void on_high_V_thresh_trackbar(int, void *)
{
    high_V = std::max(high_V,low_V+1);
	setTrackbarPos("High V",window_detection_name,high_V);
}

////////////////////////两幅图像相减////////////////////////
cv::Mat substract(cv::Mat &src, cv::Mat&dst){
	int sum = 0; 
    cv::Mat result = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
	for (int i = 0; i < src.rows; ++i){
		uchar *data1 = src.ptr<uchar>(i);
		uchar*data2 = dst.ptr<uchar>(i);
		uchar*data3 = result.ptr<uchar>(i);
		for (int j = 0; j < src.cols; ++j)
			data3[j] = abs(data1[j] - data2[j]);
	}
	return result;
}

//////////////////////////获取数字///////////////////////////
int get_One(cv::Mat image){
	int sum_one = 0;
	for (int i = 0; i < image.rows; ++i){
		uchar*data = image.ptr<uchar>(i);
		for (int j = 0; j < image.cols; ++j){
			if (data[j] > 0) 
                sum_one += 1; 
		}
	}
    return sum_one;
}

///////////////////////////模板匹配//////////////////////////
int temple_image(cv::Mat img)
{
	cv::Mat src_image; 
    int temp = 0; 
    float sum = 0; 
    float max = 200000000; 
    int seriesNum = 0;
	float c = 0.0; 
    float n1 = 0.0; 
    float n2 = 0.0; 
    cv::Mat result;
	for (int i = 0; i < 4; ++i){
		char name[520];
		std::sprintf(name, "/home/lxl/catkin_ws/src/MonoCamera/template/%d.jpg", i);
		src_image = cv::imread(name);
		if (!src_image.data) 
            std::cout << "failed to read" << std::endl; 
		result = substract(img, src_image);
		c = get_One(result);
		if (c < max) {
			max = c; 
            seriesNum = i;
		}
	}
	return seriesNum;
}

//////////////////////////订阅+发布/////////////////////////
/*
    (一)订阅飞机位姿:
    (1)话题:  /mavros/local_position/pose
    (2)消息类型:  geometry_msgs/PoseStamped.msg
            位姿: geometry_msgs/Pose pose
                    位置: geometry_msgs/Point position
                                    float64 x
                                    float64 y
                                    float64 z
                    姿态: geometry_msgs/Quaternion orientation
                                    float64 x
                                    float64 y
                                    float64 z
                                    float64 w
    (二)发布目标信息:
    (1)话题:  /object_pub
    (2) 消息类型: MonoCamera/object
            数字: int16 object_number
            位置: geometry_msgs/Point object_position
*/
class SubAndPub
{
    public:
        SubAndPub();//构造函数
        void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);//订阅回调函数
    private:
        ros::NodeHandle nh_;  //句柄
        ros::Publisher pub_;     //发布者
        ros::Subscriber sub_;   //订阅者
};
//构造函数
SubAndPub::SubAndPub()
{
    //指定发布者往object_pub这个话题上发布MonoCamera::object类型的数据
    pub_ = nh_.advertise<MonoCamera::object>("object_pub", 1);
    //指定订阅者从/mavros/local_position/pose这个话题上订阅geometry_msgs::PoseStamped类型的数据
    sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("mavros/local_position/pose", 10, &SubAndPub::poseCallback, this);//话题前面不加/表示相对命名空间,加了表示全局命名空间
}
//订阅无人机位姿信息的回调函数,利用这些信息求出目标世界坐标,将目标信息发布出去
void SubAndPub::poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    //初始化四元数,初始化顺序为w,x,y,z(注意第一项为实部)
    Eigen::Quaterniond Q = Eigen::Quaterniond(msg->pose.orientation.w, 
                                                                                               msg->pose.orientation.x, 
                                                                                               msg->pose.orientation.y, 
                                                                                               msg->pose.orientation.z);

    //通过coeffs()函数按照x,y,z,w顺序打印成员,可以通过vec()打印虚部x,y,z
    std::cout << "四元数 x y z w:\n " << Q.coeffs() << std::endl;

    //四元数归一化(Q要进行归一化才能进行计算)
    Eigen::Quaterniond Q_normalized = Q.normalized();
    std::cout << "归一化后的四元数 x y z w:\n " << Q_normalized.coeffs() << std::endl;

    //旋转矩阵
    R_Euler  = Q_normalized.matrix();
    std::cout <<"旋转矩阵:\n" << R_Euler << std::endl;

    //无人机在世界坐标系下的平移矩阵t_Euler((需要统一单位为cm,无人机坐标单位是m,所以这里要乘以100))
    t_Euler << msg->pose.position.x*100, 
                          msg->pose.position.y*100, 
                          msg->pose.position.z*100;
    std::cout << "平移矩阵:\n" << t_Euler << std::endl;

    //求中心点的世界坐标 Pw = R-1*Pc+t
    P_worldcentral =R_Euler.inverse()* (Pc_central) + t_Euler;

    //初始化MonoCamera::object类型的消息
    MonoCamera::object obj_output;
    obj_output.object_number = serise_num;
    obj_output.object_position.x = P_worldcentral[0];
    obj_output.object_position.y = P_worldcentral[1];
    obj_output.object_position.z = P_worldcentral[2];
    //发布者发布消息
    pub_.publish(obj_output);
}

/********************************************************************主函数**************************************************************************/
int main(int argc, char **argv)
{
    ros::init(argc,argv,"sub_and_pub");
    SubAndPub SAPObject;

    //原始,灰度,hsv,颜色分割图像
    cv::Mat image_raw, image_gray, image_hsv, image_threshold1, image_threshold2, image_threshold;

    //启动编号为0的相机
    cv::VideoCapture cap(0);

    while(ros::ok())
    {
        if(!cap.isOpened())
        {
            std::cout << "Cannot open the cam!" << std::endl;
            return -1;
        }

        //HSV阈值调节窗口
        namedWindow(window_capture_name);
        namedWindow(window_detection_name);

        createTrackbar("Low H1", window_detection_name, &low_H1, max_value_H1, on_low_H1_thresh_trackbar);
        createTrackbar("High H1", window_detection_name, &high_H1, max_value_H1, on_high_H1_thresh_trackbar);

        createTrackbar("Low H2", window_detection_name, &low_H2, max_value_H2, on_low_H2_thresh_trackbar);
        createTrackbar("High H2", window_detection_name, &high_H2, max_value_H2, on_high_H2_thresh_trackbar);

        createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
        createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);

        createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
        createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);

        try{
            while(1){
                //从相机读取视频流
                cap >> image_raw;

                //图像为空就退出程序
                if(image_raw.empty())
                    break;

                //显示原始图像
                cv::imshow(window_capture_name, image_raw);

                //颜色空间转换
                cv::cvtColor(image_raw, image_hsv, cv::COLOR_BGR2HSV);  

                //进行颜色分割，输出图像为CV_8UC1
                cv::inRange(image_hsv, cv::Scalar(low_H1,low_S,low_V), cv::Scalar(high_H1,high_S,high_V), image_threshold1);
                cv::inRange(image_hsv, cv::Scalar(low_H2,low_S,low_V), cv::Scalar(high_H2,high_S,high_V), image_threshold2);
                cv::bitwise_or(image_threshold1, image_threshold2, image_threshold);

                //形态学运算，腐蚀膨胀
                cv::Mat image_erode, image_dilate;
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(4,4));
                cv::dilate(image_threshold, image_dilate, kernel);
                //cv::imshow("image_dilate", image_dilate);
                cv::erode(image_dilate, image_erode, kernel);
                cv::imshow("image_erode", image_erode);
        
                //寻找轮廓
                std::vector<std::vector<cv::Point> > contours;
                std::vector<cv::Vec4i> hierarchy;
                cv::findContours(image_erode, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

                //画轮廓
                cv::Mat image_contours(image_raw.size(), CV_8UC3, cv::Scalar(0,0,0));

                //检测不到轮廓的话就退出本次循环
                if(contours.empty())
                    continue;

                //取外轮廓squares, 其中有多个轮廓，每个轮廓含有很多个点
                std::vector<std::vector<cv::Point> > squares;
                std::list<int> len_list;

                //计算每个轮廓的面积
                for (int i = 0; i < contours.size(); i++)
                {
                    len_list.push_back(cv::contourArea(contours[i]));
                }

                //寻找面积最大的轮廓
                if (len_list.size() > 0)
                {
                    squares.push_back(contours[std::distance(len_list.begin(), max_element(len_list.begin(), len_list.end()))]);
                }

                //最小外接矩形集合box
                std::vector<cv::RotatedRect> box(squares.size());
                std::vector<cv::Rect> srcRect(squares.size());

                for(int i = 0; i < squares.size(); i++)
                {
                    srcRect[i] = cv::boundingRect(squares[i]);
                    //计算轮廓i的最小外接矩形
                    box[i] = cv::minAreaRect(cv::Mat(squares[i]));
                    //绘制最小外接矩形的中心点
                    cv::circle(image_raw, cv::Point(box[i].center.x, box[i].center.y), 1, cv::Scalar(255, 0, 0), -1, 8); 
                }

                cv::Point2f rect1[4];//外接正方形的四个顶点

                rect1[0].x = box[0].center.x - srcRect[0].height/2;
                rect1[0].y = box[0].center.y - srcRect[0].height/2;//左上角顶点

                rect1[1].x = box[0].center.x + srcRect[0].height/2;
                rect1[1].y = box[0].center.y -  srcRect[0].height/2;//右上角顶点

                rect1[2].x = box[0].center.x + srcRect[0].height/2;
                rect1[2].y = box[0].center.y + srcRect[0].height/2;//右下角顶点

                rect1[3].x = box[0].center.x - srcRect[0].height/2;
                rect1[3].y = box[0].center.y + srcRect[0].height/2;//左下角顶点

                cv::circle(image_raw, rect1[0], 4, cv::Scalar(0, 0, 255), -1, 8); //绘制最小外接正方形的左上角顶点
                cv::circle(image_raw, rect1[1], 4, cv::Scalar(0, 0, 255), -1, 8); //绘制最小外接正方形的右上角顶点
                cv::circle(image_raw, rect1[2], 4, cv::Scalar(0, 0, 255), -1, 8); //绘制最小外接正方形的右下角顶点
                cv::circle(image_raw, rect1[3], 4, cv::Scalar(0, 0, 255), -1, 8); //绘制最小外接正方形的左下角顶点

                //绘制最小外接正方形的每条边
                for(int i = 0; i < 4; i++)
                    line(image_raw, rect1[i], rect1[(i+1)%4], cv::Scalar(255, 0, 0), 2, 8);

                //抠取出数字区域的二值图像
                cv::Mat roi;
                roi = image_erode(cv::Rect(srcRect[0].x, srcRect[0].y, srcRect[0].width, srcRect[0].height));
                cv::imshow("resizeRoi", roi);
                cv::resize(roi, roi, cv::Size(50, 50));
                cv::imwrite("/home/lxl/catkin_ws/src/MonoCamera/test.jpg", roi);

                roi = cv::imread("/home/lxl/catkin_ws/src/MonoCamera/roi.jpg");
                //模板匹配识别数字
                serise_num = temple_image(roi);
                std::cout << "serise_num:  " << serise_num << std::endl;

                ////////////////////////求解三点空间坐标//////////////////////////
                //四个顶点的像素坐标
                /*
                        1——2
                         |        | 
                        3——4
                */
                Eigen::Vector3d min_P1,  min_P2,  min_P3 ,  min_P4, min_Point_central;
                min_P1 << rect1[0].x, rect1[0].y, 1;//P1（左上）
                min_P2 << rect1[1].x, rect1[1].y, 1;//P2（右上）
                min_P3 << rect1[3].x, rect1[3].y, 1;//P3（左下）
                min_P4 << rect1[2].x, rect1[2].y, 1;//P4（右下）
                min_Point_central << box[0].center.x, box[0].center.y, 1;//中心点

                // std::cout << "中心点像素坐标：" << min_Point_central[0] << "  " << min_Point_central[1] << std::endl;
                std::string name = cv::format("%d (%f, %f)", serise_num, min_P1[0], min_P1[1]);
                cv::putText(image_raw, name, cv::Point(box[0].center.x, box[0].center.y), 1, 1, cv::Scalar(255,0,0));
                cv::imshow("image_detetct", image_raw);

                //相机的内参矩阵(3x3)
                Eigen::Matrix3d CameraM;
                Eigen::Vector4d CameraPram;
                CameraPram[0] = 572.7926959320354;//fx
                CameraPram[1] = 572.0023335829402;//fy
                CameraPram[2] = 316.3742545740499;//cx
                CameraPram[3] = 207.8033169148048;//cy
                CameraM << CameraPram[0], 0, CameraPram[2], 0, CameraPram[1], CameraPram[3], 0, 0, 1;

                double w_camera = 1.17492;

                //相机归一化坐标（Xc/Zc, Yc/Zc)T =  相机内参矩阵的逆 *(u, v)T
                Eigen::Vector3d trans_P1d, trans_P2d, trans_P3d, trans_P4d;
                trans_P1d = CameraM.inverse() * min_P1;
                trans_P2d = CameraM.inverse() * min_P2;
                trans_P3d = CameraM.inverse() * min_P3;
                trans_P4d = CameraM.inverse() * min_P4;

                //3.13鱼眼摄像头模型
                double Rd1 = sqrt(trans_P1d[0]*trans_P1d[0] + trans_P1d[1] * trans_P1d[1]);
                double Rd2 = sqrt(trans_P2d[0]*trans_P2d[0] + trans_P2d[1] * trans_P2d[1]);
                double Rd3 = sqrt(trans_P3d[0]*trans_P3d[0] + trans_P3d[1] * trans_P3d[1]);
                double Rd4 = sqrt(trans_P4d[0]*trans_P4d[0] + trans_P4d[1] * trans_P4d[1]);
                
                double r1_camera=tan(Rd1 * w_camera) / (2 * tan(w_camera/2));
                double r2_camera=tan(Rd2 * w_camera) / (2 * tan(w_camera/2));
                double r3_camera=tan(Rd3 * w_camera) / (2 * tan(w_camera/2));
                double r4_camera=tan(Rd4 * w_camera) / (2 * tan(w_camera/2));
                
                Eigen::Vector3d trans_P1, trans_P2, trans_P3, trans_P4;
                trans_P1[0] = trans_P1d[0] * r1_camera / Rd1;
                trans_P1[1] = trans_P1d[1] * r1_camera / Rd1;
                trans_P2[0] = trans_P2d[0] * r2_camera / Rd2;
                trans_P2[1] = trans_P2d[1] * r2_camera / Rd2;
                trans_P3[0] = trans_P3d[0] * r3_camera / Rd3;
                trans_P3[1] = trans_P3d[1] * r3_camera / Rd3;
                trans_P4[0] = trans_P4d[0] * r4_camera / Rd4;
                trans_P4[1] = trans_P4d[1] * r4_camera / Rd4;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                float Epsilon =0.01;	
                Eigen::MatrixXd X1_Jac(1,4); 

                X1_Jac <<10,10,10,10;

                Eigen::MatrixXd Y0_Jac(1,7);

                Eigen::MatrixXd X0_Jac(1,4);
                double x_start[4];
                X0_Jac << x_start[0], x_start[1], x_start[2], x_start[3];
                Eigen::MatrixXd W_Jac(1,4);
                W_Jac<<100,100,100,100;
               
                Eigen::MatrixXd YY_Jac(1,4);
                Eigen::MatrixXd Y_temp_Jac(4,7);
                Eigen::MatrixXd jacobian_temp(4,7);
                Eigen::MatrixXd jacobian_temp_M(4,7);
                Eigen::MatrixXd Z_Jac(1,4);
                Eigen::MatrixXd	JAC(4,7);//jacobin矩阵

                double y0[J_N+3]={0,0,0,0,0,0,0}, jacobian[J_N][J_N+3], invjacobian[J_N+3][J_N], x1[J_N]={0,0,0,0}, errornorm, v[12], L=0;
                double D1 = 56, D2 = 56, D3 = 79.2, D4 = 56;//内轮廓实际大小(单位cm)
                int i_inv=0, j_inv=0, k_inv=0, i_jcb=0, iter = 0;

                v[0] = trans_P1[0];//Xc1
                v[1] = trans_P2[0];//Xc2
                v[2] = trans_P3[0];//Xc3
                v[3] = trans_P4[0];//Xc4
                v[4] = trans_P1[1];//Yc1
                v[5] = trans_P2[1];//Yc2
                v[6] = trans_P3[1];//Yc3
                v[7] = trans_P4[1];//Yc4
                v[8] = D1;
                v[9] = D2;
                v[10] = D3;
                v[11] = D4;

                do{
                    iter = iter + 1;
                    //计算雅克比矩阵 jacobian
                    y0[0] = (v[0]* v[0]+ v[4]* v[4]+ 1)*x0[0]* x0[0] - 2 * (v[0] * v[1] +v[4] *v[5] +1)*x0[0] *x0[1] + (v[1] * v[1] + v[5] * v[5] + 1)*x0[1] * x0[1] - v[8] * v[8];
                    y0[1] = (v[0]* v[0]+ v[4]* v[4]+ 1)*x0[0]* x0[0] - 2 * (v[0] * v[2] +v[4] *v[6] +1)*x0[0] *x0[2] + (v[2] * v[2] + v[6] * v[6] + 1)*x0[2] * x0[2] - v[9] * v[9];
                    y0[2] = (v[1] *v[1]+ v[5] *v[5]+ 1)*x0[1]* x0[1] - 2 * (v[1] * v[2] +v[5] *v[6] +1)*x0[1] *x0[2] + (v[2] * v[2] + v[6] * v[6] + 1)*x0[2] *x0[2] - v[10] * v[10];
                    y0[3] = (v[3]* v[3]+ v[7]* v[7]+ 1)*x0[3]* x0[3] - 2 * (v[3] * v[2] +v[7] *v[6] +1)*x0[3] *x0[2] + (v[2] * v[2] + v[6] * v[6] + 1)*x0[2] *x0[2] - v[11] * v[11];
                    y0[4] = (v[0]*x0[0]-v[1]*x0[1])*(v[6]*x0[2]-v[7]*x0[3])-(v[2]*x0[2]-v[3]*x0[3])*(v[4]*x0[0]-v[5]*x0[1]);
                    y0[5] = (v[0]*x0[0]-v[1]*x0[1])*(x0[2]-x0[3])-(v[2]*x0[2]-v[3]*x0[3])*(x0[0]-x0[1]);
                    y0[6] = (v[4]*x0[0]-v[5]*x0[1])*(x0[2]-x0[3])-(v[6]*x0[2]-v[7]*x0[3])*(x0[0]-x0[1]);
                    //计算初值位置的值
                    //计算雅克比矩阵的逆矩阵 invjacobian
                    //jacobian have n*n element  计算函数雅克比的值
                    JAC(0,0)= 2 * (v[0] * v[0] + v[4] * v[4] + 1)*x0[0] - 2 * (v[0] * v[1] + v[4] * v[5] + 1)*x0[1];
                    JAC(0,1) = 2 * (v[0] * v[0] + v[4] * v[4] + 1)*x0[0] - 2 * (v[0] * v[2] + v[4] * v[6] + 1)*x0[2];
                    JAC(0,2)= 0;
                    JAC(0,3) = 0;
                    JAC(0,4) =  (v[0])*(v[6]*x0[2]-v[7]*x0[3])-(v[2]*x0[2]-v[3]*x0[3])*(v[4]); ;
                    JAC(0,5) =  (v[0])*(x0[2]-x0[3])-(v[2]*x0[2]-v[3]*x0[3]);
                    JAC(0,6) =  (v[4])*(x0[2]-x0[3])-(v[6]*x0[2]-v[7]*x0[3]);

                    JAC(1,0) = -2 * (v[0] * v[1] + v[4] * v[5] + 1)*x0[0] + 2 * (v[1] * v[1] + v[5] * v[5] + 1)*x0[1];
                    JAC(1,1) = 0;
                    JAC(1,2) = 2 * (v[1] * v[1] + v[5] * v[5] + 1)*x0[1] - 2 * (v[1] * v[2] + v[5] * v[6] + 1)*x0[2];
                    JAC(1,3) = 0;
                    JAC(1,4) = (-v[1])*(v[6]*x0[2]-v[7]*x0[3])-(v[2]*x0[2]-v[3]*x0[3])*(-v[5]);
                    JAC(1,5) = (-v[1])*(x0[2]-x0[3])-(v[2]*x0[2]-v[3]*x0[3])*(-1);
                    JAC(1,6) = (-v[5])*(x0[2]-x0[3])-(v[6]*x0[2]-v[7]*x0[3])*(-1);

                    JAC(2,0) = 0;
                    JAC(2,1) = -2 * (v[0] * v[2] + v[4] * v[6] + 1)*x0[0] + 2 * (v[2] * v[2] + v[6] * v[6] + 1)*x0[2];
                    JAC(2,2) = -2 * (v[1] * v[2] + v[5] * v[6] + 1)*x0[1] + 2 * (v[2] * v[2] + v[6] * v[6] + 1)*x0[2];
                    JAC(2,3) = 2 * (v[2] * v[2] + v[6] * v[6] + 1)*x0[2] - 2 * (v[3] * v[2] + v[7] * v[6] + 1)*x0[3];
                    JAC(2,4) = (v[0]*x0[0]-v[1]*x0[1])*(v[6])-(v[2])*(v[4]*x0[0]-v[5]*x0[1]);
                    JAC(2,5) = (v[0]*x0[0]-v[1]*x0[1])*(1)-(v[2]*(x0[0]-x0[1]));
                    JAC(2,6) = (v[4]*x0[0]-v[5]*x0[1])*(1)-(v[6]*(x0[0]-x0[1]));

                    JAC(3,0) = 0;
                    JAC(3,1) = 0;
                    JAC(3,2) = 0;
                    JAC(3,3) = 2 * (v[3] * v[3] + v[7] * v[7] + 1) * x0[3] -2 * (v[3] * v[2] + v[7] * v[6] + 1) *x0[2];
                    JAC(3,4) = (v[0]*x0[0]-v[1]*x0[1])*(-v[7])-(-v[3])*(v[4]*x0[0]-v[5]*x0[1]);
                    JAC(3,5) = (v[0]*x0[0]-v[1]*x0[1])*(-1)-(-v[3])*(x0[0]-x0[1]);
                    JAC(3,6) = (v[4]*x0[0]-v[5]*x0[1])*(-1)-(-v[7])*(x0[0]-x0[1]);

                    Eigen::JacobiSVD<Eigen::MatrixXd> svd(JAC, Eigen::ComputeFullU | Eigen::ComputeFullV);//M=USV*
                    double  pinvtoler = 1.e-8; //tolerance
                    int row_invJav = JAC.rows();
                    int col_invJav= JAC.cols();
                    int k_invJav =row_invJav>=col_invJav? col_invJav:row_invJav;
                    Eigen::MatrixXd X_invJav = Eigen::MatrixXd::Zero(col_invJav,row_invJav);
                    Eigen::MatrixXd singularValues_inv_invJav = svd.singularValues();//奇异值
                    Eigen::MatrixXd singularValues_inv_mat_invJav = Eigen::MatrixXd::Zero(col_invJav, row_invJav);
                    for (long i = 0; i<k_invJav; ++i) 
                    {
                        if (singularValues_inv_invJav(i) > pinvtoler)
                            singularValues_inv_invJav(i) = 1.0 / singularValues_inv_invJav(i);
                        else 
                            singularValues_inv_invJav(i) = 0;
                    }
                    for (long i = 0; i < k_invJav; ++i) 
                    {
                        singularValues_inv_mat_invJav(i, i) = singularValues_inv_invJav(i);
                    }
                    X_invJav=(svd.matrixV())*(singularValues_inv_mat_invJav)*(svd.matrixU().transpose());//X=VS+U*

                    //优化版迭代部分
                    Y0_Jac<<y0[0],y0[1],y0[2],y0[3],y0[4],y0[5],y0[6];
                
                    X1_Jac = X0_Jac-Y0_Jac*X_invJav;//x1=x0-f/df

                    jacobian_temp(0,0) = 2 * (v[0] * v[0] + v[4] * v[4] + 1)*X1_Jac(0,0) - 2 * (v[0] * v[1] + v[4] * v[5] + 1)*X1_Jac(0,1);
                    jacobian_temp(0,1)  = 2 * (v[0] * v[0] + v[4] * v[4] + 1)*X1_Jac(0,0) - 2 * (v[0] * v[2] + v[4] * v[6] + 1)*X1_Jac(0,2);
                    jacobian_temp(0,2)  = 0;
                    jacobian_temp(0,3)  = 0;
                    jacobian_temp(0,4)  =  (v[0])*(v[6]*X1_Jac(0,2)-v[7]*X1_Jac(0,3))-(v[2]*X1_Jac(0,2)-v[3]*X1_Jac(0,3))*(v[4]); ;
                    jacobian_temp(0,5)  =  (v[0])*(X1_Jac(0,2)-X1_Jac(0,3))-(v[2]*X1_Jac(0,2)-v[3]*X1_Jac(0,3));
                    jacobian_temp(0,6)  =  (v[4])*(X1_Jac(0,2)-X1_Jac(0,3))-(v[6]*X1_Jac(0,2)-v[7]*X1_Jac(0,3));

                    jacobian_temp(1,0)  = -2 * (v[0] * v[1] + v[4] * v[5] + 1)*X1_Jac(0,0) + 2 * (v[1] * v[1] + v[5] * v[5] + 1)*X1_Jac(0,1);
                    jacobian_temp(1,1) = 0;
                    jacobian_temp(1,2) = 2 * (v[1] * v[1] + v[5] * v[5] + 1)*X1_Jac(0,1) - 2 * (v[1] * v[2] + v[5] * v[6] + 1)*X1_Jac(0,2);
                    jacobian_temp(1,3) = 0;
                    jacobian_temp(1,4) = (-v[1])*(v[6]*X1_Jac(0,2)-v[7]*X1_Jac(0,3))-(v[2]*X1_Jac(0,2)-v[3]*X1_Jac(0,3))*(-v[5]);
                    jacobian_temp(1,5) = (-v[1])*(X1_Jac(0,2)-X1_Jac(0,3))-(v[2]*X1_Jac(0,2)-v[3]*X1_Jac(0,3))*(-1);
                    jacobian_temp(1,6) = (-v[5])*(X1_Jac(0,2)-X1_Jac(0,3))-(v[6]*X1_Jac(0,2)-v[7]*X1_Jac(0,3))*(-1);

                    jacobian_temp(2,0) = 0;
                    jacobian_temp(2,1) = -2 * (v[0] * v[2] + v[4] * v[6] + 1)*X1_Jac(0,0) + 2 * (v[2] * v[2] + v[6] * v[6] + 1)*X1_Jac(0,2);

                    jacobian_temp(2,2) = -2 * (v[1] * v[2] + v[5] * v[6] + 1)*X1_Jac(0,1) + 2 * (v[2] * v[2] + v[6] * v[6] + 1)*X1_Jac(0,2);
                    jacobian_temp(2,3)  = 2 * (v[2] * v[2] + v[6] * v[6] + 1)*X1_Jac(0,2) - 2 * (v[3] * v[2] + v[7] * v[6] + 1) *X1_Jac(0,3);
                    jacobian_temp(2,4) = (v[0]*X1_Jac(0,0)-v[1]*X1_Jac(0,1))*(v[6])-(v[2])*(v[4]*X1_Jac(0,0)-v[5]*X1_Jac(0,1));
                    jacobian_temp(2,5) = (v[0]*X1_Jac(0,0)-v[1]*X1_Jac(0,1))-(v[2])*(X1_Jac(0,0)-X1_Jac(0,1));
                    jacobian_temp(2,6) = (v[4]*X1_Jac(0,0)-v[5]*X1_Jac(0,1))*(1)-(v[6])*(X1_Jac(0,0)-X1_Jac(0,1));

                    jacobian_temp(3,0)  = 0;
                    jacobian_temp(3,1)  = 0;
                    jacobian_temp(3,2) = 0;
                    jacobian_temp(3,3)  = 2 * (v[3] * v[3] + v[7] * v[7] + 1) * X1_Jac(0,3) -2 * (v[3] * v[2] + v[7] * v[6] + 1) *X1_Jac(0,2);
                    jacobian_temp(3,4)  = (v[0]*X1_Jac(0,0)-v[1]*X1_Jac(0,1))*(-v[7])-(-v[3])*(v[4]*X1_Jac(0,1)-v[5]*X1_Jac(0,1));
                    jacobian_temp(3,5)  = (v[0]*X1_Jac(0,0)-v[1]*X1_Jac(0,1))*(-1)-(-v[3])*(X1_Jac(0,0)-X1_Jac(0,1));
                    jacobian_temp(3,6)  = (v[4]*X1_Jac(0,0)-v[5]*X1_Jac(0,1))*(-1)-(-v[7])*(X1_Jac(0,0)-X1_Jac(0,1));
		
                    for(int i=0;i<4;i++)
                    {
                        for(int j=0;j<7;j++)
                        jacobian_temp_M(i,j)=jacobian_temp(i,j)+JAC(i,j);
                    }
			
                    Eigen::JacobiSVD<Eigen::MatrixXd> svd_1(jacobian_temp_M, Eigen::ComputeFullU | Eigen::ComputeFullV);//M=USV*
                    int row_invJav_M = jacobian_temp_M.rows();
                    int col_invJav_M= jacobian_temp_M.cols();
                    int k_invJav_M =row_invJav_M>=col_invJav_M? col_invJav_M:row_invJav_M;
                    Eigen::MatrixXd X_invJav_M = Eigen::MatrixXd::Zero(col_invJav_M,row_invJav_M);
                    Eigen::MatrixXd singularValues_inv_invJav_M = svd_1.singularValues();//奇异值
                    Eigen::MatrixXd singularValues_inv_mat_invJav_M = Eigen::MatrixXd::Zero(col_invJav_M, row_invJav_M);
                    for (long i = 0; i<k_invJav_M; ++i)
                    {
                        if (singularValues_inv_invJav_M(i) > pinvtoler)
                            singularValues_inv_invJav_M(i) = 1.0 / singularValues_inv_invJav_M(i);
                        else 
                            singularValues_inv_invJav_M(i) = 0;
                    }
                    for (long i = 0; i < k_invJav_M; ++i) 
                    {
                        singularValues_inv_mat_invJav_M(i, i) = singularValues_inv_invJav_M(i);
                    }
                    X_invJav_M=(svd_1.matrixV())*(singularValues_inv_mat_invJav_M)*(svd_1.matrixU().transpose());//X=VS+U* 4*7矩阵

                    W_Jac = X0_Jac -2*Y0_Jac*X_invJav_M;//优化之后的迭代

                    double x0_temp[4];
                    for(int i=0;i<4;i++)
                        x0_temp[i]=x0[i];

                    for(int i=0;i<4;i++)
                        x0[i]=W_Jac(0,i);

                    Y0_Jac(0,0) = (v[0]* v[0]+ v[4]* v[4]+ 1)*x0[0]* x0[0] - 2 * (v[0] * v[1] +v[4] *v[5] +1)*x0[0] *x0[1] + (v[1] * v[1] + v[5] * v[5] + 1)*x0[1] * x0[1] - v[8] * v[8];
                    Y0_Jac(0,1)  = (v[0]* v[0]+ v[4]* v[4]+ 1)*x0[0]* x0[0] - 2 * (v[0] * v[2] +v[4] *v[6] +1)*x0[0] *x0[2] + (v[2] * v[2] + v[6] * v[6] + 1)*x0[2] * x0[2] - v[9] * v[9];
                    Y0_Jac(0,2)  = (v[1] *v[1]+ v[5] *v[5]+ 1)*x0[1]* x0[1] - 2 * (v[1] * v[2] +v[5] *v[6] +1)*x0[1] *x0[2] + (v[2] * v[2] + v[6] * v[6] + 1)*x0[2] *x0[2] - v[10] * v[10];
                    Y0_Jac(0,3)  = (v[3]* v[3]+ v[7]* v[7]+ 1)*x0[3]* x0[3] - 2 * (v[3] * v[2] +v[7] *v[6] +1)*x0[3] *x0[2] + (v[2] * v[2] + v[6] * v[6] + 1)*x0[2] *x0[2] - v[11] * v[11];
                    Y0_Jac(0,4)  = (v[0]*x0[0]-v[1]*x0[1])*(v[6]*x0[2]-v[7]*x0[3])-(v[2]*x0[2]-v[3]*x0[3])*(v[4]*x0[0]-v[5]*x0[1]);
                    Y0_Jac(0,5)  = (v[0]*x0[0]-v[1]*x0[1])*(x0[2]-x0[3])-(v[2]*x0[2]-v[3]*x0[3])*(x0[0]-x0[1]);
                    Y0_Jac(0,6)  = (v[4]*x0[0]-v[5]*x0[1])*(x0[2]-x0[3])-(v[6]*x0[2]-v[7]*x0[3])*(x0[0]-x0[1]);
                
                    Eigen::JacobiSVD<Eigen::MatrixXd> svd_2(jacobian_temp, Eigen::ComputeFullU | Eigen::ComputeFullV);//M=USV*
                    // double  pinvtoler = 1.e-8; //tolerance
                    int row_invJav_T = jacobian_temp.rows();
                    int col_invJav_T= jacobian_temp.cols();
                    int k_invJav_T =row_invJav_T>=col_invJav_T? col_invJav_T:row_invJav_T;
                    Eigen::MatrixXd X_invJav_T = Eigen::MatrixXd::Zero(col_invJav_T,row_invJav_T);
                    Eigen::MatrixXd singularValues_inv_invJav_T = svd_2.singularValues();//奇异值
                    Eigen::MatrixXd singularValues_inv_mat_invJav_T = Eigen::MatrixXd::Zero(col_invJav_T, row_invJav_T);
                    for (long i = 0; i<k_invJav_T; ++i) 
                    {
                        if (singularValues_inv_invJav_T(i) > pinvtoler)
                            singularValues_inv_invJav_T(i) = 1.0 / singularValues_inv_invJav_T(i);
                        else 
                            singularValues_inv_invJav_T(i) = 0;
                    }
                    for (long i = 0; i < k_invJav_T; ++i) 
                    {
                        singularValues_inv_mat_invJav_T(i, i) = singularValues_inv_invJav_T(i);
                    }
                    X_invJav_T=(svd_2.matrixV())*(singularValues_inv_mat_invJav_T)*(svd_2.matrixU().transpose());//X=VS+U* 4*7矩阵
                
                    Z_Jac=W_Jac-Y0_Jac*X_invJav_T;

                    for(int i=0;i<4;i++)     
                    {
                        x0[i]=Z_Jac(0,i);
                        X0_Jac(0,i)=Z_Jac(0,i);
                        x_start[i]=Z_Jac(0,i);   
                    }
                    
                    if(  ((x0_temp[0]-Z_Jac(0,0))+ (x0_temp[1]-Z_Jac(0,1))+ (x0_temp[2]-Z_Jac(0,2))+ (x0_temp[3]-Z_Jac(0,3)))  < Epsilon )
                    {
                        break;
                    }
                } while (iter<Max);

                x0[0]=fabs(x0[0]);
                x0[1]=fabs(x0[1]);
                x0[2]=fabs(x0[2]);
                x0[3]=fabs(x0[3]);
 ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                std::cout<<"深度：P1 " << x0[0] << " P2 " << x0[1] << " P3 " << x0[2] << " P4 " << x0[3] << "cm" <<std::endl;

                float X1_camera=trans_P1[0]*x0[0];
                float Y1_camera=trans_P1[1]*x0[0];
                float X2_camera=trans_P2[0]*x0[1];
                float Y2_camera=trans_P2[1]*x0[1];
                float X3_camera=trans_P3[0]*x0[2];
                float Y3_camera=trans_P3[1]*x0[2];
                float X4_camera=trans_P4[0]*x0[3];
                float Y4_camera=trans_P4[1]*x0[3];

                // ////////////////////////////////////////////////////相机坐标////////////////////////////////////////////////
                //相机正前方是y轴方向，右边是x轴方向，正上方是z轴方向 Pc（Xc, Yc, Zc）
                //1、深度是沿着y轴方向伸展的，因此深度x0[4]要放在Yc的位置上，Yc = x0[]
                //2、在图像中y轴(v轴)方向是朝下的，而在相机坐标系中Z轴是朝正上方的，因此Zc = - Y_camera
                // Pc1 << X1_camera, x0[0], -Y1_camera;//左上顶点在相机坐标系下的坐标
                // Pc2 << X2_camera, x0[1], -Y2_camera;//右上顶点在相机坐标系下的坐标
                // Pc3 << X3_camera, x0[2], -Y3_camera;//左下顶点在相机坐标系下的坐标
                // Pc4 << X4_camera, x0[3], -Y4_camera;//右下顶点在相机坐标系下的坐标

                //中心点的相机坐标，Yc上面加余量是因为相机与飞机中心不是重合的，所以要补偿
                Pc_central<< (X1_camera+X4_camera+X2_camera+X3_camera)/4, (x0[1]+x0[2]+x0[0]+x0[3])/4, (-Y2_camera-Y3_camera-Y1_camera-Y4_camera)/4;
                
                std::cout << "目标中心的相机坐标Pc:  (" << Pc_central[0] << ",  " << Pc_central[1] << ",  " << Pc_central[2] << ")cm"<< std::endl;

                ros::spinOnce();
            
                std::cout << "目标中心的世界坐标 Pw: (" << P_worldcentral[0] << ", " << P_worldcentral[1] << ", " << P_worldcentral[2] << ")cm\n\n";
                cv::imshow("image_detetct", image_raw);
                cv::waitKey(1);      

            }
        }

        catch(std::exception& e){
            std::cout<<"发生错误"<<std::endl;
            std::cout << e.what() << std::endl;  
        }

        cap.release();
    }     
    return 0;
}
