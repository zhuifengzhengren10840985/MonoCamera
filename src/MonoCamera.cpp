
#include <cmath>
#include <string>
#include <fstream>
#include<sstream>
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

using namespace cv;
using namespace std;

//模板总数
#define template_nums   (88)
//轮廓模板集
vector< vector<Point> > mContoursTemp;  
//待处理轮廓集
vector< vector<Point> > mContours;  

//ROI左右加宽值
#define wideRoi 10

string path = "/home/lxl/catkin_ws/src/MonoCamera/";

//像素坐标自定义结构体
struct point2D
{
    point2D(int _x, int _y) : x(_x), y(_y){}

    int x;
    int y;
};

//sort函数的第三个参数的仿函数
bool comp1(const point2D &a, const point2D &b){
    return a.x < b.x;
}
bool comp2(const point2D &a, const point2D &b){
    return a.y < b.y;
}

//Jacobi
#define J_N 4
//#define Epsilon 35
#define Max 15
const int N2 = 2 * J_N;
//深度值
double depth[J_N] = {300,300,300,300};
//牛顿迭代的初值(深度Z的初值)
double x_start[J_N] = {300,300,300,300};
//检测到目标的标志位
bool isDetected = false;

//窗口名称
const cv::String window_capture_name = "Video Capture";
const cv::String NumberDetection = "Number Detection";
const cv::String SquareDetection = "Square Detection";

//数字检测的滑动条
int low_H_white1 = 0, 
       high_H_white1 = 89, 

       low_H_white2 = 90, 
       high_H_white2 = 180, 

       low_S_white = 0, 
       high_S_white = 255, 

       low_V_white = 0,
       high_V_white = 255;

//边框检测的滑动条
int low_H_red1 = 0, 
       high_H_red1 = 89, 

       low_H_red2 = 90, 
       high_H_red2 = 180, 

       low_S_red = 0, 
       high_S_red = 255, 

       low_V_red = 0,
       high_V_red = 255;

//匹配数字
 int number = 1;

//相机坐标
Eigen::Vector3d Pc1,Pc2,Pc3, Pc4, Pc_central;

//旋转矩阵3x3
Eigen::Matrix3d R_Euler, R_Euler1;
//平移矩阵3x1
Eigen::Vector3d t_Euler;
//欧式变换矩阵4x4
Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
//顶点及中心点的世界坐标
Eigen::Vector3d P_world1, P_world2, P_world3, P_world4, P_worldcentral;


//////////////////////////////////////////////////////////////////////////////////数字滑动条///////////////////////////////////////////////////////////////////////////////

static void on_low_H1_thresh_trackbar1(int, void *)
{
    low_H_white1 = std::min(high_H_white1-1, low_H_white1);//防止最大值小于最小值
	cv::setTrackbarPos("Low H1", NumberDetection, low_H_white1);
}
static void on_high_H1_thresh_trackbar1(int, void *)
{
    high_H_white1 = std::max(high_H_white1, low_H_white1+1);
	setTrackbarPos("High H1", NumberDetection, high_H_white1);
}

static void on_low_H2_thresh_trackbar1(int, void *)
{
    low_H_white2 = std::min(high_H_white2-1, low_H_white2);//防止最大值小于最小值
	cv::setTrackbarPos("Low H2", NumberDetection, low_H_white2);
}
static void on_high_H2_thresh_trackbar1(int, void *)
{
    high_H_white2 = std::max(high_H_white2, low_H_white2+1);
	setTrackbarPos("High H2", NumberDetection, high_H_white2);
}

static void on_low_S_thresh_trackbar1(int, void *)
{
    low_S_white = std::min(high_S_white-1, low_S_white);
	setTrackbarPos("Low S",NumberDetection,low_S_white);
}
static void on_high_S_thresh_trackbar1(int, void *)
{
    high_S_white = std::max(high_S_white,low_S_white+1);
	setTrackbarPos("High S",NumberDetection,high_S_white);
}

static void on_low_V_thresh_trackbar1(int, void *)
{
    low_V_white = std::min(high_V_white-1, low_V_white);
	setTrackbarPos("Low V",NumberDetection,low_V_white);
}
static void on_high_V_thresh_trackbar1(int, void *)
{
    high_V_white = std::max(high_V_white,low_V_white+1);
	setTrackbarPos("High V",NumberDetection,high_V_white);
}


//////////////////////////////////////////////////////////////////////////////////////边框滑动条///////////////////////////////////////////////////////////////////////////////

static void on_low_H1_thresh_trackbar2(int, void *)
{
    low_H_red1 = std::min(high_H_red1-1, low_H_red1);//防止最大值小于最小值
	cv::setTrackbarPos("Low H1", SquareDetection, low_H_red1);
}
static void on_high_H1_thresh_trackbar2(int, void *)
{
    high_H_red1 = std::max(high_H_red1, low_H_red1+1);
	setTrackbarPos("High H1", SquareDetection, high_H_red1);
}

static void on_low_H2_thresh_trackbar2(int, void *)
{
    low_H_red2 = std::min(high_H_red2-1, low_H_red2);
	cv::setTrackbarPos("Low H2", SquareDetection, low_H_red2);
}
static void on_high_H2_thresh_trackbar2(int, void *)
{
    high_H_red2 = std::max(high_H_red2, low_H_red2+1);
	setTrackbarPos("High H2", SquareDetection, high_H_red2);
}

static void on_low_S_thresh_trackbar2(int, void *)
{
    low_S_red = std::min(high_S_red-1, low_S_red);
	setTrackbarPos("Low S",SquareDetection,low_S_red);
}
static void on_high_S_thresh_trackbar2(int, void *)
{
    high_S_red = std::max(high_S_red,low_S_red+1);
	setTrackbarPos("High S",SquareDetection,high_S_red);
}

static void on_low_V_thresh_trackbar2(int, void *)
{
    low_V_red = std::min(high_V_red-1, low_V_red);
	setTrackbarPos("Low V",SquareDetection,low_V_red);
}
static void on_high_V_thresh_trackbar2(int, void *)
{
    high_V_red = std::max(high_V_red,low_V_red+1);
	setTrackbarPos("High V",SquareDetection,high_V_red);
}

////////////////////////////////////////////////////////////////////////////////////////两幅图相减//////////////////////////////////////////////////////////////////////////////////

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

//////////////////////////////////////////////////////////////////////////////////////////获取数字///////////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////////////////模板匹配//////////////////////////////////////////////////////////////////////////////////

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
	for (int i = 0; i < 158; ++i){
		char name[520];
        std::sprintf(name, "/home/lxl/catkin_ws/src/MonoCamera/template/%d.jpg", i);
		src_image = cv::imread(name);
		if (!src_image.data) 
            std::cout << "数字模板库读取失败！" << std::endl; 
		result = substract(img, src_image);
		c = get_One(result);
		if (c < max) {
			max = c; 
            seriesNum = i;
		}
	}
	return seriesNum;
}


//////////////////////////////////////////////////////////////////////////////////载入模板的轮廓/////////////////////////////////////////////////////////////////////////////////
void init_hand_template(void)
{
    Mat srcImage;
    Mat dstImage;
    vector< vector<Point> > mContours;
    vector< Vec4i > mHierarchy;
 
    for(int i = 0; i < 88; i++)
    {
        char tmp_names[500];
        std::sprintf(tmp_names, "/home/lxl/catkin_ws/src/MonoCamera/Template_contours/%d.jpg", i);
        // 读取文件
        srcImage = imread(tmp_names, IMREAD_GRAYSCALE);
        // imshow("srcImage", srcImage);
        if(true == srcImage.empty())
        {
            cout << "模板图像读取失败！" << tmp_names <<endl;
            continue;
        }
 
        dstImage = Mat::zeros(srcImage.rows, srcImage.cols, CV_8UC3);
 
        // 寻找轮廓
        findContours(srcImage, mContours, mHierarchy, RETR_CCOMP, CHAIN_APPROX_NONE, Point(0, 0));
 
        if(mContours.size() > 0)
        {
            drawContours(dstImage, mContours, -1, Scalar(255, 255, 255), 1, 8);
            // drawContours(image_contours, mContours, -1, Scalar(255, 255, 255), 1, 8);
            // imshow("TempContours", dstImage);
            // waitKey(1);
            // imwrite(path + "contours.jpg", dstImage);
            mContoursTemp.insert(mContoursTemp.end(), mContours.begin(), mContours.end());
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////在图片中标注数字//////////////////////////////////////////////////////////////////////////////////
void drawNumber(Mat &img, int num){
    //如果未识别到任何数字
    if(num < 0){
        return;
    }

    stringstream st;
    st << num;
    string text = st.str();
    putText(img, text, Point(5, 100), FONT_HERSHEY_SIMPLEX, 4, Scalar(0, 0, 255), 5);
}


////////////////////////////////////////////////////////////////////////////////////四元数－>欧拉角/////////////////////////////////////////////////////////////////////////////////
struct Quaternion {
    double w, x, y, z;
};
 
struct EulerAngles {
    double roll, pitch, yaw;
};
 
EulerAngles ToEulerAngles(Quaternion q) {
    EulerAngles angles;

    // roll (X轴旋转)
    double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    angles.roll = std::atan2(sinr_cosp, cosr_cosp);
 
    // pitch (Ｙ轴旋转)
    double sinp = 2 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1)
        angles.pitch = std::copysign(M_PI / 2, sinp); 
    else
        angles.pitch = std::asin(sinp);
 
    // yaw (Ｚ轴旋转)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    angles.yaw = std::atan2(siny_cosp, cosy_cosp);
 
    return angles;
}


///////////////////////////////////////////////////////////////////////////四元数－>旋转矩阵//////////////////////////////////////////////////////////////////////
Eigen::Matrix3d QuaToRotationMatrix(Quaternion q){
    double R11, R12, R13, R21, R22, R23, R31, R32, R33;

    R11 = 1 - 2 *q. y * q.y  - 2 * q.z * q.z;
    R12 = 2 * q.x * q.y + 2 * q.w * q.z;
    R13 = 2 * q.x *q.z - 2 * q.w * q.y;

    R21 = 2 * q.x * q.y - 2 * q.w * q.z;
    R22 = 1 - 2 * q.x * q.x - 2 * q.z * q.z;
    R23 = 2 * q.y * q.z + 2 * q.w * q.x;

    R31 = 2 * q.x * q.z + 2 * q.w * q.y;
    R32 = 2 * q.y * q.z - 2 * q.w * q.x;
    R33 = 1 - 2 * q.x * q.x - 2 * q.y * q.y;

    Eigen::Matrix3d R;
    R << R11, R12, R13,
              R21, R22, R23,
              R31, R32, R33;
    return R;
}


////////////////////////////////////////////////////////////////////////////欧拉角－>旋转矩阵//////////////////////////////////////////////////////////////////
Eigen::Matrix3d EulerToRotationMatrix(EulerAngles angles){
    Eigen::Matrix3d Rx_, Ry_, Rz_, Rzyx_;
    double setaX = angles.roll;
    double setaY = angles.pitch;
    double setaZ = angles.yaw;

    Rx_ << 1, 0, 0, 
                  0, cos(setaX), -sin(setaX), 
                  0,sin(setaX), cos(setaX);
    
    Ry_ << cos(setaY),0,sin(setaY), 
                  0, 1, 0, 
                  -sin(setaY),0,cos(setaY);

    Rz_ << cos(setaZ),-sin(setaZ),0, 
                  sin(setaZ),cos(setaZ), 0, 
                  0, 0,1;

    Rzyx_ << Rz_ * Ry_ * Rx_;

    return Rzyx_;
}

//////////////////////////////////////////////////////////////////////////////订阅＋发布////////////////////////////////////////////////////////////////////////

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
    sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("mavros/local_position/pose", 
                                                                                                                        10, 
                                                                                                                        &SubAndPub::poseCallback, 
                                                                                                                        this);//话题前面不加/表示相对命名空间,加了表示全局命名空间
}
//订阅无人机位姿信息的回调函数,利用这些信息求出目标世界坐标,将目标信息发布出去
void SubAndPub::poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    //四元数
    Quaternion Q;
    Q.x = msg->pose.orientation.x;
    Q.y = msg->pose.orientation.y;
    Q.z = msg->pose.orientation.z;
    Q.w = msg->pose.orientation.w;

    EulerAngles angles = ToEulerAngles(Q);

    std::cout << "EulerAngles: " << angles.roll << " " << angles.pitch << " " << angles.yaw << endl;

    R_Euler = EulerToRotationMatrix(angles);

    std::cout << "RotationMatrix: " << R_Euler << endl; 

    R_Euler1 = QuaToRotationMatrix(Q);
    std::cout << "RotationMatrix1: " << R_Euler1 << endl; 

    t_Euler << msg->pose.position.x * 100,
                          msg->pose.position.y * 100,
                          msg->pose.position.z * 100;
    
    std::cout << "t_Euler:(" << t_Euler[0] << "," << t_Euler[1] << "," <<t_Euler[2] << ")" <<endl;

    P_worldcentral = R_Euler.inverse() * Pc_central + t_Euler;

    //初始化MonoCamera::object类型的消息
    MonoCamera::object obj_output;
    obj_output.isDetected = isDetected;
    obj_output.object_number = number;
    obj_output.object_position.x = P_worldcentral[0];
    obj_output.object_position.y = P_worldcentral[1];
    obj_output.object_position.z = P_worldcentral[2];
    //发布者发布消息
    pub_.publish(obj_output);
}


//////////////////////////////////////////////////////////////////将数据记录到txt文件中///////////////////////////////////////////////////////////////
void writeData(){
    //在当前目录下新建一个txt文件
    ofstream outfile;
    //打开文件
    // outfile.open("/home/lxl/catkin_ws/src/MonoCamera/ObjPos.txt", std::ios::app);
    outfile.open(path + "ObjPos.txt", std::ios::app);

    //如果读取txt文件失败，就打印提示信息
    if(!outfile){
        std::cout << "TXT文件读取失败！" << endl;
    }
    
    //将数据写入txt文件
    outfile << isDetected << "\t" << number << "\t" << Pc_central[0] << "\t" << Pc_central[1] << "\t" << Pc_central[2] << "\n";

    //关闭文件
    outfile.close();
}


//////////////////////////////////////////////////////////////////////计算时间周期////////////////////////////////////////////////////////////////////
float getDeltaTime(const ros::Time& _lastTime){
    ros::Time current_time = ros::Time::now();
    float current_time_sec = current_time.sec - _lastTime.sec;
    float current_time_nsec = current_time.nsec / 1e9 - _lastTime.nsec / 1e9;
    return (current_time_sec + current_time_nsec);
}


/////////////////////////////////////////////////////////////////////主函数/////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    ros::init(argc,argv,"sub_and_pub");
    ros::NodeHandle nh;

    ros::Publisher pub = nh.advertise<MonoCamera::object>("object_pub", 1);
    ros::Rate loop_rate(10);//消息发布的频率为10Hz
    // SubAndPub SAPObject;

    //原始,灰度,hsv,颜色分割图像
    cv::Mat image_raw, image_gray, image_hsv;
    cv::Mat image_threshold_white1,image_threshold_white2, image_threshold_white;
    cv::Mat image_threshold_red1, image_threshold_red2, image_threshold_red;

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    //启动编号为0的相机
    cv::VideoCapture cap(0);
    //设置图像的长宽和帧率
    cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,480); 

    //图像帧率fps 30
    double fps = cap.get(CAP_PROP_FPS);
    //图像尺寸size 640 x 480
    Size size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter writer(path + "video.avi", CV_FOURCC('M', 'P', '4', '2'), fps , size, true);

    if(!cap.isOpened())
    {
        std::cout << "相机打开失败！" << std::endl;
        return -1;
    }
    try{
        while(1){
            //按Esc键退出while
            char c = (char)waitKey(25);
            if (c == 27)
                break;

            //从相机读取视频流
            cap >> image_raw;

            //图像为空就退出程序
            if(image_raw.empty())
                break;
            
            cv::imshow("image_raw", image_raw);

            //颜色空间转换
            cv::cvtColor(image_raw, image_hsv, cv::COLOR_BGR2HSV);  

/////////////////////////////////////////////////////////////////////检测数字区域/////////////////////////////////////////////////////////////////////////
            //数字检测HSV阈值调节窗口
            namedWindow(NumberDetection, 0);
            resizeWindow(NumberDetection, 640, 480);

            createTrackbar("Low H1", NumberDetection, &low_H_white1, 89, on_low_H1_thresh_trackbar1);
            createTrackbar("High H1", NumberDetection, &high_H_white1, 89, on_high_H1_thresh_trackbar1);

            createTrackbar("Low H2", NumberDetection, &low_H_white2, 180, on_low_H2_thresh_trackbar1);
            createTrackbar("High H2", NumberDetection, &high_H_white2, 180, on_high_H2_thresh_trackbar1);

            createTrackbar("Low S", NumberDetection, &low_S_white, 255, on_low_S_thresh_trackbar1);
            createTrackbar("High S", NumberDetection, &high_S_white, 255, on_high_S_thresh_trackbar1);

            createTrackbar("Low V", NumberDetection, &low_V_white, 255, on_low_V_thresh_trackbar1);
            createTrackbar("High V", NumberDetection, &high_V_white, 255, on_high_V_thresh_trackbar1);
            
            //进行颜色分割，输出图像为CV_8UC1
            // cv::inRange(image_hsv, cv::Scalar(0, 0, 221), cv::Scalar(180,30,255), image_threshold_white);

            cv::inRange(image_hsv, cv::Scalar(low_H_white1,low_S_white,low_V_white), cv::Scalar(high_H_white1,high_S_white,high_V_white), image_threshold_white1);
            cv::inRange(image_hsv, cv::Scalar(low_H_white2,low_S_white,low_V_white), cv::Scalar(high_H_white2,high_S_white,high_V_white), image_threshold_white2);
            cv::bitwise_or(image_threshold_white1, image_threshold_white2, image_threshold_white);

            //形态学运算，先腐蚀erode再膨胀dilate
            cv::Mat image_erode, image_dilate;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(4,4));
            cv::erode(image_threshold_white, image_erode, kernel);
            cv::dilate(image_erode, image_dilate, kernel);
            cv::imshow(NumberDetection, image_dilate);

            //寻找轮廓
            cv::findContours(image_dilate, contours, hierarchy, RETR_EXTERNAL, cv::CHAIN_APPROX_NONE), Point(0,0);

            //检测不到轮廓的话就不进行后面操作
            if(contours.empty())
                continue;

            //画轮廓
            cv::Mat image_contours = Mat::zeros(image_raw.size(), CV_8UC3);
            // cv::drawContours(image_dilate, contours, -1, cv::Scalar(255, 255, 255), 1, 8, hierarchy);
            // imshow("image_contours", image_dilate);

            const int contoursSize = contours.size();//轮廓个数
            double contoursArea = 0;//轮廓面积
            double imageArea = size.area();//图像面积

            mContours.clear();//清空上次图像处理的轮廓

            //过滤掉小噪点和靠近图片边界的轮廓
            for(int i = 0; i < contoursSize; i++){
                //过滤掉小面积的轮廓（噪声）
                contoursArea = contourArea(contours[i]);
                if(contoursArea / imageArea < 0.005){
                    continue;
                }
                //如果轮廓边界与窗口贴近或者相连，则排除
                Rect bound = boundingRect(contours[i]);
                if(bound.x < 2 || bound.y < 2 || (bound.x + bound.width + 2) > size.width || (bound.y + bound.height + 2) > size.height){
                        continue;
                }
                //剩下的轮廓就是基本符合条件的轮廓，保存下来
                mContours.push_back(contours[i]);
            }

            if(mContours.empty()){
                 continue;
            }

            image_contours = Scalar::all(0);
            drawContours(image_contours, mContours, -1, Scalar(255, 255, 255), 1, 8);
            imshow("Filter Contours", image_contours);

            //取外轮廓squares, 其中有多个轮廓，每个轮廓含有很多个点
            std::vector<std::vector<cv::Point> > squares;
            std::list<int> len_list;

            //计算每个轮廓的面积
            for (int i = 0; i < contours.size(); i++){
                len_list.push_back(cv::contourArea(contours[i]));
            }

            //寻找面积最大的轮廓
            if (len_list.size() > 0){
                squares.push_back(contours[std::distance(len_list.begin(), max_element(len_list.begin(), len_list.end()))]);
            }

            //最小外接矩形集合box
            std::vector<cv::RotatedRect> box(squares.size());
            std::vector<cv::Rect> srcRect(squares.size());
            cv::Mat image_raw_copy = image_raw;

            for(int i = 0; i < squares.size(); i++){
                srcRect[i] = cv::boundingRect(squares[i]);
                //计算轮廓i的最小外接矩形
                box[i] = cv::minAreaRect(cv::Mat(squares[i]));
                //绘制最小外接矩形的中心点
                cv::circle(image_raw_copy, cv::Point(box[i].center.x, box[i].center.y), 1, cv::Scalar(255, 0, 0), -1, 8); 
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

            cv::circle(image_raw_copy, rect1[0], 4, cv::Scalar(0, 0, 255), -1, 8); //绘制最小外接正方形的左上角顶点
            cv::circle(image_raw_copy, rect1[1], 4, cv::Scalar(0, 0, 255), -1, 8); //绘制最小外接正方形的右上角顶点
            cv::circle(image_raw_copy, rect1[2], 4, cv::Scalar(0, 0, 255), -1, 8); //绘制最小外接正方形的右下角顶点
            cv::circle(image_raw_copy, rect1[3], 4, cv::Scalar(0, 0, 255), -1, 8); //绘制最小外接正方形的左下角顶点

            //绘制最小外接正方形的每条边
            for(int i = 0; i < 4; i++)
                line(image_raw_copy, rect1[i], rect1[(i+1)%4], cv::Scalar(255, 0, 0), 2, 8);

            //抠取出数字区域的二值图像
            cv::Mat roi;
            int roi_wid = abs(rect1[1].x - rect1[0].x);//方形ROI区域边长

            int min_x = std::min({rect1[0].x, rect1[1].x, rect1[2].x, rect1[3].x});
            int min_y = std::min({rect1[0].y, rect1[1].y, rect1[2].y, rect1[3].y});
            int max_x = std::max({rect1[0].x, rect1[1].x, rect1[2].x, rect1[3].x});
            int max_y = std::max({rect1[0].y, rect1[1].y, rect1[2].y, rect1[3].y});

            if(min_x > 0 && min_y > 0 && max_x < image_raw.cols && max_y < image_raw.rows){
                roi = image_dilate(cv::Rect(rect1[0].x, rect1[0].y, roi_wid, roi_wid));
                cv::imshow("resizeRoi", roi);
                cv::resize(roi, roi, cv::Size(300, 300));
                cv::imwrite(path + "roi.jpg", roi);
            }

            Mat srcImage;
            Mat dstImage;
            vector< vector<Point> > mContours1;
            vector< Vec4i > mHierarchy;
        
            // 读取文件
            srcImage = imread("/home/lxl/catkin_ws/src/MonoCamera/roi.jpg", IMREAD_GRAYSCALE);
    
            dstImage = Mat::zeros(srcImage.rows, srcImage.cols, CV_8UC3);
    
            // 寻找轮廓
            findContours(srcImage, mContours1, mHierarchy, RETR_CCOMP, CHAIN_APPROX_NONE, Point(0, 0));
    
            if(mContours1.size() > 0)
            {
                drawContours(dstImage, mContours1, -1, Scalar(255, 255, 255), 1, 8);
                imshow("TempContours", dstImage);
                waitKey(1);
                imwrite(path + "contours.jpg", dstImage);
                mContoursTemp.insert(mContoursTemp.end(), mContours1.begin(), mContours1.end());
            }

            // roi = cv::imread(path + "roi.jpg");

            // double hu = 1.0;
            // double huTmp = 0.0;//匹配度
            // int match_number = -1;//匹配到的数字
            // int mcontours_index = -1, template_index = -1;//轮廓标号和模板标号

            // //载入模板轮廓
            // // init_hand_template();

            // for(int i = 0; i < template_nums; i++){
            //     for(int j = 0; j < mContours.size(); j++){
            //         //第i个模板和第j个轮廓做匹配
            //         huTmp = matchShapes(mContoursTemp[i], mContours[j], CV_CONTOURS_MATCH_I1, 0);

            //         //hu矩越小，匹配度越高
            //         if(huTmp < hu){
            //             hu = huTmp;

            //             //保存是哪个轮廓和哪个模板匹配上了
            //             mcontours_index = i;
            //             template_index = j;
            //         }
            //     }
            // }
            
            // cout << "mcontours_index=" << mcontours_index << "; template_index = " << template_index << "; hu = " << hu <<endl;
            // //匹配到的数字
            // match_number = m;

            // //模板匹配识别数字
            // int serise_num;
            // serise_num = temple_image(roi);
            // //0
            // if(serise_num >= 0 && serise_num <= 29){
            //     number = 0;
            // }
            // //1
            // else if(serise_num >= 30 && serise_num <= 51){
            //     number = 1;
            // }
            // //2
            // else if(serise_num >= 52 && serise_num <= 119){
            //     number = 2;
            // }
            // //3
            // else if(serise_num >= 120 && serise_num <= 158){
            //     number = 3;
            // }
            // std::cout << "number:  " << number << std::endl;


///////////////////////////////////////////////////////////////////检测红色边框////////////////////////////////////////////////////////////////////
            // //边框检测HSV阈值调节窗口
            // namedWindow(SquareDetection, 0);
            // resizeWindow(SquareDetection, 640, 480);

            // createTrackbar("Low H1", SquareDetection, &low_H_red1, 89, on_low_H1_thresh_trackbar2);
            // createTrackbar("High H1", SquareDetection, &high_H_red1, 89, on_high_H1_thresh_trackbar2);

            // createTrackbar("Low H2", SquareDetection, &low_H_red2, 180, on_low_H2_thresh_trackbar2);
            // createTrackbar("High H2", SquareDetection, &high_H_red2, 180, on_high_H2_thresh_trackbar2);

            // createTrackbar("Low S", SquareDetection, &low_S_red, 255, on_low_S_thresh_trackbar2);
            // createTrackbar("High S", SquareDetection, &high_S_red, 255, on_high_S_thresh_trackbar2);

            // createTrackbar("Low V", SquareDetection, &low_V_red, 255, on_low_V_thresh_trackbar2);
            // createTrackbar("High V", SquareDetection, &high_V_red, 255, on_high_V_thresh_trackbar2);
            
            // //颜色分割出红色,得到二值图像image_threshold_red
            // // cv::inRange(image_hsv, cv::Scalar(0, 43, 46), cv::Scalar(10, 255, 255), image_threshold_red1);
            // // cv::inRange(image_hsv, cv::Scalar(156, 43, 46), cv::Scalar(180, 255, 255), image_threshold_red2);
            // // cv::bitwise_or(image_threshold_red1, image_threshold_red2, image_threshold_red);

            // cv::inRange(image_hsv, cv::Scalar(low_H_red1,low_S_red,low_V_red), cv::Scalar(high_H_red1,high_S_red,high_V_red), image_threshold_red1);
            // cv::inRange(image_hsv, cv::Scalar(low_H_red2,low_S_red,low_V_red), cv::Scalar(high_H_red2,high_S_red,high_V_red), image_threshold_red2);
            // cv::bitwise_or(image_threshold_red1, image_threshold_red2, image_threshold_red);

            // int min_x = std::min({rect1[0].x, rect1[1].x, rect1[2].x, rect1[3].x});
            // int min_y = std::min({rect1[0].y, rect1[1].y, rect1[2].y, rect1[3].y});
            // int max_x = std::max({rect1[0].x, rect1[1].x, rect1[2].x, rect1[3].x});
            // int max_y = std::max({rect1[0].y, rect1[1].y, rect1[2].y, rect1[3].y});
            // /*同时满足以下条件：
            //             roi.x >= 0
            //             roi.y >= 0
            //             roi.width >= 0
            //             roi.height >= 0
            //             roi.x + roi.width <= m.cols 
            //             roi.y + roi.height <= m.rows
            // */
            // cv::Mat image_roi;//屏蔽掉非ROI区域后的图像

            // if((min_x - wideRoi) >= 0 && (min_y - wideRoi) >= 0 && roi_wid >=0 && (max_x + wideRoi) < image_raw.cols  && (max_y + wideRoi) < image_raw.rows){
            //     cv::Rect roi_rect(rect1[0].x - wideRoi, rect1[0].y - wideRoi, roi_wid + 20, roi_wid + 20); //方形ROI区域(超出图像会报错)
            //     cv::Mat mask = cv::Mat::zeros(image_threshold_red.size(), CV_8UC1);//非ROI区域－＆0－屏蔽掉不感兴趣的区域
            //     mask(roi_rect).setTo(255);//ROI区域－＆1－保留住感兴趣的区域
            //     // cv::imshow("mask", mask);

            //     image_threshold_red.copyTo(image_roi, mask);//原图与mask图进行与运算
            //     cv::imshow("ROI", image_roi);
            // }
            // else{
            //     image_roi = image_threshold_red;
            // }


            // //形态学运算，先腐蚀(erode)再膨胀(dilate)
            // cv::Mat image_erode1, image_dilate1;
            // cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(4,4));
            // cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(4,4));
            // cv::erode(image_threshold_red, image_erode1, kernel1);
            // // imshow("red_erode",image_erode1);
            // cv::dilate(image_erode1, image_dilate1, kernel2);
            // cv::imshow(SquareDetection, image_dilate1);

            // // //形态学运算，先膨胀(dilate)再腐蚀(erode)
            // // cv::Mat image_erode1, image_dilate1;
            // // cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(4,4));
            // // cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(4,4));
            // // cv::dilate(image_threshold_red, image_dilate1, kernel1);
            // // cv::erode(image_dilate1, image_erode1, kernel2);
            // // cv::imshow(SquareDetection, image_erode1);

            // //寻找轮廓
            // std::vector<std::vector<cv::Point> > contours1;
            // std::vector<cv::Vec4i> hierarchy1;
            // cv::findContours(image_dilate1, contours1, hierarchy1, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);


//             //计算每个轮廓的面积
//             std::list<int> len_list1;
//             for (int i = 0; i < contours1.size(); i++){
//                 len_list1.push_back(cv::contourArea(contours1[i]));
//             }

//             //取面积最大的轮廓(外轮廓)
//             std::vector<std::vector<cv::Point> > squares1;
//             if (len_list1.size() > 0){
//                 squares1.push_back(contours1[std::distance(len_list1.begin(), max_element(len_list1.begin(), len_list1.end()))]);
//             }

//             // //取内轮廓
//             // std::vector<std::vector<cv::Point> > squares1;
//             // if (len_list1.size() > 0){
//             //     std::list<int>::iterator it = len_list1.begin();
//             //     advance(it,std::distance(len_list1.begin(), max_element(len_list1.begin(), len_list1.end())));
//             //     len_list1.erase(it);
//             //     squares1.push_back(contours1[std::distance(len_list1.begin(), max_element(len_list1.begin(), len_list1.end()))]);
//             // }

//             //检测不到轮廓的话就退出本次循环
//             if(contours1.empty()){
//                 continue;
//             }
                
// ///////////////////////////////////////////////////////////////////////////////拟合矩形框//////////////////////////////////////////////////////////////////////////////////
//             double epsilon = 0.1*arcLength(squares1[0], true);
//             std::vector<std::vector<cv::Point> > contourPoly(1);
//             approxPolyDP(squares1[0], contourPoly[0], epsilon, true);
//             Mat image_square(image_raw.size(), CV_8UC3, Scalar::all(0));//纯黑图像
//             drawContours(image_square, contourPoly, 0, Scalar(0, 255, 0), 2, 8);
//             //imshow("Square", image_square);
//             //凸包检测(可能检测出来不是4个点)
//             std::vector<cv::Point> squareFour;//四个角点
//             convexHull(contourPoly[0], squareFour, true, true);
//             //cout << "顶点个数为：" << squareFour.size() << endl;


// ///////////////////////////////////////////////////////////////////////////////对角点进行排序/////////////////////////////////////////////////////////////////////////////
// //顶点顺序
// //                           1 —— 2
// //                           /        / 
// //                        3 —— 4
// //
// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//             //四个角点和中心点(Point包含u和v值)
//             cv::Point min_Point1, min_Point2, min_Point3, min_Point4, min_central;
//             //首先判断是否有四个角点
//             if(squareFour.size() == 4){
//                 //检测标志位设置为true
//                 isDetected = true;

//                 //四个顶点
//                 vector<point2D> vertex;
//                 for(int i =0; i < 4; i++){
//                     point2D point_temp(squareFour[i].x, squareFour[i].y);
//                     vertex.push_back(point_temp);
//                 }

//                 //将四个顶点按照x升序排列(前两个是1,3, 后两个是2,4)
//                 sort(vertex.begin(), vertex.end(), comp1);
                
//                 //左侧角点
//                 vector<point2D> vertex13;
//                 vertex13.push_back(vertex[0]);
//                 vertex13.push_back(vertex[1]);

//                 //区分角点1和3
//                 if(vertex13[0].y < vertex13[1].y){
//                     min_Point1.x = vertex13[0].x;
//                     min_Point1.y = vertex13[0].y;
//                     min_Point3.x = vertex13[1].x;
//                     min_Point3.y = vertex13[1].y;
//                 }
//                 else{
//                     min_Point1.x = vertex13[1].x;
//                     min_Point1.y = vertex13[1].y;
//                     min_Point3.x = vertex13[0].x;
//                     min_Point3.y = vertex13[0].y;
//                 }

//                 //右侧角点
//                 vector<point2D> vertex24;
//                 vertex24.push_back(vertex[2]);
//                 vertex24.push_back(vertex[3]);

//                 //区分角点2和4
//                 if(vertex24[0].y < vertex24[1].y){
//                     min_Point2.x = vertex24[0].x;
//                     min_Point2.y = vertex24[0].y;
//                     min_Point4.x = vertex24[1].x;
//                     min_Point4.y = vertex24[1].y;
//                 }
//                 else{
//                     min_Point2.x = vertex24[1].x;
//                     min_Point2.y = vertex24[1].y;
//                     min_Point4.x = vertex24[0].x;
//                     min_Point4.y = vertex24[0].y;
//                 }
//                 min_central.x = (min_Point1.x + min_Point2.x + min_Point3.x + min_Point4.x)/4;
//                 min_central.y = (min_Point1.y + min_Point2.y + min_Point3.y + min_Point4.y)/4;

//                 circle(image_raw, min_Point1 , 3, Scalar(0, 255, 0), 2, 8, 0);
//                 circle(image_raw, min_Point2 , 3, Scalar(0, 255, 0), 2, 8, 0);
//                 circle(image_raw, min_Point3 , 3, Scalar(0, 255, 0), 2, 8, 0);
//                 circle(image_raw, min_Point4 , 3, Scalar(0, 255, 0), 2, 8, 0);
//                 circle(image_raw, min_central , 1, Scalar(0, 255, 0), 2, 8, 0);

//                 drawContours(image_raw, contourPoly, 0, Scalar(255, 255, 0), 1, 8);

// ///////////////////////////////////////////////////////////////////////////////求解空间坐标////////////////////////////////////////////////////////////////////////////
// // 顶点顺序
// //                         1——2
// //                          |        | 
// //                         3——4
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                 
//                 //像素坐标的齐次坐标(u, v, 1)
//                 Eigen::Vector3d min_P1,  min_P2,  min_P3 ,  min_P4;
//                 min_P1 << min_Point1.x, min_Point1.y, 1;
//                 min_P2 << min_Point2.x, min_Point2.y, 1;
//                 min_P3 << min_Point3.x, min_Point3.y, 1;
//                 min_P4 << min_Point4.x, min_Point4.y, 1;

// ///////////////////////////////////////////////////////////////////////////////相机参数///////////////////////////////////////////////////////////////////////////////////
//                 //相机的内参矩阵(3x3)
//                 Eigen::Matrix3d CameraM;
//                 Eigen::Vector4d CameraPram;
//                 CameraPram[0] = 366.997006;//fx
//                 CameraPram[1] = 366.447492;//fy
//                 CameraPram[2] = 331.133147;//cx
//                 CameraPram[3] = 239.626159;//cy
//                 CameraM << CameraPram[0], 0, CameraPram[2], 0, CameraPram[1], CameraPram[3], 0, 0, 1;

// //////////////////////////////////////////////////////////////////////////相机归一化坐标//////////////////////////////////////////////////////////////////////////////
//                 //相机归一化坐标（Xc/Zc, Yc/Zc)T =  相机内参矩阵的逆 *(u, v)T
//                 Eigen::Vector3d trans_P1d, trans_P2d, trans_P3d, trans_P4d;
//                 trans_P1d = CameraM.inverse() * min_P1;
//                 trans_P2d = CameraM.inverse() * min_P2;
//                 trans_P3d = CameraM.inverse() * min_P3;
//                 trans_P4d = CameraM.inverse() * min_P4;

// ///////////////////////////////////////////////////////////////牛顿迭代求解三元非线性方程组解算深度信息/////////////////////////////////////////////////////////
//                 float Epsilon =0.01;	
//                 Eigen::MatrixXd X1_Jac(1,4); 

//                 X1_Jac <<10,10,10,10;

//                 Eigen::MatrixXd Y0_Jac(1,7);

//                 Eigen::MatrixXd X0_Jac(1,4);
//                 double x_start[4];
//                 X0_Jac << x_start[0], x_start[1], x_start[2], x_start[3];
//                 Eigen::MatrixXd W_Jac(1,4);
//                 W_Jac<<100,100,100,100;
            
//                 Eigen::MatrixXd YY_Jac(1,4);
//                 Eigen::MatrixXd Y_temp_Jac(4,7);
//                 Eigen::MatrixXd jacobian_temp(4,7);
//                 Eigen::MatrixXd jacobian_temp_M(4,7);
//                 Eigen::MatrixXd Z_Jac(1,4);
//                 Eigen::MatrixXd	JAC(4,7);//jacobin矩阵

//                 double y0[J_N+3]={0,0,0,0,0,0,0}, jacobian[J_N][J_N+3], invjacobian[J_N+3][J_N], x1[J_N]={0,0,0,0}, errornorm, v[12], L=0;
//                 double D1 = 56.35, D2 = 56.35, D3 = 79.69, D4 = 56.35;//外轮廓实际大小(单位cm)
//                 int i_inv=0, j_inv=0, k_inv=0, i_jcb=0, iter = 0;

//                 v[0] = trans_P1d[0];//Xc1
//                 v[1] = trans_P2d[0];//Xc2
//                 v[2] = trans_P3d[0];//Xc3
//                 v[3] = trans_P4d[0];//Xc4
//                 v[4] = trans_P1d[1];//Yc1
//                 v[5] = trans_P2d[1];//Yc2
//                 v[6] = trans_P3d[1];//Yc3
//                 v[7] = trans_P4d[1];//Yc4
//                 v[8] = D1;
//                 v[9] = D2;
//                 v[10] = D3;
//                 v[11] = D4;

//                 do{
//                     iter = iter + 1;
//                     //计算雅克比矩阵 jacobian
//                     y0[0] = (v[0]* v[0]+ v[4]* v[4]+ 1)*depth[0]* depth[0] - 2 * (v[0] * v[1] +v[4] *v[5] +1)*depth[0] * depth[1] + (v[1] * v[1] + v[5] * v[5] + 1)*depth[1] * depth[1] - v[8] * v[8];
//                     y0[1] = (v[0]* v[0]+ v[4]* v[4]+ 1)*depth[0]* depth[0] - 2 * (v[0] * v[2] +v[4] *v[6] +1)*depth[0] *depth[2] + (v[2] * v[2] + v[6] * v[6] + 1)*depth[2] * depth[2] - v[9] * v[9];
//                     y0[2] = (v[1] *v[1]+ v[5] *v[5]+ 1)*depth[1]* depth[1] - 2 * (v[1] * v[2] +v[5] *v[6] +1)*depth[1] *depth[2] + (v[2] * v[2] + v[6] * v[6] + 1)*depth[2] *depth[2] - v[10] * v[10];
//                     y0[3] = (v[3]* v[3]+ v[7]* v[7]+ 1)*depth[3]* depth[3] - 2 * (v[3] * v[2] +v[7] *v[6] +1)*depth[3] *depth[2] + (v[2] * v[2] + v[6] * v[6] + 1)*depth[2] *depth[2] - v[11] * v[11];
//                     y0[4] = (v[0]*depth[0]-v[1]*depth[1])*(v[6]*depth[2]-v[7]*depth[3])-(v[2]*depth[2]-v[3]*depth[3])*(v[4]*depth[0]-v[5]*depth[1]);
//                     y0[5] = (v[0]*depth[0]-v[1]*depth[1])*(depth[2]-depth[3])-(v[2]*depth[2]-v[3]*depth[3])*(depth[0]-depth[1]);
//                     y0[6] = (v[4]*depth[0]-v[5]*depth[1])*(depth[2]-depth[3])-(v[6]*depth[2]-v[7]*depth[3])*(depth[0]-depth[1]);
//                     //计算初值位置的值
//                     //计算雅克比矩阵的逆矩阵 invjacobian
//                     //jacobian have n*n element  计算函数雅克比的值
//                     JAC(0,0)= 2 * (v[0] * v[0] + v[4] * v[4] + 1)*depth[0] - 2 * (v[0] * v[1] + v[4] * v[5] + 1)*depth[1];
//                     JAC(0,1) = 2 * (v[0] * v[0] + v[4] * v[4] + 1)*depth[0] - 2 * (v[0] * v[2] + v[4] * v[6] + 1)*depth[2];
//                     JAC(0,2)= 0;
//                     JAC(0,3) = 0;
//                     JAC(0,4) =  (v[0])*(v[6]*depth[2]-v[7]*depth[3])-(v[2]*depth[2]-v[3]*depth[3])*(v[4]); ;
//                     JAC(0,5) =  (v[0])*(depth[2]-depth[3])-(v[2]*depth[2]-v[3]*depth[3]);
//                     JAC(0,6) =  (v[4])*(depth[2]-depth[3])-(v[6]*depth[2]-v[7]*depth[3]);

//                     JAC(1,0) = -2 * (v[0] * v[1] + v[4] * v[5] + 1)*depth[0] + 2 * (v[1] * v[1] + v[5] * v[5] + 1)*depth[1];
//                     JAC(1,1) = 0;
//                     JAC(1,2) = 2 * (v[1] * v[1] + v[5] * v[5] + 1)*depth[1] - 2 * (v[1] * v[2] + v[5] * v[6] + 1)*depth[2];
//                     JAC(1,3) = 0;
//                     JAC(1,4) = (-v[1])*(v[6]*depth[2]-v[7]*depth[3])-(v[2]*depth[2]-v[3]*depth[3])*(-v[5]);
//                     JAC(1,5) = (-v[1])*(depth[2]-depth[3])-(v[2]*depth[2]-v[3]*depth[3])*(-1);
//                     JAC(1,6) = (-v[5])*(depth[2]-depth[3])-(v[6]*depth[2]-v[7]*depth[3])*(-1);

//                     JAC(2,0) = 0;
//                     JAC(2,1) = -2 * (v[0] * v[2] + v[4] * v[6] + 1)*depth[0] + 2 * (v[2] * v[2] + v[6] * v[6] + 1)*depth[2];
//                     JAC(2,2) = -2 * (v[1] * v[2] + v[5] * v[6] + 1)*depth[1] + 2 * (v[2] * v[2] + v[6] * v[6] + 1)*depth[2];
//                     JAC(2,3) = 2 * (v[2] * v[2] + v[6] * v[6] + 1)*depth[2] - 2 * (v[3] * v[2] + v[7] * v[6] + 1)*depth[3];
//                     JAC(2,4) = (v[0]*depth[0]-v[1]*depth[1])*(v[6])-(v[2])*(v[4]*depth[0]-v[5]*depth[1]);
//                     JAC(2,5) = (v[0]*depth[0]-v[1]*depth[1])*(1)-(v[2]*(depth[0]-depth[1]));
//                     JAC(2,6) = (v[4]*depth[0]-v[5]*depth[1])*(1)-(v[6]*(depth[0]-depth[1]));

//                     JAC(3,0) = 0;
//                     JAC(3,1) = 0;
//                     JAC(3,2) = 0;
//                     JAC(3,3) = 2 * (v[3] * v[3] + v[7] * v[7] + 1) * depth[3] -2 * (v[3] * v[2] + v[7] * v[6] + 1) *depth[2];
//                     JAC(3,4) = (v[0]*depth[0]-v[1]*depth[1])*(-v[7])-(-v[3])*(v[4]*depth[0]-v[5]*depth[1]);
//                     JAC(3,5) = (v[0]*depth[0]-v[1]*depth[1])*(-1)-(-v[3])*(depth[0]-depth[1]);
//                     JAC(3,6) = (v[4]*depth[0]-v[5]*depth[1])*(-1)-(-v[7])*(depth[0]-depth[1]);

//                     Eigen::JacobiSVD<Eigen::MatrixXd> svd(JAC, Eigen::ComputeFullU | Eigen::ComputeFullV);//M=USV*
//                     double  pinvtoler = 1.e-8; //tolerance
//                     int row_invJav = JAC.rows();
//                     int col_invJav= JAC.cols();
//                     int k_invJav =row_invJav>=col_invJav? col_invJav:row_invJav;
//                     Eigen::MatrixXd X_invJav = Eigen::MatrixXd::Zero(col_invJav,row_invJav);
//                     Eigen::MatrixXd singularValues_inv_invJav = svd.singularValues();//奇异值
//                     Eigen::MatrixXd singularValues_inv_mat_invJav = Eigen::MatrixXd::Zero(col_invJav, row_invJav);
//                     for (long i = 0; i<k_invJav; ++i) 
//                     {
//                         if (singularValues_inv_invJav(i) > pinvtoler)
//                             singularValues_inv_invJav(i) = 1.0 / singularValues_inv_invJav(i);
//                         else 
//                             singularValues_inv_invJav(i) = 0;
//                     }
//                     for (long i = 0; i < k_invJav; ++i) 
//                     {
//                         singularValues_inv_mat_invJav(i, i) = singularValues_inv_invJav(i);
//                     }
//                     X_invJav=(svd.matrixV())*(singularValues_inv_mat_invJav)*(svd.matrixU().transpose());//X=VS+U*

//                     //优化版迭代部分
//                     Y0_Jac<<y0[0],y0[1],y0[2],y0[3],y0[4],y0[5],y0[6];
                
//                     X1_Jac = X0_Jac-Y0_Jac*X_invJav;//x1=depth-f/df

//                     jacobian_temp(0,0) = 2 * (v[0] * v[0] + v[4] * v[4] + 1)*X1_Jac(0,0) - 2 * (v[0] * v[1] + v[4] * v[5] + 1)*X1_Jac(0,1);
//                     jacobian_temp(0,1)  = 2 * (v[0] * v[0] + v[4] * v[4] + 1)*X1_Jac(0,0) - 2 * (v[0] * v[2] + v[4] * v[6] + 1)*X1_Jac(0,2);
//                     jacobian_temp(0,2)  = 0;
//                     jacobian_temp(0,3)  = 0;
//                     jacobian_temp(0,4)  =  (v[0])*(v[6]*X1_Jac(0,2)-v[7]*X1_Jac(0,3))-(v[2]*X1_Jac(0,2)-v[3]*X1_Jac(0,3))*(v[4]); ;
//                     jacobian_temp(0,5)  =  (v[0])*(X1_Jac(0,2)-X1_Jac(0,3))-(v[2]*X1_Jac(0,2)-v[3]*X1_Jac(0,3));
//                     jacobian_temp(0,6)  =  (v[4])*(X1_Jac(0,2)-X1_Jac(0,3))-(v[6]*X1_Jac(0,2)-v[7]*X1_Jac(0,3));

//                     jacobian_temp(1,0)  = -2 * (v[0] * v[1] + v[4] * v[5] + 1)*X1_Jac(0,0) + 2 * (v[1] * v[1] + v[5] * v[5] + 1)*X1_Jac(0,1);
//                     jacobian_temp(1,1) = 0;
//                     jacobian_temp(1,2) = 2 * (v[1] * v[1] + v[5] * v[5] + 1)*X1_Jac(0,1) - 2 * (v[1] * v[2] + v[5] * v[6] + 1)*X1_Jac(0,2);
//                     jacobian_temp(1,3) = 0;
//                     jacobian_temp(1,4) = (-v[1])*(v[6]*X1_Jac(0,2)-v[7]*X1_Jac(0,3))-(v[2]*X1_Jac(0,2)-v[3]*X1_Jac(0,3))*(-v[5]);
//                     jacobian_temp(1,5) = (-v[1])*(X1_Jac(0,2)-X1_Jac(0,3))-(v[2]*X1_Jac(0,2)-v[3]*X1_Jac(0,3))*(-1);
//                     jacobian_temp(1,6) = (-v[5])*(X1_Jac(0,2)-X1_Jac(0,3))-(v[6]*X1_Jac(0,2)-v[7]*X1_Jac(0,3))*(-1);

//                     jacobian_temp(2,0) = 0;
//                     jacobian_temp(2,1) = -2 * (v[0] * v[2] + v[4] * v[6] + 1)*X1_Jac(0,0) + 2 * (v[2] * v[2] + v[6] * v[6] + 1)*X1_Jac(0,2);

//                     jacobian_temp(2,2) = -2 * (v[1] * v[2] + v[5] * v[6] + 1)*X1_Jac(0,1) + 2 * (v[2] * v[2] + v[6] * v[6] + 1)*X1_Jac(0,2);
//                     jacobian_temp(2,3)  = 2 * (v[2] * v[2] + v[6] * v[6] + 1)*X1_Jac(0,2) - 2 * (v[3] * v[2] + v[7] * v[6] + 1) *X1_Jac(0,3);
//                     jacobian_temp(2,4) = (v[0]*X1_Jac(0,0)-v[1]*X1_Jac(0,1))*(v[6])-(v[2])*(v[4]*X1_Jac(0,0)-v[5]*X1_Jac(0,1));
//                     jacobian_temp(2,5) = (v[0]*X1_Jac(0,0)-v[1]*X1_Jac(0,1))-(v[2])*(X1_Jac(0,0)-X1_Jac(0,1));
//                     jacobian_temp(2,6) = (v[4]*X1_Jac(0,0)-v[5]*X1_Jac(0,1))*(1)-(v[6])*(X1_Jac(0,0)-X1_Jac(0,1));

//                     jacobian_temp(3,0)  = 0;
//                     jacobian_temp(3,1)  = 0;
//                     jacobian_temp(3,2) = 0;
//                     jacobian_temp(3,3)  = 2 * (v[3] * v[3] + v[7] * v[7] + 1) * X1_Jac(0,3) -2 * (v[3] * v[2] + v[7] * v[6] + 1) *X1_Jac(0,2);
//                     jacobian_temp(3,4)  = (v[0]*X1_Jac(0,0)-v[1]*X1_Jac(0,1))*(-v[7])-(-v[3])*(v[4]*X1_Jac(0,1)-v[5]*X1_Jac(0,1));
//                     jacobian_temp(3,5)  = (v[0]*X1_Jac(0,0)-v[1]*X1_Jac(0,1))*(-1)-(-v[3])*(X1_Jac(0,0)-X1_Jac(0,1));
//                     jacobian_temp(3,6)  = (v[4]*X1_Jac(0,0)-v[5]*X1_Jac(0,1))*(-1)-(-v[7])*(X1_Jac(0,0)-X1_Jac(0,1));
        
//                     for(int i=0;i<4;i++)
//                     {
//                         for(int j=0;j<7;j++)
//                         jacobian_temp_M(i,j)=jacobian_temp(i,j)+JAC(i,j);
//                     }
        
//                     Eigen::JacobiSVD<Eigen::MatrixXd> svd_1(jacobian_temp_M, Eigen::ComputeFullU | Eigen::ComputeFullV);//M=USV*
//                     int row_invJav_M = jacobian_temp_M.rows();
//                     int col_invJav_M= jacobian_temp_M.cols();
//                     int k_invJav_M =row_invJav_M>=col_invJav_M? col_invJav_M:row_invJav_M;
//                     Eigen::MatrixXd X_invJav_M = Eigen::MatrixXd::Zero(col_invJav_M,row_invJav_M);
//                     Eigen::MatrixXd singularValues_inv_invJav_M = svd_1.singularValues();//奇异值
//                     Eigen::MatrixXd singularValues_inv_mat_invJav_M = Eigen::MatrixXd::Zero(col_invJav_M, row_invJav_M);
//                     for (long i = 0; i<k_invJav_M; ++i)
//                     {
//                         if (singularValues_inv_invJav_M(i) > pinvtoler)
//                             singularValues_inv_invJav_M(i) = 1.0 / singularValues_inv_invJav_M(i);
//                         else 
//                             singularValues_inv_invJav_M(i) = 0;
//                     }
//                     for (long i = 0; i < k_invJav_M; ++i) 
//                     {
//                         singularValues_inv_mat_invJav_M(i, i) = singularValues_inv_invJav_M(i);
//                     }
//                     X_invJav_M=(svd_1.matrixV())*(singularValues_inv_mat_invJav_M)*(svd_1.matrixU().transpose());//X=VS+U* 4*7矩阵

//                     W_Jac = X0_Jac -2*Y0_Jac*X_invJav_M;//优化之后的迭代

//                     double depth_temp[4];
//                     for(int i=0;i<4;i++)
//                         depth_temp[i]=depth[i];

//                     for(int i=0;i<4;i++)
//                         depth[i]=W_Jac(0,i);

//                     Y0_Jac(0,0) = (v[0]* v[0]+ v[4]* v[4]+ 1)*depth[0]* depth[0] - 2 * (v[0] * v[1] +v[4] *v[5] +1)*depth[0] *depth[1] + (v[1] * v[1] + v[5] * v[5] + 1)*depth[1] * depth[1] - v[8] * v[8];
//                     Y0_Jac(0,1)  = (v[0]* v[0]+ v[4]* v[4]+ 1)*depth[0]* depth[0] - 2 * (v[0] * v[2] +v[4] *v[6] +1)*depth[0] *depth[2] + (v[2] * v[2] + v[6] * v[6] + 1)*depth[2] * depth[2] - v[9] * v[9];
//                     Y0_Jac(0,2)  = (v[1] *v[1]+ v[5] *v[5]+ 1)*depth[1]* depth[1] - 2 * (v[1] * v[2] +v[5] *v[6] +1)*depth[1] *depth[2] + (v[2] * v[2] + v[6] * v[6] + 1)*depth[2] *depth[2] - v[10] * v[10];
//                     Y0_Jac(0,3)  = (v[3]* v[3]+ v[7]* v[7]+ 1)*depth[3]* depth[3] - 2 * (v[3] * v[2] +v[7] *v[6] +1)*depth[3] *depth[2] + (v[2] * v[2] + v[6] * v[6] + 1)*depth[2] *depth[2] - v[11] * v[11];
//                     Y0_Jac(0,4)  = (v[0]*depth[0]-v[1]*depth[1])*(v[6]*depth[2]-v[7]*depth[3])-(v[2]*depth[2]-v[3]*depth[3])*(v[4]*depth[0]-v[5]*depth[1]);
//                     Y0_Jac(0,5)  = (v[0]*depth[0]-v[1]*depth[1])*(depth[2]-depth[3])-(v[2]*depth[2]-v[3]*depth[3])*(depth[0]-depth[1]);
//                     Y0_Jac(0,6)  = (v[4]*depth[0]-v[5]*depth[1])*(depth[2]-depth[3])-(v[6]*depth[2]-v[7]*depth[3])*(depth[0]-depth[1]);
                
//                     Eigen::JacobiSVD<Eigen::MatrixXd> svd_2(jacobian_temp, Eigen::ComputeFullU | Eigen::ComputeFullV);//M=USV*
//                     // double  pinvtoler = 1.e-8; //tolerance
//                     int row_invJav_T = jacobian_temp.rows();
//                     int col_invJav_T= jacobian_temp.cols();
//                     int k_invJav_T =row_invJav_T>=col_invJav_T? col_invJav_T:row_invJav_T;
//                     Eigen::MatrixXd X_invJav_T = Eigen::MatrixXd::Zero(col_invJav_T,row_invJav_T);
//                     Eigen::MatrixXd singularValues_inv_invJav_T = svd_2.singularValues();//奇异值
//                     Eigen::MatrixXd singularValues_inv_mat_invJav_T = Eigen::MatrixXd::Zero(col_invJav_T, row_invJav_T);
//                     for (long i = 0; i<k_invJav_T; ++i) 
//                     {
//                         if (singularValues_inv_invJav_T(i) > pinvtoler)
//                             singularValues_inv_invJav_T(i) = 1.0 / singularValues_inv_invJav_T(i);
//                         else 
//                             singularValues_inv_invJav_T(i) = 0;
//                     }
//                     for (long i = 0; i < k_invJav_T; ++i) 
//                     {
//                         singularValues_inv_mat_invJav_T(i, i) = singularValues_inv_invJav_T(i);
//                     }
//                     X_invJav_T=(svd_2.matrixV())*(singularValues_inv_mat_invJav_T)*(svd_2.matrixU().transpose());//X=VS+U* 4*7矩阵
                
//                     Z_Jac=W_Jac-Y0_Jac*X_invJav_T;

//                     for(int i=0;i<4;i++)     
//                     {
//                         depth[i]=Z_Jac(0,i);
//                         X0_Jac(0,i)=Z_Jac(0,i);
//                         x_start[i]=Z_Jac(0,i);   
//                     }
                    
//                     if(  ((depth_temp[0]-Z_Jac(0,0))+ (depth_temp[1]-Z_Jac(0,1))+ (depth_temp[2]-Z_Jac(0,2))+ (depth_temp[3]-Z_Jac(0,3)))  < Epsilon )
//                     {
//                         break;
//                     }
//                 } while (iter<Max);

// /////////////////////////////////////////////////////////////////////////得到的深度信息//////////////////////////////////////////////////////////////////////////////
//                 depth[0]=fabs(depth[0]);
//                 depth[1]=fabs(depth[1]);
//                 depth[2]=fabs(depth[2]);
//                 depth[3]=fabs(depth[3]);
//                 // std::cout<<"深度：P1 " << depth[0] << " P2 " << depth[1] << " P3 " << depth[2] << " P4 " << depth[3] << "cm" <<std::endl;

//                 //X--与u同向--右方
//                 //Y--与v同向--下方
//                 float X1_camera=trans_P1d[0]*depth[0];
//                 float Y1_camera=trans_P1d[1]*depth[0];
//                 float X2_camera=trans_P2d[0]*depth[1];
//                 float Y2_camera=trans_P2d[1]*depth[1];
//                 float X3_camera=trans_P3d[0]*depth[2];
//                 float Y3_camera=trans_P3d[1]*depth[2];
//                 float X4_camera=trans_P4d[0]*depth[3];
//                 float Y4_camera=trans_P4d[1]*depth[3];

// //////////////////// ////////////////////////////////////////////////////////相机坐标/////////////////////////////////////////////////////////////////////////////////////
//                 //相机正前方是y轴方向，右边是x轴方向，正上方是z轴方向 Pc（Xc, Yc, Zc）
//                 //1、深度是沿着y轴方向伸展的，因此深度depth[4]要放在Yc的位置上，Yc = depth[]
//                 //2、在图像中y轴(v轴)方向是朝下的，而在相机坐标系中Z轴是朝正上方的，因此Zc = - Y_camera
//                 // Pc1 << X1_camera, depth[0], -Y1_camera;//左上顶点在相机坐标系下的坐标
//                 // Pc2 << X2_camera, depth[1], -Y2_camera;//右上顶点在相机坐标系下的坐标
//                 // Pc3 << X3_camera, depth[2], -Y3_camera;//左下顶点在相机坐标系下的坐标
//                 // Pc4 << X4_camera, depth[3], -Y4_camera;//右下顶点在相机坐标系下的坐标

//                 //中心点的相机坐标，Yc上面加余量是因为相机与飞机中心不是重合的，所以要补偿
//                 // u-右, v-下
//                 //1. X-右, Y-前, Z-上
//                 // Pc_central << (X1_camera+X4_camera+X2_camera+X3_camera)/4, 
//                 //                              (depth[1]+depth[2]+depth[0]+depth[3])/4, 
//                 //                              (-Y2_camera-Y3_camera-Y1_camera-Y4_camera)/4;
//                 //2. X-前, Y-左, Z-上

//                 // Pc_central << (depth[1]+depth[2]+depth[0]+depth[3])/4, 
//                 //                             -(X1_camera+X4_camera+X2_camera+X3_camera)/4, 
//                 //                             -(Y2_camera+Y3_camera+Y1_camera+Y4_camera)/4;

//                 Pc_central << (depth[1]+depth[2]+depth[0]+depth[3])/4, //正前方
//                                             -(X1_camera+X4_camera+X2_camera+X3_camera)/4,//左方
//                                             -(Y2_camera+Y3_camera+Y1_camera+Y4_camera)/4;//正上方

// ///////////////////////////////////////////////////////////////////解算世界坐标(回调函数中进行)//////////////////////////////////////////////////////////////////////
//                 std::cout << "目标中心的相机坐标Pc:  (" << Pc_central[0] << ",  " << Pc_central[1] << ",  " << Pc_central[2] << ")cm\n\n"; 
            
//                 // std::cout << "目标中心的世界坐标 Pw: (" << P_worldcentral[0] << ", " << P_worldcentral[1] << ", " << P_worldcentral[2] << ")cm\n\n";   
//             }

//             else{
//                 isDetected = false;
//                 cout << "未检测到边框!" << endl;
//             }

//             //初始化MonoCamera::object类型的消息
//             MonoCamera::object obj_msg;
//             obj_msg.isDetected = isDetected;
//             obj_msg.object_number = number;
//             obj_msg.object_position.x = Pc_central[0];
//             obj_msg.object_position.y = Pc_central[1];
//             obj_msg.object_position.z = Pc_central[2];

//             //发布器发布消息
//             pub.publish(obj_msg);
//             loop_rate.sleep();

//             //保存数据
//             writeData();  

//             //imshow("image_raw", image_raw);
//             cv::waitKey(1); 

//             //保存视频
//             writer << image_raw;
        }
    }
    catch(std::exception& e){
        std::cout<<"发生错误"<<std::endl;
        std::cout << e.what() << std::endl;  
    }
    cap.release();
    writer.release();
    return 0;
}
