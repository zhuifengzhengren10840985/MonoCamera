包名:MonoCamera
节点名:MonoCamera
launch文件名:findObject.launch
运行方式:rosrun MonoCamera MonoCamera
或者roslaunch MonoCamera findObject.launch
# MonoCamera

ObjPos.txt文件中记录的是目标数字，目标在相机坐标系下的位置Pc(Xc,Yc,Zc)
roi.jpg为数字区域的局部图像，尺寸为50x50
video.avi为对相机拍摄的原始图像做好标记后的图像

数字识别部分
轮廓匹配用的是image_dilate文件夹下的模板图像（最新版采用的是轮廓匹配）
模板匹配用的是template文件下的模板图像

