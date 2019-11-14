/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include "pointcloudmapping.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include "Converter.h"



//add
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

#include <rmd/depthmap.h>
#include <rmd/check_cuda_device.cuh> 


// #include "../include/rmd/depthmap.h"
// #include "../include/rmd/check_cuda_device.cuh"

#include "../include/dataset.h"


#include<iostream>

PointCloudMapping::PointCloudMapping(double resolution_)
{
    this->resolution = resolution_;
    voxel.setLeafSize( resolution, resolution, resolution);
    globalMap = boost::make_shared< PointCloud >( );
    
    viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lock(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

void PointCloudMapping::insertFrame(cv::Mat color, cv::Mat mTcw){
    
    //cout << "insert frame1" << endl;
    //assert colorImgs_previous.size() == mTcw_vec.size();
    //保存下来用于更新深度 TODO 可能太多撑爆内存 这里强行保存最近的五十张
    if(colorImgs_previous.size() > 10){
        colorImgs_previous.erase(colorImgs_previous.begin());
        mTcw_vec.erase(mTcw_vec.begin());
    }

    colorImgs_previous.push_back(color);
    mTcw_vec.push_back(mTcw);
    //cout << "insert frame2" << endl;
}

//使用remode求出keyframe的depth
void PointCloudMapping::update_KeyFrameDepth(KeyFrame* kf, cv::Mat& color){
    
    if(!rmd::checkCudaDevice()){
        std::cerr << "No Cuda Device!" << std::endl;
        return;
    }

    rmd::PinholeCamera cam(520.908620, 521.007327, 325.141442, 249.701764);
    
    const size_t width  = color.cols;
    const size_t height = color.rows;

    //bool first_img = true;
    rmd::Depthmap depthmap(width, height, cam.fx, cam.cx, cam.fy, cam.cy);

    // store the timings
    // update
    std::vector<double> update_time;
    
    //SE3(Type qw, Type qx, Type qy, Type qz, Type tx, Type ty, Type tz)
    
    double min_depth = 0.6, max_depth=0.6;

    //这里是把相机的点转向世界的矩阵
    cv::Mat Rwc = (kf->GetRotation()).t();
    cv::Mat twc = -Rwc*(kf->GetTranslation());
    vector<float> q = Converter::toQuaternion(Rwc);

 
    rmd::SE3<float> T_world_curr(q[3], q[0], q[1], q[2], twc.at<float>(0), twc.at<float>(1), twc.at<float>(2));
    
    
    cout << "T_world_curr: " << endl << T_world_curr<< endl;
    cout << twc.at<float>(0) << "//" << twc.at<float>(1)  << "//" << twc.at<float>(2)  << endl;

    depthmap.setReferenceImage(color, T_world_curr.inv(), min_depth, max_depth);

    cv::imshow("keyframeImge", color);
    cv::waitKey(10);
    //for(int i = colorImgs_previous.size() - 1; i > 0; i--)
    for(int i = 0; i < colorImgs_previous.size() - 2; i++)
    {
        //位姿 TODO 
        cv::Mat Tcw = mTcw_vec[i];
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);
        vector<float> q = Converter::toQuaternion(Rwc);

        rmd::SE3<float> T_world_curr(q[3], q[0], q[1], q[2], twc.at<float>(0), twc.at<float>(1), twc.at<float>(2));
        cv::Mat img = colorImgs_previous[i];
        double t = (double)cv::getTickCount();
        depthmap.update(img, T_world_curr.inv());
        t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        printf("\nUPDATE execution time: %f seconds.\n", t);
        update_time.push_back(t);

        // 在这里输出一下深度图
        depthmap.downloadDenoisedDepthmap(0.5f, 6);
        cv::Mat denoised_result = depthmap.getDepthmap();
        cv::Mat colored_denoised = rmd::Depthmap::scaleMat(denoised_result);
        cv::imshow("denoised_depth", colored_denoised);
        cv::imshow("current", img);
        cv::waitKey(10);
    }
}


void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}


//使用光流计算图像深度
void PointCloudMapping::update_KeyFrameDepth_UsingFlow(KeyFrame* kf, Frame pre_frame, cv::Mat& im, cv::Mat& depthMap){
    //得到图片编号
    long unsigned int frameId = kf -> mnFrameId;

        //已经执行到了第三张
        if(frameId > 3){
            cout << "frameId = " << frameId << endl;
            //得到位姿
            cv::Mat T1w = pre_frame.mTcw;
            cv::Mat T2w = kf -> GetPose();
            
            //求解R t 2 到1的
            //世界点转换到相机
            cv::Mat R1w = T1w.rowRange(0,3).colRange(0,3);
            cv::Mat t1w = T1w.rowRange(0,3).col(3);

            //相机点到世界
            cv::Mat Rw1 = R1w.t();
            cv::Mat tw1 = -R1w.t()*t1w;

            //第二个相机位姿
            cv::Mat R2w = T2w.rowRange(0,3).colRange(0,3);
            cv::Mat t2w = T2w.rowRange(0,3).col(3);
            cv::Mat Rw2 = R2w.t();
            cv::Mat tw2 = -R2w.t()*t2w;


            //求解第二个相机到第一个的 R t
            cv::Mat R12 = R1w * Rw2;
            cv::Mat t12 = R1w * tw2 + t1w;

            // Camera 1 Projection Matrix K[I|0]
            cv::Mat K = kf->mK;
            cv::Mat P2(3,4,CV_32F,cv::Scalar(0));
            K.copyTo(P2.rowRange(0,3).colRange(0,3));


            cv::Mat P1(3,4,CV_32F);
            R12.copyTo(P1.rowRange(0,3).colRange(0,3));
            t12.copyTo(P1.rowRange(0,3).col(3));
            P1 = K*P1;

            //读取矩阵
            stringstream ss;
            //int num = ni;
            ss << setfill('0') << setw(6) << frameId;
            //vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
            string x_str = "/media/slbs/DATA/03/image_2/res/x_" + ss.str() + ".txt";
            string y_str = "/media/slbs/DATA/03/image_2/res/y_" + ss.str() + ".txt";


            //cout << "x_str = " << x_str << endl;

            //cv::Mat depthMap = cv::Mat::zeros(im.rows, im.cols, CV_32FC1);
            depthMap = cv::Mat::zeros(im.rows, im.cols, CV_32FC1);

            ifstream file_x;//创建文件流对象
            file_x.open(x_str);

            if(!file_x) 
            { 
                cout <<"fail to open file x" <<endl; 
                //system("pause");
                
            } 

            ifstream file_y;//创建文件流对象
            file_y.open(y_str);
            
            cv::Mat x_Data = cv::Mat::zeros(im.rows, im.cols, CV_32FC1);//创建Mat类矩阵，定义初始化值全部是0，矩阵大小和txt一致
            cv::Mat y_Data = cv::Mat::zeros(im.rows, im.cols, CV_32FC1);//同理

            cv:: Mat im_cur = im.clone();

            //将txt文件数据写入到Data矩阵中
            for (int i = 0; i < im.rows; i++){
                //列
                for (int j = 0; j < im.cols; j++){
                        
                    float x2 = j;
                    float y2 = i;
                    cv::Mat p3dC1;
                    float offset_x, offset_y;

                    file_x >> offset_x;
                    file_y >> offset_y;
                    
                    cv::KeyPoint kp2(x2, y2, CV_32FC1);

                    float x1 = x2 + offset_x;
                    float y1 = y2 + offset_y;

                    cv::KeyPoint kp1(x1, y1, CV_32FC1);
                    //三角化后得到点的深度
                    Triangulate(kp2,kp1,P2,P1,p3dC1);

                    float z = p3dC1.at<float>(2);


                    //坏的点都用255表示
                    if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
                    {
                        z = 255;
                    }

                    //检查视察和重投影误差 TODO *************************
                     // Check parallax

                    //相机2的中心位置
                    cv::Mat O2 = cv::Mat::zeros(3,1,CV_32F);
                    //相机1光心在2坐标系下的位置
                    cv::Mat O1 = -R12.t()*t12;

                    //点到相机2光心的连线
                    cv::Mat normal2 = p3dC1 - O2;
                    float dist2 = cv::norm(normal2);

                    //点到相机1光心的连线
                    cv::Mat normal1 = p3dC1 - O1;
                    float dist1 = cv::norm(normal1);

                    //夹角
                    float cosParallax = normal1.dot(normal2)/(dist1*dist2);

                    // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
                    //p3dC1 为在相机2坐标系下的坐标
                    if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
                        z = 255;

                    // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
                    //在相机1中的坐标
                    cv::Mat p3dC11 = R12*p3dC1+t12;

                    if(p3dC11.at<float>(2)<=0 && cosParallax<0.99998)
                        z = 255;

                    //if(cosParallax>0.99998) z = 255;

                    // // Check reprojection error in reference image


                    float fx = 721.5377;
                    float fy = 721.5377;
                    float cx = 609.5593;
                    float cy = 172.854;
                    float th2  = 4;

                    float im1x, im1y;
                    float invZ1 = 1.0/p3dC1.at<float>(2);
                    im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
                    im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

                    float squareError1 = (im1x-kp2.pt.x)*(im1x-kp2.pt.x)+(im1y-kp2.pt.y)*(im1y-kp2.pt.y);

                    if(squareError1>th2)
                        z = 255;

                    // Check reprojection error in second image
                    float im2x, im2y;
                    float invZ2 = 1.0/p3dC11.at<float>(2);
                    im2x = fx*p3dC11.at<float>(0)*invZ2+cx;
                    im2y = fy*p3dC11.at<float>(1)*invZ2+cy;

                    float squareError2 = (im2x-kp1.pt.x)*(im2x-kp1.pt.x)+(im2y-kp1.pt.y)*(im2y-kp1.pt.y);

                    if(squareError2>th2)
                        z = 255;
                    
                    /***************************************************************** */



                    if(x1 < 0 || x1 > im.cols || y1 < 0 || y1 > im.rows) z = 255;
                   

                    // 使用ORB
                    //z = z * 10;

                    //if(z<0) z = 0;
                    if(z<0) z = 255;
                    if(z>100) z = 100;

                    depthMap.at<float>(i,j) = z;

                    // unsigned char zz = z;
                    // depthMap_show.at<uchar>(i,j) = zz;

                    //cout << "z=" << z << endl;

                    //if(i < 50 || j < 50 || j % 50 == 0) continue;
                    //if (i == 0 || j == 0) continue;
                    if(i % 60 == 0 && j % 60 == 0) {

                        // cout << "location" << endl;
                        // cout << kp1.pt.x << "," << kp1.pt.y << endl;
                        // cout << kp2.pt.x << "," << kp2.pt.y << endl<< "offset" << endl;
                        // cout << offset_x << "," << offset_y << endl;


                        //cv::circle(img_pre, cv::Point(kp1.pt.x, kp1.pt.y), 3, cv::Scalar(0, 0, 255));
                        // cv::imshow("img_pre", img_pre);
                        cv::circle(im_cur, cv::Point(kp2.pt.x, kp2.pt.y), 3, cv::Scalar(0, 0, 255));
                        
                        //cv::Mat im_cur_txt = im_cur.clone();
                        ostringstream buffer;
                        buffer << z;
                        string str_z = buffer.str();
                        //加上字符的起始点
                        cv::putText(im_cur, str_z, cv::Point(kp2.pt.x, kp2.pt.y), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 255, 0), 0.8, CV_AA);


                        //cv::imshow("im_cur_txt", im_cur);

                        // cv::waitKey(10);


                    }


                }
            }

            file_x.close();
            file_y.close();


            cv::Mat depthMap_show = depthMap.clone();

            
            double min; 
            double max; 
           

            cv::minMaxIdx(depthMap_show, &min, &max); 

            cout << "min: " << min << "max: " << max << endl;
            //cv::Rect roi(0, 0, depthMap_show.cols - 0, depthMap_show.rows - 0);


            depthMap_show -= min; 
            cv::Mat adjMap; 
            cv::convertScaleAbs(depthMap_show, adjMap, 255.0/double(max-min)); 
            //cv::imshow("Out", adjMap); 

            //cv::Mat depth_roi2 = adjMap(roi);
            cv::Mat color_depth2;
            cv::applyColorMap(adjMap, color_depth2, 2);

            cv::imshow("depthMap_show", color_depth2); 

            cv::imshow("img_curtxt", im_cur);
            //cv::imshow("img_pre", img_pre);
            cv::imshow("img_cur", im);
              
            cv::waitKey(10);
        }


    
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, Frame pre_frame){
    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    
    unique_lock<mutex> lock(keyframeMutex);
    //cout << "kf push_back" << endl;
    keyframes.push_back( kf );
    //cout << "color push_back" << endl;
    colorImgs.push_back( color.clone() );
    // cout << "depth push_back" << endl;

    // //在通知之前先把深度估计出来 TODO 删掉没啥用的 mnFrameId
    // update_KeyFrameDepth(kf, color);
    
    update_KeyFrameDepth_UsingFlow(kf, pre_frame, color, depth);

    depthImgs.push_back(depth.clone() );
    
    keyFrameUpdated.notify_one();
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, unsigned int N)
{
    PointCloud::Ptr tmp( new PointCloud() );
    // point cloud is null ptr
    for ( int m=0; m<depth.rows; m+=3 )
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {
            float d = depth.ptr<float>(m)[n];
            //如果大于10
            if (d < 0.01 || d>5)
                continue;
            PointT p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;
            
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];
                
            tmp->points.push_back(p);
        }
    }
    // //pose是求得世界的点转到kf的坐标系下的坐标 ******原来的以世界坐标显示*************
    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;
    
    //显示可视化到当前最新的关键帧 Tcw Tcn   Tnw*Twc
    // Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( keyframes[N-1]->GetPose() * kf->GetPoseInverse() );
    // PointCloud::Ptr cloud(new PointCloud);
    // pcl::transformPointCloud( *tmp, *cloud, T.matrix());
    // cloud->is_dense = false;


    
    //cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}


void PointCloudMapping::viewer()
{
    pcl::visualization::CloudViewer viewer("viewer");
    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }
        
        // keyframe is updated 
        size_t N=0;
        {
            unique_lock<mutex> lock( keyframeMutex );
            N = keyframes.size();
        }
        
        // TODO
        //globalMap = boost::make_shared< PointCloud >( );

        for ( size_t i=lastKeyframeSize; i<N ; i++ )
        //for ( size_t i=lastKeyframeSize; i<N ; i++ )
        {   
            //这里直接是把每一次点云进行累加,只是做了一个可视化，并没有对点云进行回环优化
            PointCloud::Ptr p = generatePointCloud( keyframes[i], colorImgs[i], depthImgs[i], N);
            *globalMap += *p;
        }

        // PointCloud::Ptr tmp(new PointCloud());
        // voxel.setInputCloud( globalMap );
        // voxel.filter( *tmp );
        // globalMap->swap( *tmp );
        // viewer.showCloud( globalMap );
        // cout<<"show global map, size="<<globalMap->points.size()<<endl;
        // lastKeyframeSize = N;


        /* *转换到当前坐标系下 进行显示 TODO**/
        //原来都是初始的世界坐标下位置,直接乘以当前的Tcw,转换到当前
        Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( keyframes[N-1]->GetPose());
        PointCloud::Ptr globalMap_now_view(new PointCloud);
        pcl::transformPointCloud( *globalMap, *globalMap_now_view, T.matrix());

        PointCloud::Ptr tmp(new PointCloud());
        voxel.setInputCloud( globalMap_now_view );
        voxel.filter( *tmp );
        globalMap_now_view->swap( *tmp );
        viewer.showCloud( globalMap_now_view );
        cout<<"show global map, size="<<globalMap_now_view->points.size()<<endl;
        lastKeyframeSize = N;

    }
}

