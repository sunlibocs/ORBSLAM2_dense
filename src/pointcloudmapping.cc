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


void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    unique_lock<mutex> lock(keyframeMutex);
    //cout << "kf push_back" << endl;
    keyframes.push_back( kf );
    //cout << "color push_back" << endl;
    colorImgs.push_back( color.clone() );
    // cout << "depth push_back" << endl;

    // //在通知之前先把深度估计出来 TODO 删掉没啥用的
    // update_KeyFrameDepth(kf, color);

    // cout << "depth push_back" << endl;

    depthImgs.push_back( depth.clone() );
    
    keyFrameUpdated.notify_one();
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    PointCloud::Ptr tmp( new PointCloud() );
    // point cloud is null ptr
    for ( int m=0; m<depth.rows; m+=3 )
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d>10)
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
    //pose是求得世界的点转到kf的坐标系下的坐标
    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;
    
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
        
        for ( size_t i=lastKeyframeSize; i<N ; i++ )
        {   
            //这里直接是把每一次点云进行累加,只是做了一个可视化，并没有对点云进行回环优化
            PointCloud::Ptr p = generatePointCloud( keyframes[i], colorImgs[i], depthImgs[i] );
            *globalMap += *p;
        }
        PointCloud::Ptr tmp(new PointCloud());
        voxel.setInputCloud( globalMap );
        voxel.filter( *tmp );
        globalMap->swap( *tmp );
        viewer.showCloud( globalMap );
        cout<<"show global map, size="<<globalMap->points.size()<<endl;
        lastKeyframeSize = N;
    }
}

