/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>

#include<opencv2/core/core.hpp>

#include"System.h"

using namespace std;

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

int main(int argc, char **argv)
{
    cout << "begin main" << endl;
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    cout << "read image" << endl;
    LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    cout << "read image end" << endl;
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;


    cv::Mat T1w;

    // Main loop
    cv::Mat im;
    cv::Mat img_pre;



    //读取旋转和平移
    string pose_path = "/home/slbs/Downloads/data_odometry_poses/dataset/poses/03.txt";
    ifstream pose_file;//创建文件流对象
    pose_file.open(pose_path);

    for(int ni=0; ni<nImages; ni++)
    {
        cout << "begin to process " << ni << endl;


        // Read image from file
        im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        //return mCurrentFrame.mTcw.clone();

        cv::Mat Tcw = SLAM.TrackMonocular(im,tframe);



        // //读取旋转和平移 相机到世界的
        // cv::Mat Twc = cv::Mat::zeros(3, 4, CV_32FC1);
        // for (int i = 0; i < 3; i++){
        //     for (int j = 0; j < 4; j++){
        //         pose_file >> Twc.at<float>(i, j);
        //     }
        // }

        // //求出世界到相机的
        // cv::Mat Rcw = Twc.rowRange(0,3).colRange(0,3).t();
        // cv::Mat tcw = -Rcw*Twc.rowRange(0,3).col(3);

        // cv::Mat Tcw = cv::Mat::zeros(3, 4, CV_32FC1);
        // Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
        // tcw.copyTo(Tcw.rowRange(0,3).col(3));



        //强行认为100张以后已经完成了初始化
        if(ni == 15){
            //T1w = Tcw;
            Tcw.copyTo(T1w);
            img_pre= im;
        }

        if(ni > 15){


            cv::Mat T2w = Tcw;
            
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
            cv::Mat K = SLAM.K;
            cv::Mat P2(3,4,CV_32F,cv::Scalar(0));
            K.copyTo(P2.rowRange(0,3).colRange(0,3));


            cv::Mat P1(3,4,CV_32F);
            R12.copyTo(P1.rowRange(0,3).colRange(0,3));
            t12.copyTo(P1.rowRange(0,3).col(3));
            P1 = K*P1;

            cout << "T1w" << endl << T1w << endl << "T2w" << endl << T2w << endl;

            cout << "****R12*** " << endl << R12 << endl << "***t12*** " << endl << t12 << endl << "K " << endl << K << endl;
         

            //读取矩阵

            stringstream ss;
            int num = ni;
            ss << setfill('0') << setw(6) << num;
            //vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
            string x_str = "/media/slbs/DATA/03/image_2/res/x_" + ss.str() + ".txt";
            string y_str = "/media/slbs/DATA/03/image_2/res/y_" + ss.str() + ".txt";
      

            cv::Mat depthMap = cv::Mat::zeros(im.rows, im.cols, CV_32FC1);

            cv::Mat depthMap_show = cv::Mat::zeros(im.rows, im.cols, CV_8UC1);


            // ifstream testfile;
            
            // //x_str = "/home/slbs/Desktop/times.txt";

            // testfile.open(x_str.c_str());
            
            // while(!testfile.eof())
            // {
            //     string s;
            //     getline(testfile,s);
            //     cout << s << " xxxxx" << endl;
            //     
            //     if(!s.empty())
            //     {
            //         stringstream ss;
            //         ss << s;
            //         double t;
            //         ss >> t;
            //         cout << t << endl;
            //        
            //     }
            // }


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
                    
                    // file_x >> x_Data.at<float>(i, j);
                    // file_y >> y_Data.at<float>(i, j);

                    // offset_x = x_Data.at<float>(i, j);
                    // offset_y = y_Data.at<float>(i, j);
                    
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
                    //生成深度图
                    // cout << "set z -depth: " << z << endl;

                    // 使用ORB
                    z = z * 10;

                    //if(z<0) z = 0;
                    if(z<0) z = 255;
                    if(z>100) z = 100;

                    depthMap.at<float>(i,j) = z;

                    unsigned char zz = z;
                    depthMap_show.at<uchar>(i,j) = zz;

                    //cout << "z=" << z << endl;

                    //if(i < 50 || j < 50 || j % 50 == 0) continue;
                    //if (i == 0 || j == 0) continue;
                    if(i % 60 == 0 && j % 60 == 0) {

                        // cout << "location" << endl;
                        // cout << kp1.pt.x << "," << kp1.pt.y << endl;
                        // cout << kp2.pt.x << "," << kp2.pt.y << endl<< "offset" << endl;
                        // cout << offset_x << "," << offset_y << endl;


                        cv::circle(img_pre, cv::Point(kp1.pt.x, kp1.pt.y), 3, cv::Scalar(0, 0, 255));
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


                    // float invZ1 = 1.0/p3dC1.at<float>(2);
                    // im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
                    // im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

                }
            }

            file_x.close();
            file_y.close();

            
            double min; 
            double max; 
           

            cv::minMaxIdx(depthMap, &min, &max); 

            cout << "min: " << min << "max: " << max << endl;
            cv::Mat color_depth, color_depth2;
            cv::Rect roi(0, 0, depthMap_show.cols - 0, depthMap_show.rows - 0);
            //Create the cv::Mat with the ROI you need, where "image" is the cv::Mat you want to extract the ROI from
            cv::Mat depth_roi = depthMap_show(roi);

            cv::applyColorMap(depth_roi, color_depth, 2);

            //cv::imshow("depthMap_show", color_depth); 


            depthMap -= min; 
            cv::Mat adjMap; 
            cv::convertScaleAbs(depthMap, adjMap, 255.0/double(max-min)); 
            //cv::imshow("Out", adjMap); 

            cv::Mat depth_roi2 = adjMap(roi);
            cv::applyColorMap(depth_roi2, color_depth2, 2);

            cv::imshow("depthMap_show2", color_depth2); 

            cv::imshow("img_curtxt", im_cur);
            cv::imshow("img_pre", img_pre);
            cv::imshow("img_cur", im);
              
            cv::waitKey(10);
            getchar();
            // // Camera 1 Projection Matrix K[I|0]
            // cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
            // K.copyTo(P1.rowRange(0,3).colRange(0,3));

            // // Camera 2 Projection Matrix K[R|t]
            // cv::Mat P2(3,4,CV_32F);
            // R.copyTo(P2.rowRange(0,3).colRange(0,3));
            // t.copyTo(P2.rowRange(0,3).col(3));
            // P2 = K*P2;

            //Triangulate(kp1,kp2,P1,P2,p3dC1);

            

            //赋值给之前的帧
            img_pre = im;
            T1w = Tcw;
        }


#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");    

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_2/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
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
