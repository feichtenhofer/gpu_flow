//************************************************************************
// compute_flow.cpp
// Computes OpenCV GPU Brox et al. [1] and Zach et al. [2] TVL1 Optical Flow
// Dependencies: OpenCV and Qt5 for iterating (sub)directories
// Author: Christoph Feichtenhofer
// Institution: Graz University of Technology
// Email: feichtenhofer@tugraz
// Date: Nov. 2015
// [1] T. Brox, A. Bruhn, N. Papenberg, J. Weickert. High accuracy optical flow estimation based on a theory for warping. ECCV 2004.
// [2] C. Zach, T. Pock, H. Bischof: A duality based approach for realtime TV-L 1 optical flow. DAGM 2007.
//************************************************************************

#define N_CHAR 500
#define WRITEOUT_IMGS 1

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <time.h>
#include <sstream>

#include <QDirIterator>
#include <QFileInfo>
#include <QString>

#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

// These are required for warp flow
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
// Some surf stuff in this one below
#include "opencv2/nonfree/features2d.hpp"
//#include "opencv2/cudaarithm.hpp"
/*#include "opencv2/cudaoptflow.hpp"*
#include "opencv2/cudacodec.hpp"*/

#include <dirent.h>
//#include "warp_flow.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

float MIN_SZ = 256;
float OUT_SZ = 256;

bool clipFlow = true; // clips flow to [-20 20]
bool resize_img = true;

// These are default paths if nothing is passed
std::string vid_path = "/home/pedro/gpu_flow/avis/";
std::string out_path	= "/home/pedro/gpu_flow/tvl1_flow/";
std::string out_path_jpeg	= "/home/pedro/gpu_flow/rgb/";

bool createOutDirs = true;

/* THESE ARE MY PARAMS, NOT FEICHENHOFER'S */

bool debug = false;
bool rgb = false;
bool bins = false;

// Global variables for gpu::BroxOpticalFlow
const float alpha_ = 0.197;
const float gamma_ = 50;
const float scale_factor_ = 0.8;
const int inner_iterations_ = 10;
const int outer_iterations_ = 77;
const int solver_iterations_ = 10;

const bool warp = true;

inline void initializeMats(const Mat& frame,
                           Mat& capture_image, Mat& capture_gray,
                           Mat& prev_image, Mat& prev_gray){
    capture_image.create(frame.size(), CV_8UC3);
    capture_gray.create(frame.size(), CV_8UC1);

    prev_image.create(frame.size(), CV_8UC3);
    prev_gray.create(frame.size(), CV_8UC1);
}

/*

cv::Mat windowedMatchingMask( const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
                          float maxDeltaX, float maxDeltaY )
{
  if( keypoints1.empty() || keypoints2.empty() )
    return cv::Mat();

  int n1 = (int)keypoints1.size(), n2 = (int)keypoints2.size();
  cv::Mat mask( n1, n2, CV_8UC1 );
  for( int i = 0; i < n1; i++ )
    {
      for( int j = 0; j < n2; j++ )
        {
          cv::Point2f diff = keypoints2[j].pt - keypoints1[i].pt;
          mask.at<uchar>(i, j) = std::abs(diff.x) < maxDeltaX && std::abs(diff.y) < maxDeltaY;
        }
    }
  return mask;
}
*/


void MyWarpPerspective(Mat& prev_src, Mat& src, Mat& dst, Mat& M0, int flags=INTER_LINEAR,
                       int borderType=BORDER_CONSTANT, const Scalar& borderValue=Scalar())
{
    int width = src.cols;
    int height = src.rows;
    dst.create( height, width, CV_8UC1 );

    Mat mask = Mat::zeros(height, width, CV_8UC1);
    const int margin = 5;

    const int BLOCK_SZ = 32;
    short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];

    int interpolation = flags & INTER_MAX;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    double M[9];
    Mat matM(3, 3, CV_64F, M);
    M0.convertTo(matM, matM.type());
    if( !(flags & WARP_INVERSE_MAP) )
        invert(matM, matM);

    int x, y, x1, y1;

    int bh0 = min(BLOCK_SZ/2, height);
    int bw0 = min(BLOCK_SZ*BLOCK_SZ/bh0, width);
    bh0 = min(BLOCK_SZ*BLOCK_SZ/bw0, height);

    for( y = 0; y < height; y += bh0 ) {
        for( x = 0; x < width; x += bw0 ) {
            int bw = min( bw0, width - x);
            int bh = min( bh0, height - y);

            Mat _XY(bh, bw, CV_16SC2, XY);
            Mat matA;
            Mat dpart(dst, Rect(x, y, bw, bh));

            for( y1 = 0; y1 < bh; y1++ ) {

                short* xy = XY + y1*bw*2;
                double X0 = M[0]*x + M[1]*(y + y1) + M[2];
                double Y0 = M[3]*x + M[4]*(y + y1) + M[5];
                double W0 = M[6]*x + M[7]*(y + y1) + M[8];
                short* alpha = A + y1*bw;

                for( x1 = 0; x1 < bw; x1++ ) {

                    double W = W0 + M[6]*x1;
                    W = W ? INTER_TAB_SIZE/W : 0;
                    double fX = max((double)INT_MIN, min((double)INT_MAX, (X0 + M[0]*x1)*W));
                    double fY = max((double)INT_MIN, min((double)INT_MAX, (Y0 + M[3]*x1)*W));

                    double _X = fX/double(INTER_TAB_SIZE);
                    double _Y = fY/double(INTER_TAB_SIZE);

                    if( _X > margin && _X < width-1-margin && _Y > margin && _Y < height-1-margin )
                        mask.at<uchar>(y+y1, x+x1) = 1;

                    int X = saturate_cast<int>(fX);
                    int Y = saturate_cast<int>(fY);

                    xy[x1*2] = saturate_cast<short>(X >> INTER_BITS);
                    xy[x1*2+1] = saturate_cast<short>(Y >> INTER_BITS);
                    alpha[x1] = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (X & (INTER_TAB_SIZE-1)));
                }
            }

            Mat _matA(bh, bw, CV_16U, A);
            remap( src, dpart, _XY, _matA, interpolation, borderType, borderValue );
        }
    }

    for( y = 0; y < height; y++ ) {
        const uchar* m = mask.ptr<uchar>(y);
        const uchar* s = prev_src.ptr<uchar>(y);
        uchar* d = dst.ptr<uchar>(y);
        for( x = 0; x < width; x++ ) {
            if(m[x] == 0)
                d[x] = s[x];
        }
    }
}

void ComputeMatch(const std::vector<KeyPoint>& prev_kpts, const std::vector<KeyPoint>& kpts,
                  const Mat& prev_desc, const Mat& desc, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts)
{
    prev_pts.clear();
    pts.clear();

    if(prev_kpts.size() == 0 || kpts.size() == 0)
        return;

    Mat mask = windowedMatchingMask(kpts, prev_kpts, 25, 25);

	//-- Step 3: Matching descriptor vectors with a brute force matcher
    BFMatcher desc_matcher(NORM_L2);
    std::vector<DMatch> matches;
    desc_matcher.match(desc, prev_desc, matches, mask);

    prev_pts.reserve(matches.size());
    pts.reserve(matches.size());

    for(size_t i = 0; i < matches.size(); i++) {
        const DMatch& dmatch = matches[i];
        // get the point pairs that are successfully matched
        prev_pts.push_back(prev_kpts[dmatch.trainIdx].pt);
        pts.push_back(kpts[dmatch.queryIdx].pt);
    }

    return;
}

void MergeMatch(const std::vector<Point2f>& prev_pts1, const std::vector<Point2f>& pts1,
                const std::vector<Point2f>& prev_pts2, const std::vector<Point2f>& pts2,
                std::vector<Point2f>& prev_pts_all, std::vector<Point2f>& pts_all)
{
    prev_pts_all.clear();
    prev_pts_all.reserve(prev_pts1.size() + prev_pts2.size());

    pts_all.clear();
    pts_all.reserve(pts1.size() + pts2.size());

    for(size_t i = 0; i < prev_pts1.size(); i++) {
        prev_pts_all.push_back(prev_pts1[i]);
        pts_all.push_back(pts1[i]);
    }

    for(size_t i = 0; i < prev_pts2.size(); i++) {
        prev_pts_all.push_back(prev_pts2[i]);
        pts_all.push_back(pts2[i]);
    }

    return;
}

void MatchFromFlow_copy(const Mat& prev_grey, const Mat& flow_x, const Mat& flow_y, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts, const Mat& mask)
{
    int width = prev_grey.cols;
    int height = prev_grey.rows;
    prev_pts.clear();
    pts.clear();

    const int MAX_COUNT = 1000;
    goodFeaturesToTrack(prev_grey, prev_pts, MAX_COUNT, 0.001, 3, mask);

    if(prev_pts.size() == 0)
        return;

    for(unsigned int i = 0; i < prev_pts.size(); i++) {
        int x = std::min<int>(std::max<int>(cvRound(prev_pts[i].x), 0), width-1);
        int y = std::min<int>(std::max<int>(cvRound(prev_pts[i].y), 0), height-1);

        const float* f_x = flow_x.ptr<float>(y);
        const float* f_y = flow_y.ptr<float>(y);
        pts.push_back(Point2f(x+f_x[x], y+f_y[y]));
    }
}

void converFlowMat(Mat& flowIn, Mat& flowOut,float min_range_, float max_range_)
{
	float value = 0.0f;
	for(int i = 0; i < flowIn.rows; i++)
	{
		float* Di = flowIn.ptr<float>(i);
		char* Ii = flowOut.ptr<char>(i);
		for(int j = 0; j < flowIn.cols; j++)
		{
			value = (Di[j]-min_range_)/(max_range_-min_range_);

			value *= 255;
			value = cvRound(value);

			Ii[j] = (char) value;
		}
	}
}

static void convertFlowToImage(const Mat &flowIn, Mat &flowOut,
		float lowerBound, float higherBound) {
	#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
	for (int i = 0; i < flowIn.rows; ++i) {
		for (int j = 0; j < flowIn.cols; ++j) {
			float x = flowIn.at<float>(i,j);
			flowOut.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
		}
	}
	#undef CAST
}

int main( int argc, char *argv[] )
{
    GpuMat frame0GPU, frame1GPU, uGPU, vGPU;
    Mat frame0_rgb_, frame1_rgb_, frame0_rgb, frame1_rgb, frame0, frame1, rgb_out;
    Mat frame0_32, frame1_32, imgU, imgV;
    Mat motion_flow, flow_rgb;

    char cad[N_CHAR];
    struct timeval tod1;
    double t1 = 0.0, t2 = 0.0, tdflow = 0.0, t1fr = 0.0, t2fr = 0.0, tdframe = 0.0;

	int start_with_vid = 1;
	int gpuID = 0;
	int type = 1;
	int frameSkip = 1;


	int vidcount = 0;

	const char* keys = "{ h  | help      | false | print help message }"
				"{ v  | start_video     |  1    | start video id }"
				"{ g  | gpuID     |  1    | use this gpu}"
				"{ f  | type     |  1    | use this flow method (0=Brox, 1=TV-L1)}"
				"{ s  | skip     |  1    | frame skip}"
				"{ vp  | vid_path     |  ./    | path input (where the videos are)}"
				"{ op  | out_path     |  ./    | path output}";

	CommandLineParser cmd(argc, argv, keys);

	if (cmd.get<bool>("help"))
	{
		cout << "Usage: compute_flow [options]" << endl;
		cout << "Avaible options:" << endl;
		cmd.printParams();
		return 0;
	}

	if (argc > 1) {
		start_with_vid = cmd.get<int>("start_video");
		gpuID = cmd.get<int>("gpuID");
		type = cmd.get<int>("type");
		frameSkip = cmd.get<int>("skip");
		vid_path = cmd.get<std::string>("vid_path");
		out_path = cmd.get<std::string>("out_path");
		out_path_jpeg = out_path + "/rgb/";
		cout << "start_vid:" << start_with_vid << "gpuID:" << gpuID << "flow method: "<< type << " frameSkip: " << frameSkip << " vid_path: " << vid_path << " out_path" << out_path << " jpegs: " << out_path_jpeg << endl;
	}

	int totalvideos = 0;
	DIR * dirp;
	struct dirent * entry;

	dirp = opendir(vid_path.c_str()); /* There should be error handling after this */
	while ((entry = readdir(dirp)) != NULL) {
	    if (entry->d_type == DT_REG) { /* If the entry is a regular file */
	         totalvideos++;
	    }
	}
	closedir(dirp);

	//cv::Ptr<Feature2D> detector_surf = xfeatures2d::SurfFeatureDetector::create(200);
	//cv::Ptr<Feature2D> extractor_surf = xfeatures2d::SurfDescriptorExtractor::create(true, true);
	SurfFeatureDetector detector_surf(200);
	SurfDescriptorExtractor extractor_surf;

	std::vector<Point2f> prev_pts_flow, pts_flow;
	std::vector<Point2f> prev_pts_surf, pts_surf;
	std::vector<Point2f> prev_pts_all, pts_all;
	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf;
	Mat capture_frame, capture_image, prev_image, capture_gray, prev_gray, human_mask;

	cv::gpu::setDevice(gpuID);

    cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

    cv::gpu::BroxOpticalFlow dflow(alpha_,gamma_,scale_factor_,inner_iterations_,outer_iterations_,solver_iterations_);

	cv::gpu::OpticalFlowDual_TVL1_GPU alg_tvl1;

	QString vpath = QString::fromStdString(vid_path);
	QStringList filters;

	QDirIterator dirIt(vpath, QDirIterator::Subdirectories);


	int vidID = 0;
	std::string video, outfile_u, outfile_v, outfile_jpeg, outfile;

	for (; (dirIt.hasNext()); )
	{
		//std::cout << "asdf "<< std::endl;
		dirIt.next();
		QString file = dirIt.fileName();
			if ((QFileInfo(dirIt.filePath()).suffix() == "mp4") || (QFileInfo(dirIt.filePath()).suffix() == "avi"))
			{
				video = dirIt.filePath().toStdString();
			}

			else
				continue;

			vidID++;

			if (vidID < start_with_vid)
				continue;


			std::string fName(video);
			std::string path(video);
			size_t last_slash_idx = std::string::npos;
			if (!createOutDirs)
			{
				// Remove directory if present.
				// Do this before extension removal incase directory has a period character.
				std::cout << "removing directories: " << fName << std::endl;
				last_slash_idx = fName.find_last_of("\\/");
				if (std::string::npos != last_slash_idx)
				{
					fName.erase(0, last_slash_idx + 1);
					path.erase(last_slash_idx + 1, path.length());
				}
			}
			else
			{
				last_slash_idx = fName.find(vid_path);
				fName.erase(0, vid_path.length());
				path.erase(vid_path.length(), path.length());
			}

			// Remove extension if present.
			const size_t period_idx = fName.rfind('.');
			if (std::string::npos != period_idx)
				fName.erase(period_idx);

			/*QString out_folder_u = QString::fromStdString(out_path + "x/" + fName);
			bool folder_exists = QDir(out_folder_u).exists();*/

			QString out_folder = QString::fromStdString(out_path + fName);

			bool folder_exists = QDir(out_folder).exists();

			if (folder_exists) {
				std::cout << "already exists: " << out_path << fName << std::endl;
				continue;
			}

			bool folder_created = QDir().mkpath(out_folder);
			if (!folder_created) {
				std::cout << "cannot create: " << out_path << fName << std::endl;
				continue;
			}

			/*QString out_folder_v = QString::fromStdString(out_path + "y/" + fName);
			QDir().mkpath(out_folder_v);*/
			if(rgb){
				QString out_folder_jpeg = QString::fromStdString(out_path_jpeg + fName);
				QDir().mkpath(out_folder_jpeg);
				outfile_jpeg = out_folder_jpeg.toStdString();
			}

			// Create a separate folder for the .bins
			FILE *fx = NULL;
			if (bins == true){
				QString out_folder_bins = QString::fromStdString(out_path + "bins/" + fName);
				QDir().mkpath(out_folder_bins);
				std::string outfile = out_path + "bins/" + fName + ".bin";
				// Uncomment this if you want bins
				//FILE *fx = fopen(outfile.c_str(),"wb");
			}


			//if(debug){
			std::cout << video << "    " << vidcount << "/" << totalvideos <<  std::endl;
			//}
			vidcount++;

			VideoCapture cap;
			try
			{
				cap.open(video);
			}
			catch (std::exception& e)
			{
				std::cout << e.what() << '\n';
			}
			int width_out = 224, height_out = 224;
			int nframes = 0, width = 0, height = 0;
			float factor = 0, factor_out = 0;

			if( cap.isOpened() == 0 )
			{
				return -1;
			}

			cap >> frame1_rgb_;

			if( resize_img == true )
			{
				factor = std::max<float>(MIN_SZ/frame1_rgb_.cols, MIN_SZ/frame1_rgb_.rows);

				width = std::floor(frame1_rgb_.cols*factor);
				width -= width%2;
				height = std::floor(frame1_rgb_.rows*factor);
				height -= height%2;

				frame1_rgb = cv::Mat(Size(width,height),CV_8UC3);
				width = frame1_rgb.cols;
				height = frame1_rgb.rows;
				cv::resize(frame1_rgb_,frame1_rgb,cv::Size(width_out,height_out),0,0,INTER_CUBIC);

				factor_out = std::max<float>(OUT_SZ/width, OUT_SZ/height);

				rgb_out = cv::Mat(Size(cvRound(width*factor_out),cvRound(height*factor_out)),CV_8UC3);
				//width_out = rgb_out.cols;
				//height_out = rgb_out.rows;
			}
			else
			{
				frame1_rgb = cv::Mat(Size(frame1_rgb_.cols,frame1_rgb_.rows),CV_8UC3);
				width = frame1_rgb.cols;
				height = frame1_rgb.rows;
				frame1_rgb_.copyTo(frame1_rgb);
			}

			// Allocate memory for the images
			frame0_rgb = cv::Mat(Size(width,height),CV_8UC3);
			flow_rgb = cv::Mat(Size(width,height),CV_8UC3);
			motion_flow = cv::Mat(Size(width,height),CV_8UC3);
			frame0 = cv::Mat(Size(width,height),CV_8UC1);
			frame1 = cv::Mat(Size(width,height),CV_8UC1);
			frame0_32 = cv::Mat(Size(width,height),CV_32FC1);
			frame1_32 = cv::Mat(Size(width,height),CV_32FC1);

			// Convert the image to grey and float
			cvtColor(frame1_rgb,frame1,CV_BGR2GRAY);
			frame1.convertTo(frame1_32,CV_32FC1,1.0/255.0,0);

			outfile = out_folder.toStdString();
			//outfile_u = out_folder_u.toStdString();
			//outfile_v = out_folder_v.toStdString();


			while( frame1.empty() == false )
			{
			    gettimeofday(&tod1,NULL);
			    t1fr = tod1.tv_sec + tod1.tv_usec / 1000000.0;
				if( nframes >= 1 )
				{
				    gettimeofday(&tod1,NULL);
					//	GetSystemTime(&tod1);
				    t1 = tod1.tv_sec + tod1.tv_usec / 1000000.0;
				    switch(type){
						case 0:
							frame1GPU.upload(frame1_32);
							frame0GPU.upload(frame0_32);
							dflow(frame0GPU,frame1GPU,uGPU,vGPU);
						case 1:
							frame1GPU.upload(frame1);
							frame0GPU.upload(frame0);
							alg_tvl1(frame0GPU,frame1GPU,uGPU,vGPU);
					}
					if (warp == true){
						//get back flow map
						Mat flow_x(uGPU);
						Mat flow_y(vGPU);

						// warp to reduce holistic motion (i'm not sure if frame1 is grayscale)
                        //std::cout << "Detector:" << std::endl;
						detector_surf.detect(frame1, kpts_surf, human_mask);
                        //std::cout << "Extractor:" << std::endl;
						extractor_surf.compute(frame1, kpts_surf, desc_surf);
                        //std::cout << "Computing match:" << std::endl;
						ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);
                        //std::cout << "Matching flows:" << std::endl;
                        MatchFromFlow_copy(frame1, flow_x, flow_y, prev_pts_flow, pts_flow, human_mask);
                        //std::cout << "Fixing match:" << std::endl;
                        MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);

						Mat H = Mat::eye(3, 3, CV_64FC1);
						if(pts_all.size() > 50) {
							std::vector<unsigned char> match_mask;
							Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
							if(cv::countNonZero(Mat(match_mask)) > 25)
								H = temp;
						}

						Mat H_inv = H.inv();
						Mat gray_warp = Mat::zeros(frame1.size(), CV_8UC1);
						MyWarpPerspective(frame0, frame1, gray_warp, H_inv); // Most important function

						// re-extract flow on warped images
                        //std::cout << "Recomputing flow on warped images:" << std::endl;
					    frame1GPU.upload(gray_warp);
					    frame0GPU.upload(frame0);
					    alg_tvl1(frame0GPU,frame1GPU,uGPU,vGPU);
					}
					uGPU.download(imgU);
					vGPU.download(imgV);

			    gettimeofday(&tod1,NULL);
	            t2 = tod1.tv_sec + tod1.tv_usec / 1000000.0;
	            tdflow = 1000.0*(t2-t1);

			}else{
				if (warp == true){
					//video_stream >> capture_frame;
					//if (capture_frame.empty()) return; // read frames until end
					initializeMats(frame1, capture_image, capture_gray, prev_image, prev_gray);
					capture_frame.copyTo(frame0);
					cvtColor(prev_image, prev_gray, CV_BGR2GRAY);

					//detect key points
					human_mask = Mat::ones(capture_frame.size(), CV_8UC1);
					detector_surf.detect(prev_gray, prev_kpts_surf, human_mask);
					extractor_surf.compute(prev_gray, prev_kpts_surf, prev_desc_surf);
					// TODO! check detector_surf->detectAndCompute()
				}
			}

				if( WRITEOUT_IMGS == true &&  nframes >= 1 )
				{
					if( resize_img == true )
					{

						cv::resize(imgU,imgU,cv::Size(width_out,height_out),0,0,INTER_CUBIC);
						cv::resize(imgV,imgV,cv::Size(width_out,height_out),0,0,INTER_CUBIC);

					}


					double min_u, max_u;
					cv::minMaxLoc(imgU, &min_u, &max_u);
					double min_v, max_v;
					cv::minMaxLoc(imgV, &min_v, &max_v);


					float min_u_f = min_u;
					float max_u_f = max_u;

					float min_v_f = min_v;
					float max_v_f = max_v;

					if (clipFlow) {
						min_u_f = -20;
						max_u_f = 20;

						min_v_f = -20;
						max_v_f = 20;
					}

					cv::Mat img_u(imgU.rows, imgU.cols, CV_8UC1);
					cv::Mat img_v(imgV.rows, imgV.cols, CV_8UC1);
					cv::Mat img_norm_uv(imgV.rows, imgV.cols, CV_8UC1);

					GpuMat u_sq, v_sq, uv_sq, norm_uv;
					Mat normUV;
					cv::gpu::sqr(uGPU, u_sq);
					cv::gpu::sqr(vGPU, v_sq);
					cv::gpu::add(u_sq, v_sq, uv_sq);
					cv::gpu::sqrt(uv_sq, norm_uv);
					norm_uv.download(normUV);
					convertFlowToImage(normUV, img_norm_uv, min_u_f, max_u_f);\
					convertFlowToImage(imgU, img_u, min_u_f, max_u_f);
					convertFlowToImage(imgV, img_v, min_v_f, max_v_f);

					std::vector<cv::Mat> images(3);
					Mat black = Mat::zeros(imgU.rows, imgU.cols, CV_8UC1);
					images.at(0) = img_u; //for blue channel
					images.at(1) = img_v;   //for green channel
					images.at(2) = img_norm_uv;  //for red channel

					cv::Mat colorImage;
					cv::merge(images, colorImage);
					sprintf(cad,"/frame%06d.jpg",nframes);

					//imwrite(outfile_u+cad,img_u);
					//imwrite(outfile_v+cad,img_v);
					imwrite(outfile+cad,colorImage);
					if (bins == true){
						fwrite(&min_u_f,sizeof(float),1,fx);
						fwrite(&max_u_f,sizeof(float),1,fx);
						fwrite(&min_v_f,sizeof(float),1,fx);
						fwrite(&max_v_f,sizeof(float),1,fx);
					}
				}

				sprintf(cad,"/frame%06d.jpg",nframes + 1);
				if(rgb){
					if( resize_img == true )
					{
						cv::resize(frame1_rgb,rgb_out,cv::Size(width_out,height_out),0,0,INTER_CUBIC);
						imwrite(outfile_jpeg+cad,rgb_out);
					}
					else
						imwrite(outfile_jpeg+cad,frame1_rgb);
				}
				if(debug){
					std::cout << "writing:" << outfile_jpeg+cad << std::endl;
				}
				frame1_rgb.copyTo(frame0_rgb);
				cvtColor(frame0_rgb,frame0,CV_BGR2GRAY);
				frame0.convertTo(frame0_32,CV_32FC1,1.0/255.0,0);

				nframes++;
				for (int iskip = 0; iskip<frameSkip; iskip++)
				{
					cap >> frame1_rgb_;
				}
				if( frame1_rgb_.empty() == false )
				{
					if( resize_img == true )
					{
						cv::resize(frame1_rgb_,frame1_rgb,cv::Size(width_out,height_out),0,0,INTER_CUBIC);
					}
					else
					{
						frame1_rgb_.copyTo(frame1_rgb);
					}

					cvtColor(frame1_rgb,frame1,CV_BGR2GRAY);
					frame1.convertTo(frame1_32,CV_32FC1,1.0/255.0,0);
				}
				else
				{
					break;
				}

				gettimeofday(&tod1,NULL);
				if(debug){
					t2fr = tod1.tv_sec + tod1.tv_usec / 1000000.0;
					tdframe = 1000.0*(t2fr-t1fr);
					cout << "Processing video" << fName << "ID="<< vidID <<  " Frame Number: " << nframes << endl;
					cout << "Time type=" << type <<  " Flow: " << tdflow << " ms" << endl;
					cout << "Time All: " << tdframe << " ms" <<  endl;
				}

			}
			if (bins == true){
				fclose(fx);
			}
		}

    return 0;
}
