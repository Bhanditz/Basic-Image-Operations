#ifndef Bio_H
#define Bio_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include <chrono>

/**
@brief Class of basic image processing methods as an extension to OpenCV.

@author Yuan Li
*/
class Bio
{
public:
	static struct Point_data  // keeps track of all the points that you clicked on 
	{
		cv::Mat* image;
		bool update;
		int counter;
		std::vector<cv::Point3i> initialPoints;  // clicked points on the computer screen
	};

	static void load(cv::String path, std::vector<cv::String> &names, std::vector<cv::Mat> &images, int isColor);
	static void write(std::vector<cv::Point3i> points, cv::String name, std::string outFile);
	static void blend(std::vector<cv::String> &names, std::vector<cv::Mat> &imagesIn, std::vector<cv::Mat> &imagesOut, cv::Mat mask, double alpha, double beta);
	static void resize(cv::Mat inImage, cv::Mat* outImage, int outD);
	static void pointsFunc(int event, int x, int y, int flags, void* ptr);
	static void label(const cv::String &winname, struct Point_data *point_data);

	static void separateRGB(cv::Mat &inImage, std::vector<cv::Mat> &bgr);
};

#endif
