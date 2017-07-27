#include "BasicImageOperations.h"

// load multiple images from a single folder
void Bio::load(cv::String path, std::vector<cv::String> &names, std::vector<cv::Mat> &images, int isColor)
{
	cv::glob(path, names, true);

	cv::Mat im;
	for (size_t i = 0; i < names.size(); ++i)
	{
		im = cv::imread(names[i], isColor);
		if (im.empty())
			continue;

		images.push_back(im);
	}
}

// write labels into file
void Bio::write(std::vector<cv::Point3i> points, cv::String name, std::string outFile)
{
	std::ofstream fout;
	fout.open(outFile, std::ios_base::app);

	if (fout.is_open())
	{
		printf("Writing labels to file...\n");
		for (std::vector<cv::Point3i>::iterator it = points.begin(); it != points.end(); ++it)
		{
			fout << it->x << ", " << it->y << ", " << it->z << ", " << name << "\n";
		}
	}
	else //file could not be opened
	{
		printf("File to write to could not be opened.\n");
	}
}

// mouse callback function
void Bio::pointsFunc(int event, int x, int y, int flags, void* ptr)
{
	Point_data* point_data = (Point_data*)ptr;

	if (event == cv::EVENT_LBUTTONDOWN)
	{
		point_data->counter++;

		cv::Scalar I = point_data->image->at<uchar>(cv::Point(x, y));
		point_data->initialPoints.push_back(cv::Point3i(x, y, I[0]));

		std::cout << "Point " << point_data->counter << ": (" << x << ", " << y << ") I = " << I[0] << std::endl;
		//printf("Point %d is taken at (%d, %d) with intensity I = %d.\n", point_data->counter, x, y, I[0]);
	}
}

// record clicked points
void Bio::label(const cv::String &winname, struct Point_data *point_data)
{
	cv::setMouseCallback("img", pointsFunc, point_data); //click to label IR data, store into vector

	int key = 0;
	while (key != 110) // wait for 'n' being pressed, get next image
	{
		point_data->update = true;
		cv::imshow("img", *point_data->image);
		key = cv::waitKey(1);
	}
}

// blend a set of images with a mask
void Bio::blend(std::vector<cv::String> &names, std::vector<cv::Mat> &imagesIn, std::vector<cv::Mat> &imagesOut, cv::Mat mask, double alpha, double beta)
{
	cv::Mat dst;
	for (size_t i = 0; i < names.size(); ++i)
	{
		cv::addWeighted(imagesIn[i], alpha, mask, beta, 0.0, dst);
		imagesOut.push_back(dst);
	}
}

void Bio::separateRGB(cv::Mat &inImage, std::vector<cv::Mat> &bgr)
{
	std::vector<cv::Mat> channels;
	cv::split(inImage, channels);

	// Create an zero pixel image for filling purposes
	cv::Mat empty_image = cv::Mat::zeros(inImage.rows, inImage.cols, CV_8UC1);
	cv::Mat channel_blue(inImage.rows, inImage.cols, CV_8UC3);
	cv::Mat channel_green(inImage.rows, inImage.cols, CV_8UC3);
	cv::Mat channel_red(inImage.rows, inImage.cols, CV_8UC3);

	// Create blue channel
	cv::Mat b[] = { channels[0], empty_image, empty_image };
	int from_to_b[] = { 0, 0, 1, 1, 2, 2 };
	cv::mixChannels(b, 3, &channel_blue, 1, from_to_b, 3);
	bgr.push_back(channel_blue);

	// Create green channel
	cv::Mat g[] = { empty_image, channels[1], empty_image };
	int from_to_g[] = { 0, 0, 1, 1, 2, 2 };
	cv::mixChannels(g, 3, &channel_green, 1, from_to_g, 3);
	bgr.push_back(channel_green);

	// Create red channel
	cv::Mat r[] = { empty_image, empty_image, channels[2] };
	int from_to_r[] = { 0, 0, 1, 1, 2, 2 };
	cv::mixChannels(r, 3, &channel_red, 1, from_to_r, 3);
	bgr.push_back(channel_red);
}

// resize the image with the aspect ratio kept
void Bio::resize(cv::Mat inImage, cv::Mat* outImage, int outD)
{
	int inW = inImage.cols, newW;
	int inH = inImage.rows, newH;

	int border;

	float ratio = (float)inW / (float)inH;

	cv::Mat temp;
	//cv::Mat blackImage(outD, outD, CV_8UC1, Scalar(0));

	if (ratio > 1)
	{
		newH = outD / ratio;
		cv::resize(inImage, temp, cv::Size(outD, newH));
		border = (outD - newH) / 2;
		cv::copyMakeBorder(temp, *outImage, border, outD - newH - border, 0, 0, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	}
	else
	{
		newW = outD * ratio;
		cv::resize(inImage, temp, cv::Size(newW, outD));
		border = (outD - newW) / 2;
		cv::copyMakeBorder(temp, *outImage, 0, 0, border, outD - newW - border, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	}

	//std::cout << "outImage.rows = " << outImage->rows << "; " << "outImage.cols = " << outImage->cols << std::endl;
}
