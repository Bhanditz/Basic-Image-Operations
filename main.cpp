/**
main.cpp Main function for training and testing SVM classifier.
@Brief This file implements a complete image classification using Support Vector Machine
with OpenCV and my own mini library - Bio (Basic Image Operations).

@last update 7/26/2017
@author Yuan Li
*/

#include <stdlib.h>
#include "BasicImageOperations.h"

#ifdef _WIN32
#define WINPAUSE system("pause")
#endif

#define numTrain 1009
#define numTest 450
#define imgSize 32

//using namespace cv;
//using namespace cv::ml;
//using namespace std;

int main(int, char**)
{
	// Set up training data
	std::string name, label;
	std::ifstream inputfile;

	//load training data and labels
	std::string root_train = "data/training/";

	int labelsTrain[numTrain];
	float dataTrain[numTrain][imgSize * imgSize];
	std::vector<float> dataTrainVec;
	inputfile.open(root_train + "train.txt");
	
	if (inputfile.is_open())
	{
		int i = 0;
		for (std::string line; std::getline(inputfile, line); i++)
		{
			inputfile >> name >> label;
			labelsTrain[i] = atoi(label.c_str()); //convert string to int
			cv::Mat imgOri = cv::imread(root_train + name);
			cv::Mat imgRsz;
			//cv::resize(imgOri, imgRsz, cv::Size(imgSize, imgSize)); //unpadded
			Bio::resize(imgOri, &imgRsz, imgSize); //padded

			cvtColor(imgRsz, imgRsz, CV_BGR2GRAY); //convert to 1 channel

			dataTrainVec.assign(imgRsz.datastart, imgRsz.dataend);

			for (int j = 0; j < imgSize * imgSize; j++)
			{
				dataTrain[i][j] = dataTrainVec[j];
			}

			//cv::imshow("dataTrain", imgRsz); cv::waitKey(0); cvDestroyWindow("dataTrain");
		}
	}
	else
	{
		std::cout << "Failed to read the training data." << std::endl;
		inputfile.close();

		WINPAUSE;
		return 0;
	}

	inputfile.close();
	cv::Mat labelsMatTrain(numTrain, 1, CV_32SC1, labelsTrain);
	cv::Mat dataMatTrain(numTrain, imgSize * imgSize, CV_32FC1, dataTrain);

	// Set up SVM's parameters
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR);
	//svm->setDegree(3);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10000, 1e-8));

	// Train the SVM with given parameters
	cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(dataMatTrain, cv::ml::ROW_SAMPLE, labelsMatTrain);
	svm->train(td);
	// Or train the SVM with optimal parameters
	//svm->trainAuto(td, 10);

	svm->save("svm_trained.xml");

	cv::Ptr<cv::ml::SVM> svm_trained = cv::ml::SVM::load<cv::ml::SVM>("svm_trained.xml");

	//load testing data and labels
	std::string root_test = "data/testing/";

	int labelsTest[numTest];
	float resultsTest[numTest];
	float dataTest[numTest][imgSize * imgSize];
	std::vector<float> dataTestVec;
	inputfile.open(root_test + "test.txt");

	auto start = std::chrono::high_resolution_clock::now();

	if (inputfile.is_open())
	{
		int i = 0;
		for (std::string line; getline(inputfile, line); i++)
		{
			inputfile >> name >> label;
			labelsTest[i] = atoi(label.c_str()); //convert string to int
			cv::Mat imgOri = cv::imread(root_test + name);
			cv::Mat imgRsz;

			//resize(imgOri, imgRsz, cv::Size(imgSize, imgSize)); //unpadded
			Bio::resize(imgOri, &imgRsz, imgSize); //padded

			cv::cvtColor(imgRsz, imgRsz, CV_BGR2GRAY); //convert to 1 channel

			dataTestVec.assign(imgRsz.datastart, imgRsz.dataend);

			std::copy(dataTestVec.begin(), dataTestVec.end(), dataTest[i]);

			cv::Mat svmTest(1, imgSize * imgSize, CV_32FC1, dataTest[i]);
			resultsTest[i] = svm_trained->predict(svmTest);

			if (resultsTest[i] != labelsTest[i])
			{
				imwrite(root_test + "Miss Classified/" + std::to_string((int)resultsTest[i]) +
					"_" + name.substr(2, name.length()), imgOri);
			}
		}
	}
	else
	{
		std::cout << "Failed to read the test data." << std::endl;
		inputfile.close();
		
		WINPAUSE;
		return 0;
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "profiling: " << elapsed.count() << "ms" << std::endl;

	inputfile.close();
	cv::Mat labelsMatTest(numTest, 1, CV_32SC1, labelsTest);
	cv::Mat dataMatTest(numTest, imgSize * imgSize, CV_32FC1, dataTest);

	//Compute error
	float errTest = 0;
	for (int i = 0; i < numTest; i++)
		errTest += (resultsTest[i] != labelsTest[i]);
	errTest /= numTest;
	std::cout << "errTest = " << errTest << std::endl;
	std::cout << "accTest = " << 100 * (1 - errTest) << "%" << std::endl;

	WINPAUSE;

	return (1);
}
