#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

void GLT(Mat srcimage, Mat dstImg, int a, int b); //Global Linear Transformation
void NPT(Mat srcimage, Mat dstImg, int a); //Negative phase transformation
void GLL(Mat dstImg); // Gray Level Layering
void binary(int num);
void BPL(Mat srcImage); // Bit plane layering
void LT(Mat srcImage); // Logarithmic transformation
void logTransform1(Mat srcImage, int c); // 对数变换方法1
void logTransform2(Mat srcImage, float c); // 对数变换方法2
Mat logTransform3(Mat srcImage, float c); // 对数变换方法3
void MyGammaCorrection(Mat& src, Mat& dst, float fGamma); //Gamma transformation
int b[8];

int main()
{
	Mat srcimage = imread("4.jpg");
	if (srcimage.empty()) { cout << "image empty!" << endl; return -1; }
	imshow("原图", srcimage);

	Mat dstImg = srcimage.clone();

	//计算图像像素的最大值和最小值
	int pixMax = -1000, pixMin = 1000;
	for (int j = 0; j < srcimage.rows; j++)
	{
		uchar* pDataMat = srcimage.ptr<uchar>(j);
		for (int i = 0; i < srcimage.cols; i++)
		{
			if (pDataMat[i] > pixMax)
				pixMax = pDataMat[i];
			if (pDataMat[i] < pixMin)
				pixMin = pDataMat[i];
		}
	}

	// cout << pixMax << endl << pixMin << endl;
	// cout << srcimage.rows << endl << srcimage.cols << endl;


	Mat dstImg1 = dstImg.clone();
	GLT(srcimage, dstImg1, pixMin, pixMax); // 全域线性变换

	Mat dstImg2 = dstImg.clone();
	NPT(srcimage, dstImg2, pixMax); // 负相变换

	Mat dstImg3 = dstImg.clone();
	GLL(dstImg3); // 灰度层级

	BPL(srcimage); // 8比特层级

	Mat Image; // cimg -> gimg 
	cvtColor(dstImg, Image, COLOR_BGR2GRAY); // 灰度化图像
	LT(Image);

	Mat dst;
	float fGamma = 1 / 2.2;
	MyGammaCorrection(dstImg, dst, fGamma);

	imshow("Dst", dst);

	waitKey();
	return 0;
}

void GLT(Mat srcimage, Mat dstImg, int a, int b) //Global Linear Transformation
{
	int c = 0, d = 255;
	for (int i = 0; i < srcimage.rows; i++)
	{
		uchar* srcData = srcimage.ptr<uchar>(i);
		for (int j = 0; j < srcimage.cols; j++)
		{
			float k = (d - c) / (b - a);
			float kb = c - k * a;
			int pixel = srcData[j] * k + kb;

			if (srcimage.type() == CV_8UC1)
			{
				if (srcimage.at<uchar>(i, j) < a)
				{
					dstImg.at<uchar>(i, j) = c;
				}
				else if (srcimage.at<uchar>(i, j) < b)
				{
					dstImg.at<uchar>(i, j) = pixel;
				}
				else
				{
					dstImg.at<uchar>(i, j) = d;
				}
			}
			else if (srcimage.type() == CV_8UC3)
			{
				for (int x = 0; x < 3; x++)
				{
					if (srcimage.at<Vec3b>(i, j)[x] < a)
					{
						dstImg.at<Vec3b>(i, j)[x] = c;
					}
					else if (srcimage.at<Vec3b>(i, j)[x] < b)
					{
						dstImg.at<Vec3b>(i, j)[x] = srcimage.at<Vec3b>(i, j)[x] * k + kb;
					}
					else
					{
						dstImg.at<Vec3b>(i, j)[x] = d;
					}
				}
			}
		}
	}
	imshow("灰度线性变换", dstImg);
}

void NPT(Mat srcimage, Mat dstImg, int a)//Negative phase transformation
{
	for (int i = 0; i < srcimage.rows; i++)
	{
		for (int j = 0; j < srcimage.cols; j++)
		{
			if (srcimage.type() == CV_8UC1)
			{
				dstImg.at<uchar>(i, j) = a - srcimage.at<uchar>(i, j);
			}
			else if (srcimage.type() == CV_8UC3)
			{
				for (int x = 0; x < 3; x++)
				{
					dstImg.at<Vec3b>(i, j)[x] = a - srcimage.at<Vec3b>(i, j)[x];
				}
			}
		}
	}
	imshow("负相变换", dstImg);
}

void GLL(Mat dstImg)// Gray Level Layering
{
	int controlMin = 100, controlMax = 200;
	for (int i = 0; i < dstImg.rows; i++)
	{
		for (int j = 0; j < dstImg.cols; j++)
		{
			//第一种方法，二值映射
			if (dstImg.type() == CV_8UC1)
			{
				/*if (dstImg.at<uchar>(i, j) > controlMin)
					dstImg.at<uchar>(i, j) = 255;
				else
					dstImg.at<uchar>(i, j) = 0;*/
					//第二种方法：区域映射
				if (dstImg.at<uchar>(i, j) < controlMax && dstImg.at<uchar>(i, j) > controlMin)
					dstImg.at<uchar>(i, j) = controlMax;
			}
			else if (dstImg.type() == CV_8UC3)
			{
				for (int x = 0; x < 3; x++)
				{
					/*if (dstImg.at<Vec3b>(i, j)[x] > controlMin)
						dstImg.at<Vec3b>(i, j)[x] = 255;
					else
						dstImg.at<Vec3b>(i, j)[x] = 0;*/
						//第二种方法：区域映射
					if (dstImg.at<Vec3b>(i, j)[x] < controlMax && dstImg.at<Vec3b>(i, j)[x] > controlMin)
						dstImg.at<Vec3b>(i, j)[x] = controlMax;
				}
			}
		}
	}
	imshow("灰度层级变换", dstImg);
}

void binary(int num)
{
	for (int i = 0; i < 8; i++)
		b[i] = 0;
	int i = 0;
	while (num != 0)
	{
		b[i] = num % 2;
		num = num / 2;
		i++;
	}
}

void BPL(Mat srcImage)// Bit plane layering
{
	resize(srcImage, srcImage, cv::Size(), 0.5, 0.5);
	Mat d[8];
	for (int k = 0; k < 8; k++)
		d[k].create(srcImage.size(), srcImage.type());

	int rowNumber = srcImage.rows, colNumber = srcImage.cols;

	for (int i = 0; i < rowNumber; i++)
	{
		for (int j = 0; j < colNumber; j++)
		{
			if (srcImage.type() == CV_8UC1)
			{
				int num = srcImage.at<uchar>(i, j);
				binary(num);
				for (int k = 0; k < 8; k++)
				{
					d[k].at<uchar>(i, j) = b[k] * 255;
				}
			}
			else
			{
				for (int x = 0; x < 3; x++)
				{
					int num = srcImage.at<Vec3b>(i, j)[x];
					binary(num);
					for (int k = 0; k < 8; k++)
					{
						d[k].at<Vec3b>(i, j)[x] = b[k] * 255;
					}
				}
			}
		}
	}

	imshow("src", srcImage);

	for (int k = 0; k < 8; k++)
	{
		imshow("level" + std::to_string(k), d[k]);
	}
}


void logTransform1(Mat srcImage, int c) // 对数变换方法1
{
	// 输入图像判断
	if (srcImage.empty())
		cout << "No data!" << endl;
	Mat resultImage = Mat::zeros(srcImage.size(), srcImage.type());
	// 计算 1 + r
	add(srcImage, cv::Scalar(1.0), srcImage);
	// 转换为32位浮点数
	srcImage.convertTo(srcImage, CV_32F);
	// 计算 log(1 + r)
	log(srcImage, resultImage);
	resultImage = c * resultImage;
	// 归一化处理
	normalize(resultImage, resultImage, 0, 255, cv::NORM_MINMAX);
	convertScaleAbs(resultImage, resultImage);
	imshow("logTransform1", resultImage);
}

void logTransform2(Mat srcImage, float c) // 对数变换方法2
{
	// 输入图像判断
	if (srcImage.empty())
		cout << "No data!" << endl;
	Mat resultImage = Mat::zeros(srcImage.size(), srcImage.type());
	double gray = 0;
	// 图像遍历分别计算每个像素点的对数变换 
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {
			gray = (double)srcImage.at<uchar>(i, j);
			gray = c * log((double)(1 + gray));
			resultImage.at<uchar>(i, j) = saturate_cast<uchar>(gray);
		}
	}
	// 归一化处理
	normalize(resultImage, resultImage, 0, 255, cv::NORM_MINMAX);
	convertScaleAbs(resultImage, resultImage);
	imshow("logTransform2", resultImage);
}

Mat logTransform3(Mat srcImage, float c) // 对数变换方法3
{
	// 输入图像判断
	if (srcImage.empty())
		cout << "No data!" << endl;
	Mat resultImage = Mat::zeros(srcImage.size(), srcImage.type());
	srcImage.convertTo(resultImage, CV_32F);
	resultImage = resultImage + 1;
	log(resultImage, resultImage);
	resultImage = c * resultImage;
	normalize(resultImage, resultImage, 0, 255, cv::NORM_MINMAX);
	convertScaleAbs(resultImage, resultImage);
	return resultImage;
}

void LT(Mat srcImage) // Logarithmic transformation
{
	// 验证三种不同方式的对数变换速度
	float c = 2;
	Mat resultImage;
	double tTime;
	imshow("graySrcimage", srcImage);
	tTime = (double)getTickCount();

	Mat srcImage1 = srcImage.clone();
	resultImage = logTransform3(srcImage1, c);
	tTime = 1000 * ((double)getTickCount() - tTime) / getTickFrequency();
	cout << "第三种方法耗时：" << tTime << "ms" << endl;

	Mat srcImage2 = srcImage.clone();
	logTransform1(srcImage2, c);
	Mat srcImage3 = srcImage.clone();
	logTransform2(srcImage3, c);

	imshow("resultImage", resultImage);
}

void MyGammaCorrection(Mat& src, Mat& dst, float fGamma) //Gamma transformation
{

	// build look up table 
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}

	dst = src.clone();
	const int channels = dst.channels();
	switch (channels)
	{
	case 1:   //灰度图的情况
	{
		MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			//*it = pow((float)(((*it))/255.0), fGamma) * 255.0; 
			*it = lut[(*it)];

		break;
	}
	case 3:  //彩色图的情况
	{

		MatIterator_<Vec3b> it, end;
		for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
		{
			//(*it)[0] = pow((float)(((*it)[0])/255.0), fGamma) * 255.0; 
			//(*it)[1] = pow((float)(((*it)[1])/255.0), fGamma) * 255.0; 
			//(*it)[2] = pow((float)(((*it)[2])/255.0), fGamma) * 255.0; 
			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}

		break;

	}
	}
}