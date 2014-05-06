#pragma once
#ifndef UTILS_H
#define UTILS_H

#include <cstdarg>
#include <Eigen/Eigen>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace Eigen;
using namespace std;

double round(double d);

double sigmoid(double z);

double sigmoidGradient(double z);

string vformat(const char *fmt, va_list ap);

string formatString(const char *fmt, ...);

int reverseInt(int i);

void read_mnist(const char* Xfile, MatrixXd& X, const char* yfile, VectorXi& y, unsigned int maxImages);

void displayData(MatrixXd X);

#endif