#pragma once
#include <fstream>
#include <direct.h>
#include <opencv2/opencv.hpp>
#include <utility>

#include "utils.h"


class detector_cv
{
public:

    std::vector<Detection> detect(cv::Mat& image, const float& confThreshold, const float& iouThreshold);

    const std::vector<cv::Scalar> colors = { cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0) };

    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float SCORE_THRESHOLD = 0.2;
    const float NMS_THRESHOLD = 0.4;
    const float CONFIDENCE_THRESHOLD = 0.4;

    const int dimensions = 85;
    const int rows = 25200;

    const int classNamesize = 80;



    void load_net(const std::string& modelPath,  const bool& isGPU);

private:
    cv::dnn::Net cvdnnNET;


};

