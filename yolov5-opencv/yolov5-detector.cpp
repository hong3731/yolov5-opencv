// yolov5-opencv.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <direct.h>
#include <fstream>
#include "utils.h"
#include "detector.h"
#include "detector_cv.h"
#include "detector_OV.h"

std::string current_working_directory()
{
    char buff[250];
    _getcwd(buff, 250);
    std::string current_working_directory(buff);
    return current_working_directory;
}

std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;

    std::string dir = current_working_directory();

    std::ifstream ifs(dir + "/config_files/classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}



int main()
{

    const float confThreshold = 0.3f;
    const float iouThreshold = 0.4f;

    std::vector<std::string> classNames = load_class_list();
    const std::string modelPath = "config_files/yolov5n.onnx";
    const std::string imagePath = "zidane.jpg";


    detector_cv detector_usingcv;
    detector_usingcv.load_net(modelPath, false);
    
    YOLODetector detector_usingonnx{ nullptr };
    detector_usingonnx = YOLODetector(modelPath, true, cv::Size(640, 640));

    Detector_OV detector_usingov{ nullptr };
    detector_usingov = Detector_OV(modelPath, true, cv::Size(640, 640));



    cv::Mat image;

    std::vector<Detection> result_ONNX;
    std::vector<Detection> result_CVDNN;
    std::vector<Detection> result_OV;


   
    


    std::cout << "Model was initialized." << std::endl;
    
    try
    {
        image = cv::imread(imagePath);

        result_ONNX = detector_usingonnx.detect(image, confThreshold, iouThreshold);

        result_CVDNN =  detector_usingcv.detect(image, confThreshold, iouThreshold);

        result_OV = detector_usingov.detect(image, confThreshold, iouThreshold);


        //utils::visualizeDetection(image, result_CVDNN, classNames, cv::Scalar(229, 160, 21));
        //utils::visualizeDetection(image, result_ONNX, classNames, cv::Scalar(21, 160,229));
        //utils::visualizeDetection(image, result_OV, classNames, cv::Scalar(229, 21, 229));


        //cv::imshow("result", image);
        //// cv::imwrite("result.jpg", image);
        //cv::waitKey(0);

        


        
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }


    auto start_ONNX = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 100; i++)
    {
        result_ONNX = detector_usingonnx.detect(image, confThreshold, iouThreshold);
    }

    utils::visualizeDetection(image, result_ONNX, classNames, cv::Scalar(229, 160, 21));
    auto end_ONNX = std::chrono::high_resolution_clock::now();
    auto time_ONNC =  std::chrono::duration_cast<std::chrono::milliseconds>(end_ONNX - start_ONNX).count();

    std::ostringstream time_label_ONNX;
    time_label_ONNX << std::fixed << std::setprecision(2);
    time_label_ONNX << "ONNX Detect Time= " << time_ONNC/100;
    std::string time_label_str_ONNX = time_label_ONNX.str();
    std::cout<< time_label_str_ONNX <<std::endl;


    auto start_CV = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 100; i++)
    {
        result_CVDNN = detector_usingcv.detect(image, confThreshold, iouThreshold);
    }

    utils::visualizeDetection(image, result_CVDNN, classNames, cv::Scalar(21, 160, 229));
    auto end_CV = std::chrono::high_resolution_clock::now();
    auto time_CV = std::chrono::duration_cast<std::chrono::milliseconds>(end_CV - start_CV).count();

    std::ostringstream time_label_CV;
    time_label_CV << std::fixed << std::setprecision(2);
    time_label_CV << "CVDNN Detect Time= " << time_CV / 100;
    std::string time_label_str_CV = time_label_CV.str();
    std::cout << time_label_str_CV << std::endl;


    auto start_OV = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 100; i++)
    {
        result_OV = detector_usingov.detect(image, confThreshold, iouThreshold);
    }

    utils::visualizeDetection(image, result_OV, classNames, cv::Scalar(160, 21, 229));
    auto end_OV = std::chrono::high_resolution_clock::now();
    auto time_OV = std::chrono::duration_cast<std::chrono::milliseconds>(end_OV - start_OV).count();

    std::ostringstream time_label_OV;
    time_label_OV << std::fixed << std::setprecision(2);
    time_label_OV << "OPENVINO Detect Time= " << time_OV / 100;
    std::string time_label_str_OV = time_label_OV.str();
    std::cout << time_label_str_OV << std::endl;

    cv::imshow("result", image);
    // cv::imwrite("result.jpg", image);
    cv::waitKey(0);



}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
