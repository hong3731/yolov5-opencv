#include "detector_cv.h"

void detector_cv::load_net(const std::string& modelPath, const bool& isGPU = true)
{

    cvdnnNET = cv::dnn::readNet("config_files/yolov5s.onnx");
    if (isGPU)
    {
        std::cout << "Attempty to use CUDA\n";
        cvdnnNET.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        cvdnnNET.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "Running on CPU\n";
        cvdnnNET.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        cvdnnNET.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

}

std:: vector<Detection> detector_cv::detect(cv::Mat& image, const float& confThreshold = 0.4,
    const float& iouThreshold = 0.45)
{
    //cv::Size& inputSize = cv::Size(640, 640);
    cv::Size2f inputImageShape = cv::Size2f(640,640);

    cv::Mat resizedImage, blob;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    utils::letterbox(resizedImage, resizedImage, inputImageShape,
        cv::Scalar(114, 114, 114), false,
        false, true, 32);




    cv::dnn::blobFromImage(resizedImage, blob, 1. / 255., inputImageShape, cv::Scalar(), false, false);
    cvdnnNET.setInput(blob);
    std::vector<cv::Mat> outputs;
    cvdnnNET.forward(outputs, cvdnnNET.getUnconnectedOutLayersNames());

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float x_factor = image.cols / INPUT_WIDTH;
    float y_factor = image.rows / INPUT_HEIGHT;
    float* data = (float*)outputs[0].data;

    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {

            float* classes_scores = data + 5;
            cv::Mat scores(1, classNamesize, CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) );
                int top = int((y - 0.5 * h) );
                int width = int(w );
                int height = int(h );
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }

        data += 85;

    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    std::vector<Detection> detections;

    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.classId = class_ids[idx];
        result.conf = confidences[idx];
        result.box = boxes[idx];
        utils::scaleCoords(resizedImage.size(), result.box, image.size());
        detections.emplace_back(result);
    }

   

    //for (int idx : indices)
    //{
    //    Detection det;
    //    det.box = cv::Rect(boxes[idx]);
    //    utils::scaleCoords(resizedImageShape, det.box, originalImageShape);

    //    det.conf = confs[idx];
    //    det.classId = classIds[idx];
    //    detections.emplace_back(det);
    //}

    return detections;



}



