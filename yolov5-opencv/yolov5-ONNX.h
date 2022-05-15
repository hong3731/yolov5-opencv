#include <onnxruntime/core/session/onnxruntime_cxx_api.h>


class detector
{
public:
	detector();
	~detector();

public:
	void setNumThread(int numOfThread);
	void initModel(const std::string& pathStr);
	void detecrun();

private:
	Ort::Session* session;
	Ort::Env env = Ort::Env();// ORT_LOGGING_LEVEL_ERROR, "Yolov5");
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();

	char* inputName;
	char* outputName;



};

detector::detector()
{
}

detector::~detector()
{
}
