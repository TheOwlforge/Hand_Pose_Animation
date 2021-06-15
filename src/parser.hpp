#pragma once
#include "pch.h"

class Parser {
public:
	~Parser()
	{
		for (uint8_t i = 0; i < this->caps.size(); i++)
			caps[i].release();
		for (uint8_t i = 0; i < this->videos.size(); i++)
			videos[i].release();
	}

	static inline cv::Mat readImageCV(const char* filename)
	{
		return cv::imread(filename, cv::IMREAD_COLOR);
	}

	template <typename T>
	static inline T* readImageRaw(const char* filename)
	{
		return (T*)cv::imread(filename).data;
	}

	static inline void saveImageCV(const char* filename, cv::Mat& mat)
	{
		cv::imwrite(filename, mat);
	}

	static inline bool readVideoCV(const char* filename, cv::VideoCapture& cap)
	{
		cap = cv::VideoCapture(filename);
		if (cap.isOpened())
		{
			Parser::caps.push_back(cap);
			return true;
		}
		std::cout << "Error opening video stream or file" << std::endl;
		return false;
	}

	static inline bool getNextFrameCV(cv::VideoCapture& cap, cv::Mat& frame)
	{
		cap >> frame;
		if (!frame.empty())
			return true;
		std::cout << "Error reading next frame" << std::endl;
		return false;
	}

	template <typename T>
	static inline bool getNextFrameRaw(cv::VideoCapture& cap, T* frame)
	{
		cv::Mat frame;
		if(getNextFrameCV(cap, frame))
		{
			frame = (T*)frame.data;
			return true;
		}
		return false;
	}

	static inline void saveVideoCV(const char* filename, cv::VideoCapture& cap, int framerate = 10)
	{
		cv::VideoWriter video = createVideoCV(filename, cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT), framerate);
		cv::Mat frame;
		while (getNextFrameCV(cap, frame)) {
			video.write(frame);
		}
	}

	static inline cv::VideoWriter createVideoCV(const char* filename, int frame_width, int frame_height, int framerate = 10)
	{
		cv::VideoWriter video(filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), framerate, cv::Size(frame_width, frame_height));
		Parser::videos.push_back(video);
		return video;
	}

	static inline void saveNextFrame(cv::VideoWriter& video, cv::Mat& frame)
	{
		video.write(frame);
	}

	static void test()
	{
		cv::imshow("test image", readImageCV("data/test1.png"));
		cv::waitKey(0);

		cv::VideoCapture cap;
		readVideoCV("data/test.mp4", cap);

		cv::VideoWriter out = createVideoCV("out.avi", cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));

		cv::Mat frame;
		while (getNextFrameCV(cap, frame)) {
			cv::imshow("test video", frame);
			cv::Mat flipped;
			cv::flip(frame, flipped, 1);
			saveNextFrame(out, frame);
			cv::waitKey(0);
		}

		cv::destroyAllWindows();
	}

private:
	static std::vector<cv::VideoCapture> caps;
	static std::vector<cv::VideoWriter> videos;
};

std::vector<cv::VideoCapture> Parser::caps;
std::vector<cv::VideoWriter> Parser::videos;