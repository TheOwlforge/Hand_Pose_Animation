#pragma once
#include "pch.h"
#include "json.hpp"

#define NUM_KEYPOINTS 21

class Parser
{
	using json = nlohmann::json;

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

	/**
	Input: Filename (.json)
	Output: Left and Right Keypoints (#keypoints = 20) (Format: x, y, c)
	*/
	static inline std::array<std::array<float, NUM_KEYPOINTS * 3>, 2> readJsonCV(std::string filename)
	{
		// Read and parse json file
		std::ifstream keypoints_file(filename);
		json js = json::parse(keypoints_file);

		// Assign keypoints to float array
		std::array<float, NUM_KEYPOINTS * 3> left_keypoints = js["people"][0]["hand_left_keypoints_2d"];
		std::array<float, NUM_KEYPOINTS * 3> right_keypoints = js["people"][0]["hand_right_keypoints_2d"];

		std::array<std::array<float, NUM_KEYPOINTS * 3>, 2> keypoints = { left_keypoints, right_keypoints };

		return keypoints;
	}

	static inline void printKeypoints(std::array<std::array<float, NUM_KEYPOINTS * 3>, 2> keypoints)
	{
		std::array<float, NUM_KEYPOINTS * 3> left_keypoints = keypoints[0];
		std::array<float, NUM_KEYPOINTS * 3> right_keypoints = keypoints[1];

		// Print each of left/right keypoints: x0, y0, c0, x1, y1, c1, x2, y2, c2, ..., x20, y20, c20

		std::cout << "Left keypoints:" << std::endl;

		for (int i = 0; i < NUM_KEYPOINTS; i++) {
			std::cout << i << " : " << left_keypoints[3 * i] << "," << left_keypoints[3 * i + 1] << "," << left_keypoints[3 * i + 2] << std::endl; // (x,y, confidence_score)
		}
		std::cout << std::endl << "Right keypoints:" << std::endl;

		for (int i = 0; i < NUM_KEYPOINTS; i++) {
			std::cout << i << " : " << right_keypoints[3 * i] << "," << right_keypoints[3 * i + 1] << "," << right_keypoints[3 * i + 2] << std::endl; // (x,y, confidence_score)
		}
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

	static inline void saveNextFrameCV(cv::VideoWriter& video, cv::Mat& frame)
	{
		video.write(frame);
	}

	static void testCV()
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
			saveNextFrameCV(out, frame);
			cv::waitKey(0);
		}

		cv::destroyAllWindows();
	}

	static void testJson() {
		std::string filename = "samples/webcam_examples/000000000000_keypoints.json";

		std::array<std::array<float, NUM_KEYPOINTS * 3>, 2> keypoints = readJsonCV(filename);

		printKeypoints(keypoints);
	}

private:
	static std::vector<cv::VideoCapture> caps;
	static std::vector<cv::VideoWriter> videos;
};

std::vector<cv::VideoCapture> Parser::caps;
std::vector<cv::VideoWriter> Parser::videos;