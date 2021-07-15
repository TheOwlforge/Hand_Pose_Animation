#pragma once
#include "json.hpp"
#include "mano.h"

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
			std::cout << i << " : " << left_keypoints[3 * i] << ", " << left_keypoints[3 * i + 1] << ", " << left_keypoints[3 * i + 2] << std::endl; // (x,y, confidence_score)
		}
		std::cout << std::endl << "Right keypoints:" << std::endl;

		for (int i = 0; i < NUM_KEYPOINTS; i++) {
			std::cout << i << " : " << right_keypoints[3 * i] << ", " << right_keypoints[3 * i + 1] << ", " << right_keypoints[3 * i + 2] << std::endl; // (x,y, confidence_score)
		}
		std::cout << std::endl;
	}

	static inline ManoHand* readJsonMANO(std::string filename)
	{
		std::cout << "Opening " << filename << std::endl;

		auto start = std::chrono::high_resolution_clock::now();

		// Read json file
		std::ifstream mano_file(filename);
		json js = json::parse(mano_file);

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> duration = end - start;
		std::cout << "...took " << duration.count() << " ms." << std::endl << std::endl;

		// parse json file
		std::array<Eigen::Vector3f, NUM_MANO_VERTICES> vertices_template;
		Eigen::MatrixXf weights(NUM_MANO_VERTICES, NUM_MANO_JOINTS);
		std::array<Eigen::Vector3f, NUM_MANO_JOINTS> joints;
		std::array<Eigen::Vector3i, NUM_MANO_FACES> face_indices;
		std::map<unsigned int, unsigned int>* kinematic_tree = new std::map<unsigned int, unsigned int>();
		Eigen::MatrixXf joint_regressor(NUM_MANO_JOINTS, NUM_MANO_VERTICES);
		Eigen::MatrixXf pose_blend_shapes(NUM_MANO_VERTICES * 3, (NUM_MANO_JOINTS - 1) * 9);
		Eigen::MatrixXf shape_blend_shapes(NUM_MANO_VERTICES * 3, MANO_BETA_SIZE);

		for (uint16_t i = 0; i < js["face_indices"].size(); i++)
		{
			face_indices[i] = Eigen::Vector3i(js["face_indices"][i][0], js["face_indices"][i][1], js["face_indices"][i][2]);
		}
		for (uint16_t i = 0; i < js["vertices_template"].size(); i++)
		{
			vertices_template[i] = Eigen::Vector3f(js["vertices_template"][i][0], js["vertices_template"][i][1], js["vertices_template"][i][2]);
		}
		for (uint16_t i = 0; i < js["joints"].size(); i++)
		{
			joints[i] = Eigen::Vector3f(js["joints"][i][0], js["joints"][i][1], js["joints"][i][2]);
		}
		for (uint16_t j = 0; j < js["weights"].size(); j++)
		{
			for (uint16_t i = 0; i < js["weights"][j].size(); i++)
			{
				weights(j, i) = js["weights"][j][i];
			}
		}
		for (uint16_t j = 0; j < js["joint_regressor"].size(); j++)
		{
			for (uint16_t i = 0; i < js["joint_regressor"][j].size(); i++)
			{
				joint_regressor(j, i) = js["joint_regressor"][j][i];
			}
		}

		for (uint16_t j = 0; j < NUM_MANO_VERTICES; j++)
		{
			for (uint16_t i = 0; i < MANO_R_SIZE; i++)
			{
				pose_blend_shapes(3 * j, i) = js["pose_blend_shapes"][j][0][i];
				pose_blend_shapes(3 * j + 1, i) = js["pose_blend_shapes"][j][1][i];
				pose_blend_shapes(3 * j + 2, i) = js["pose_blend_shapes"][j][2][i];
			}
		}
		for (uint16_t j = 0; j < NUM_MANO_VERTICES; j++)
		{
			for (uint16_t i = 0; i < MANO_BETA_SIZE; i++)
			{
				shape_blend_shapes(3 * j, i) = js["shape_blend_shapes"][j][0][i];
				shape_blend_shapes(3 * j + 1, i) = js["shape_blend_shapes"][j][1][i];
				shape_blend_shapes(3 * j + 2, i) = js["shape_blend_shapes"][j][2][i];
			}
		}
		// std::array<unsigned int, 16> value = js["kinematic_tree"][0];
		// std::array<unsigned int, 16> key = js["kinematic_tree"][1];
		for (uint16_t i = 0; i < js["kinematic_tree"][0].size(); i++)
		{
			(*kinematic_tree)[js["kinematic_tree"][1][i]] = js["kinematic_tree"][0][i];
		}

		ManoHand* result = new ManoHand(vertices_template, weights, joints, face_indices, kinematic_tree, joint_regressor, pose_blend_shapes, shape_blend_shapes);

		return result;
	}

	static inline void print_map(const std::map<unsigned int, unsigned int>& m)
	{
		for (const auto& [key, value] : m) {
			std::cout << key << " = " << value << "; ";
		}
		std::cout << "\n";
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
		if (getNextFrameCV(cap, frame))
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