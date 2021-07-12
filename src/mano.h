#pragma once

constexpr int NUM_MANO_VERTICES = 778;
constexpr int NUM_MANO_JOINTS = 16;
constexpr int NUM_MANO_FACES = 1538;

enum class Hand {
	RIGHT,
	LEFT
};

struct ManoHand
{
	std::array<Eigen::Vector3f, NUM_MANO_VERTICES> vertices; // current position

	std::array<Eigen::Vector3f, NUM_MANO_VERTICES> T; // zero pose
	Eigen::Matrix<float, NUM_MANO_VERTICES, NUM_MANO_JOINTS> W; // blend weights
	std::array<Eigen::Vector3f, NUM_MANO_JOINTS> J; // joints

	std::array<Eigen::Vector3i, NUM_MANO_FACES> face_indices;

	std::array<float, 1554 * 45> hands_coefficients;
	std::array<float, 45 * 45> hands_components;
	std::array<float, 45> hands_mean;

	std::map<unsigned int, unsigned int> kinematic_tree; // 16 entries -> parent for each joint
	Eigen::Matrix<float, NUM_MANO_JOINTS, NUM_MANO_VERTICES> joint_regressor; 


	std::vector<float> pose_blend_shapes; // dimension 778 * 3 * 135
	std::vector<float> shape_blend_shapes; // dimension 778 * 3 * 10
};



class HandModel
{
public:
	ManoHand* rightHand;
	ManoHand* leftHand;

	HandModel(std::string rightHand_filename, std::string leftHand_filename);
	void applyTransformation(Eigen::Vector3f t, Hand hand);
	bool saveToObj(std::string output_filename);
};