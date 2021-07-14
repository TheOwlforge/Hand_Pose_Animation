#pragma once

constexpr int NUM_MANO_VERTICES = 778;
constexpr int NUM_MANO_JOINTS = 16;
constexpr int NUM_MANO_FACES = 1538;
constexpr int MANO_THETA_SIZE = (NUM_MANO_JOINTS - 1) * 3;
constexpr int MANO_BETA_SIZE = 10;
constexpr int MANO_R_SIZE = (NUM_MANO_JOINTS - 1) * 9;
constexpr int NUM_OPENPOSE_KEYPOINTS = 21;

enum class Hand {
	RIGHT,
	LEFT
};

struct ManoHand
{
	const std::array<Eigen::Vector3d, NUM_MANO_VERTICES> T; // zero pose
	const Eigen::Matrix<double, NUM_MANO_VERTICES, NUM_MANO_JOINTS> W; // blend weights
	const std::array<Eigen::Vector3d, NUM_MANO_JOINTS> J; // joints

	const std::array<Eigen::Vector3i, NUM_MANO_FACES> face_indices;

	const std::map<unsigned int, unsigned int>* const kinematic_tree; // 16 entries -> parent for each joint
	const Eigen::Matrix<double, NUM_MANO_JOINTS, NUM_MANO_VERTICES> joint_regressor;

	/*const std::array<float, 1554 * 45> hands_coefficients;
	const std::array<float, 45 * 45> hands_components;
	const std::array<float, 45> hands_mean;*/

	const Eigen::Matrix<double, NUM_MANO_VERTICES * 3, MANO_R_SIZE> pose_blend_shapes; // dimension 778 * 3 * 135
	const Eigen::Matrix<double, NUM_MANO_VERTICES * 3, MANO_BETA_SIZE> shape_blend_shapes; // dimension 778 * 3 * 10

	std::array<Eigen::Vector4d, NUM_MANO_VERTICES> vertices; // current position in NDC
	std::array<Eigen::Vector3d, NUM_MANO_JOINTS> joints; // current joint positions
	std::array<double, MANO_THETA_SIZE> theta;
	Eigen::Vector<double, MANO_BETA_SIZE> beta;

	std::array<Eigen::Matrix4d, NUM_MANO_JOINTS> inv_G_zero;
	std::array<Eigen::Vector<double, 9>, NUM_MANO_JOINTS - 1> R_zero;
};



class HandModel
{
public:
	ManoHand* rightHand;
	ManoHand* leftHand;

	bool isVisible_left;
	bool isVisible_right;

	HandModel(std::string rightHand_filename, std::string leftHand_filename);
	~HandModel();

	void setTheta(std::array<double, MANO_THETA_SIZE> theta, Hand hand);
	void setBeta(std::array<double, MANO_BETA_SIZE> theta, Hand hand);
	std::array<Eigen::Vector3d, NUM_OPENPOSE_KEYPOINTS> getJointLocations(Hand hand);
	void applyTransformation(Eigen::Vector3d t, Hand hand);
	bool saveToObj(std::string output_filename);

private:
	Eigen::Matrix3d rodrigues(Eigen::Vector3d w);
	bool hasAncestor(ManoHand* h, unsigned int joint_index);
	unsigned int getAncestor(ManoHand* h, unsigned int joint_index);
	Eigen::Matrix4d computeG(ManoHand* h, unsigned int joint_index);
	Eigen::Vector<double, 9> computeR(ManoHand* h, unsigned int joint_index);
};