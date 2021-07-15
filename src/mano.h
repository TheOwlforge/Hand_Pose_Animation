#pragma once

constexpr int NUM_MANO_VERTICES = 778;
constexpr int NUM_MANO_JOINTS = 16;
constexpr int NUM_MANO_FACES = 1538;
constexpr int MANO_THETA_SIZE = (NUM_MANO_JOINTS - 1) * 3;
constexpr int MANO_BETA_SIZE = 10;
constexpr int MANO_R_SIZE = (NUM_MANO_JOINTS - 1) * 9;
constexpr int NUM_OPENPOSE_KEYPOINTS = 21;

constexpr int ADDITIONAL_JOINTS[5] = { 744, 320, 443, 554, 671 };

enum class Hand {
	RIGHT,
	LEFT
};

struct ManoHand
{
	const std::array<Eigen::Vector3f, NUM_MANO_VERTICES> const T; // zero pose
	const Eigen::Matrix<float, NUM_MANO_VERTICES, NUM_MANO_JOINTS> W; // blend weights
	const std::array<Eigen::Vector3f, NUM_MANO_JOINTS> const J; // joints

	const std::array<Eigen::Vector3i, NUM_MANO_FACES> const face_indices;

	const std::map<unsigned int, unsigned int>* const kinematic_tree; // 16 entries -> parent for each joint
	const Eigen::Matrix<float, NUM_MANO_JOINTS, NUM_MANO_VERTICES> joint_regressor;

	/*const std::array<float, 1554 * 45> hands_coefficients;
	const std::array<float, 45 * 45> hands_components;
	const std::array<float, 45> hands_mean;*/

	const Eigen::Matrix<float, NUM_MANO_VERTICES * 3, MANO_R_SIZE> pose_blend_shapes; // dimension 778 * 3 * 135
	const Eigen::Matrix<float, NUM_MANO_VERTICES * 3, MANO_BETA_SIZE> shape_blend_shapes; // dimension 778 * 3 * 10

	std::array<Eigen::Vector4f, NUM_MANO_VERTICES> vertices; // current position in NDC
	std::array<Eigen::Vector4f, NUM_MANO_JOINTS> joints; // current joint positions
	std::array<float, MANO_THETA_SIZE> theta; // pose parameters
	std::array<float, MANO_BETA_SIZE> beta; // shape parameters

	std::array<Eigen::Matrix4f, NUM_MANO_JOINTS> inv_G_zero;
	std::array<Eigen::Vector<float, 9>, NUM_MANO_JOINTS - 1> R_zero;
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

	void setModelParameters(std::array<float, MANO_THETA_SIZE> theta, std::array<float, MANO_BETA_SIZE> beta, Hand hand);

	std::array<std::array<float, 2>, NUM_OPENPOSE_KEYPOINTS> get2DJointLocations(Hand hand);
	void applyTransformation(Eigen::Vector3f t, Hand hand);

	bool saveVertices();
	bool saveMANOJoints();
	bool saveOPJoints();

	template <int T>
	static void fillRandom(std::array<float, T>* test, int bounds)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> distrib(0, 1);

		for (auto& val : *test) {
			val = distrib(gen) * 2 * bounds - bounds;
		}
	}

private:
	Eigen::Matrix3f rodrigues(Eigen::Vector3f w);
	bool hasAncestor(ManoHand* h, unsigned int joint_index);
	unsigned int getAncestor(ManoHand* h, unsigned int joint_index);
	Eigen::Matrix4f computeG(ManoHand* h, unsigned int joint_index);
	Eigen::Vector<float, 9> computeR(ManoHand* h, unsigned int joint_index);
};