#pragma once
#include "camera.h"

constexpr int NUM_MANO_VERTICES = 778;
constexpr int NUM_MANO_JOINTS = 16;
constexpr int NUM_MANO_FACES = 1538;
constexpr int MANO_THETA_SIZE = NUM_MANO_JOINTS * 3;
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
	//Eigen::Matrix<float, x, y> is allocated on the stack -> not good for large matrices
	//use MatrixXf instead, which is dynamic in size (which we don't really need as all dimensions are known), but allocated on heap

	const std::array<Eigen::Vector3f, NUM_MANO_VERTICES> T; // zero pose
	const Eigen::MatrixXf W; // blend weights of dimension 778 * 16
	const std::array<Eigen::Vector3f, NUM_MANO_JOINTS> J; // joints

	const std::array<Eigen::Vector3i, NUM_MANO_FACES> face_indices;

	const std::map<unsigned int, unsigned int>* const kinematic_tree; // 16 entries -> parent for each joint
	const Eigen::MatrixXf joint_regressor; // dimension 16 * 778

	/*const std::array<float, 1554 * 45> hands_coefficients;
	const std::array<float, 45 * 45> hands_components;*/
	const std::array<float, (NUM_MANO_JOINTS - 1) * 3> hands_mean;

	const Eigen::MatrixXf pose_blend_shapes; // dimension (778 * 3) * 135 (-> 15 * 9 = 135)
	const Eigen::MatrixXf shape_blend_shapes; // dimension (778 * 3) * 10

	std::array<Eigen::Vector4f, NUM_MANO_VERTICES> vertices; // current position in NDC
	std::array<Eigen::Vector4f, NUM_MANO_JOINTS> joints; // current joint positions
	std::array<double, MANO_THETA_SIZE> theta; // pose parameters
	std::array<double, MANO_BETA_SIZE> beta; // shape parameters

	std::array<Eigen::Matrix4f, NUM_MANO_JOINTS> inv_G_zero;
	std::array<Eigen::Vector<float, 9>, NUM_MANO_JOINTS - 1> R_zero;

	ManoHand(std::array<Eigen::Vector3f, NUM_MANO_VERTICES>& t,
		Eigen::MatrixXf& w,
		std::array<Eigen::Vector3f, NUM_MANO_JOINTS>& j,
		std::array<Eigen::Vector3i, NUM_MANO_FACES>& f,
		std::map<unsigned int, unsigned int>* kt,
		Eigen::MatrixXf& jr,
		std::array<float, (NUM_MANO_JOINTS - 1) * 3> hm,
		Eigen::MatrixXf& p,
		Eigen::MatrixXf& s)
		: T(t), J(j), W(w), face_indices(f), kinematic_tree(kt), joint_regressor(jr), hands_mean(hm), pose_blend_shapes(p), shape_blend_shapes(s)
	{ }

	~ManoHand()
	{
		// kinematic tree had to be pointer, otherwise values weren't passed on correctly
		delete kinematic_tree;
	}
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

	template<typename T>
	void setModelParameters(const T* const theta, const T* const beta, Hand hand);
	void reset();

	std::array<std::array<double, 2>, NUM_OPENPOSE_KEYPOINTS> get2DJointLocations(Hand hand, SimpleCamera camera);
	std::array<std::array<float, 2>, NUM_MANO_VERTICES> get2DVertexLocations(Hand hand, SimpleCamera camera);
	void applyTranslation(Eigen::Vector3f t, Hand hand);
	void applyScale(float factor, Hand hand);
	void applyRotation(float alpha, float beta, float gamma, Hand hand); //for testing purposes only, not meant to be used later as global rotation is included in theta

	bool saveVertices();
	bool saveMANOJoints();
	bool saveOPJoints();

	void display(const char* filename, Hand hand);

	template <int T>
	static void fillRandom(std::array<float, T>* test, float bounds)
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
	void applyTransformation(Eigen::Matrix4f transform, Hand hand);
};