#include "pch.h"
#include "mano.h"
#include "parser.hpp"
#include "Eigen.h"
#include "mano.h"
#include "ceres/ceres.h"
#include <iostream>
#include <fstream>
#include "camera.h"

struct EnergyCostFunction
{
	EnergyCostFunction(double pointX_, double pointY_, double weight_, HandModel hands_, const int iteration_, Hand LorR_)
		: pointX(pointX_), pointY(pointY_), weight(weight_), hands_to_optimize(hands_), i(iteration_), left_or_right(LorR_)
	{

	}

	template<typename T>
	bool operator()(const T* const shape, const T* const pose, T* residual) const
	{
		//camera settings for meshlab
		SimpleCamera meshlabcam;
		meshlabcam.f_x = (28.709295 * 1531) / 0.0369161;
		meshlabcam.f_y = (28.709295 * 898) / 0.0369161;
		meshlabcam.m_x = 765;
		meshlabcam.m_y = 449;

		//TODO: Bring the initialization outside
		HandModel testHand("mano/model/mano_right.json", "mano/model/mano_left.json");

		//create MANO surface thorugh setting shape and pose parameters for predefined Hand Model 
		testHand.setModelParameters(shape, pose, left_or_right);

		//transform MANO to OpenPose and Project to 2D given camera intrinsics
		std::array<std::array<double, 2>, NUM_OPENPOSE_KEYPOINTS> hand_projected = testHand.get2DJointLocations(left_or_right, meshlabcam);

		//simple, weighted L1 norm
		residual[0] = T(weight) * (hand_projected[i][0] - T(pointX) + (hand_projected[i][1] - T(pointY)));

		//reset hand shape
		testHand.reset();

		return true;
	}

private:
	double pointX;
	double pointY;
	double weight;
	HandModel hands_to_optimize;
	const int i;
	Hand left_or_right; 
};

int main(int argc, char** argv)
{
	// Read OpenPose keypoints
	std::string filename;
	if (argc > 1) {
		filename = argv[1];
	}
	else {
		filename = "samples/webcam_examples/000000000000_keypoints.json";
	}

	std::array<std::array<double, NUM_KEYPOINTS * 3>, 2> keypoints = Parser::readJsonCV(filename);

	std::array<double, NUM_KEYPOINTS * 3> left_keypoints = keypoints[0];
	std::array<double, NUM_KEYPOINTS * 3> right_keypoints = keypoints[1];

	//std::array<double, NUM_KEYPOINTS * 3> left_keypoints;
	//std::array<double, NUM_KEYPOINTS * 3> right_keypoints;
	
	// Define initial values for parameters of pose and shape 
	const VectorXf poseInitial = VectorXf::Random(MANO_THETA_SIZE);
	const VectorXf shapeInitial = VectorXf::Random(MANO_BETA_SIZE);

	// Assign initial values to parameters
	//VectorXf pose = poseInitial;
	//VectorXf shape = shapeInitial;
	std::array<double, MANO_THETA_SIZE> pose;
	std::array<double, MANO_BETA_SIZE> shape;
	//double shape;
	//double pose;

	//create initial HandModel for further optimization
	HandModel hands_to_optimize("mano/model/mano_right.json", "mano/model/mano_left.json");

	// FOR TESTING: we use only the right hand 
	Hand left_or_right = Hand::RIGHT;

	ceres::Problem problem;

	// Residual block for right hand 
	for (int i = 0; i < NUM_KEYPOINTS; ++i)
	{
		ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<EnergyCostFunction, 1, 1, 1>(
				new EnergyCostFunction(right_keypoints[3 * i], right_keypoints[3 * i + 1], right_keypoints[3 * i + 2], hands_to_optimize, i, left_or_right));
		problem.AddResidualBlock(cost_function, nullptr, &shape[0], &pose[0]);

		//problem.AddResidualBlock(
		//	new ceres::AutoDiffCostFunction<EnergyCostFunction, 1, 1, 1>(
		//		new EnergyCostFunction(right_keypoints[3 * i], right_keypoints[3 * i + 1], right_keypoints[3 * i + 2], hands_to_optimize, i, left_or_right)),
		//	nullptr, &shape[0], &pose[0]);
	}

	ceres::Solver::Options options;
	options.max_num_iterations = 25;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << std::endl;

	// Output the final pose and shape

	std::cout << "Initial pose: " << poseInitial << "shape: " << shapeInitial << std::endl;
	//std::cout << "Final pose: " << pose << "shape: " << shape << std::endl;

	system("pause");



	//Run solver and output - trivial

	return 0;
}