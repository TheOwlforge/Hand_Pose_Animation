#include "pch.h"
#include "mano.h"
#include "parser.hpp"
#include "Eigen.h"
#include "mano.h"
#include "ceres/ceres.h"
#include <iostream>
#include <fstream>

#define PRIOR_COEFF_EST 0.5  // Balancing term for surface estimation difference prior
#define PRIOR_COEFF_SHAPE 0.5 // Balancing term for difference between shape parameters and mean estimation of shape parameters
#define PRIOR_COEFF_POSE 0.5 // Balancing term for pose prior


struct EnergyCostFunction
{
    EnergyCostFunction(const float pointX_, const float pointY_, const float weight_, HandModel hands_, const int iteration_, const Hand LorR_, const VectorXf mean_shape_)
      : pointX(pointX_), pointY(pointY_), weight(weight_), hands_to_optimize(hands_), i(iteration_), left_or_right(LorR_), mean_shape(mean_shape_)
	{

	}

	template<typename T>
	bool operator()(const T* const shape, const T* const pose, T* residual) const
	{
		//create MANO surface thorugh setting shape and pose parameters for predefined Hand Model 
		hands_to_optimize.setModelParameters(shape, pose, left_or_right);
		//transform MANO to OpenPose and Project to 2D given 
		std::array<std::array<double, 2>, NUM_OPENPOSE_KEYPOINTS> hand_projected = hands.get2DJointsLocations(left_or_right);

		//simple, weighted L1 norm
		residual[0] = weight * ((hand_projected[i][0] - pointX) + (hand_projected[i][1] - pointY));

		// Priors

		// Prior 1: Make previous hand projections closer to the previous one.

		if num_sequences > 0:
  		    residual[0] += PRIOR_COEFF_EST * pow((hand_projected[i][0] - prev_surface_est[i][0]) + (hand_projected[i][1] - prev_surface_est[i][0]), 2).sum();

		// Prior 2: Make shape parameters closer to mean	    

		residual[0] += PRIOR_COEFF_SHAPE * pow(shape - mean_shape, 2).sum();

		// Prior 3: Gaussian pose prior

		residual[0] += PRIOR_COEFF_POSE * pow(pose, 2).sum();

		/*
		  Optional:  Temporal Regularizer: zero-velocity prior (Real-time Pose and Shape Reconstruction of Two Interacting Hands With a Single Depth Camera)

		  residual[0] += pow(shape - prev_shape, 2).sum() + pow(pose - prev_pose, 2).sum()

		  prev_pose = pose;
		  prev_shape = shape;

		 */

		prev_surface_est = hand_projected; // save current estimations for next iteration
		mean_shape = (mean_shape * num_sequences + shape) / (num_sequences + 1);  // new mean shape after adding shape params in current iteration
		num_sequences++;

		return true;
	}

private:
	const float pointX;
	const float pointY;
	const float weight;
	HandModel hands_to_optimize;
	const int i;
	const Hand left_or_right;
        std::array<std::array<double, 2>, NUM_OPENPOSE_KEYPOINTS> prev_surface_est;
        VectorXf mean_shape;
        int num_sequences;
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

	std::array<std::array<float, NUM_KEYPOINTS * 3>, 2> keypoints = Parser::readJsonCV(filename);

	std::array<float, NUM_KEYPOINTS * 3> left_keypoints = keypoints[0];
	std::array<float, NUM_KEYPOINTS * 3> right_keypoints = keypoints[1];

	num_sequences = 0;
	mean_shape = VectorXf::Zero(MANO_BETA_SIZE);


	//TODO: SOLVER!

	/*
	// Define initial values for parameters of pose and shape 
	const VectorXf poseInitial = VectorXf::Random(MANO_THETA_SIZE);
	const VectorXf shapeInitial = VectorXf::Random(MANO_BETA_SIZE);

	// Assign initial values to parameters
	VectorXf pose = poseInitial;
	VectorXf shape = shapeInitial;

	//create initial HandModel for further optimization
	HandModel hands_to_optimize("mano/model/mano_right.json", "mano/model/mano_left.json");

	// FOR TESTING: we use only the right hand 
	Hand left_or_right = Hand::RIGHT;



	ceres::Problem problem;

	// Residual block for right hand 
	for (int i = 0; i < NUM_KEYPOINTS; ++i)
	{
		//CostFunction* cost_function =
		//	new ceres::AutoDiffCostFunction<EnergyCostFunction, 1, 1, 1>(
		//		new EnergyCostFunction(right_keypoints[3 * i], right_keypoints[3 * i + 1], right_keypoints[3 * i + 2], hands_to_optimize, i, left_or_right));
		//problem.AddResidualBlock(cost_function, nullptr, &shape, &pose);


		problem.AddResidualBlock(
			new ceres::AutoDiffCostFunction<EnergyCostFunction, 1, 1, 1>(
				new EnergyCostFunction(right_keypoints[3 * i], right_keypoints[3 * i + 1], right_keypoints[3 * i + 2], hands_to_optimize, i, left_or_right)),
			nullptr, &shape, &pose);//VECTORX IST EIN PROBLEM!!!!
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
	std::cout << "Final pose: " << pose << "shape: " << shape << std::endl;

	system("pause");

	*/


	//Run solver and output - trivial

	return 0;
}
