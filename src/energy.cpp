#include "pch.h"
#include "mano.h"
#include "parser.hpp"
#include "Eigen.h"
#include "mano.h"
#include "ceres/ceres.h"
#include <iostream>
#include <fstream>
#include "camera.h"
#include "energy.h"

struct EnergyCostFunction
{
	EnergyCostFunction(std::array<double, NUM_KEYPOINTS> pointX_, std::array<double, NUM_KEYPOINTS> pointY_, std::array<double, NUM_KEYPOINTS> weight_, HandModel hands_, Hand LorR_)
		: pointX(pointX_), pointY(pointY_), weight(weight_), hands_to_optimize(hands_), left_or_right(LorR_)
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

		SimpleCamera denniscam;

		//TODO: Bring the initialization outside
		HandModel testHand("mano/model/mano_right.json", "mano/model/mano_left.json");

		std::cout << "Pose:typeid:" << pose[5] << std::endl;

		//std::array<double, MANO_THETA_SIZE> rnd1 = std::array<double, MANO_THETA_SIZE>();
		//std::array<double, MANO_BETA_SIZE> rnd2 = std::array<double, MANO_BETA_SIZE>();

		//create MANO surface thorugh setting shape and pose parameters for predefined Hand Model 
		std::cout << "Setting the model parameters pose and shape..." << std::endl;
		/*std::cout << "Pose: ";
		for (int i = 0; i < MANO_THETA_SIZE; i++)
		{
			std::cout << pose[i] << " ";
		}
		std::cout << std::endl;*/
		
		testHand.setModelParameters((double*)shape, (double*)pose, left_or_right);
		//testHand.setModelParameters(sh, po, left_or_right);
		//testHand.setModelParameters(rnd1.data(), rnd2.data(), Hand::RIGHT);
		//testHand.setModelParameters((double*)constrnd1, (double*)constrnd2, Hand::RIGHT);

		//Translate and rotate (proper for onehand1 dataset only!!!
	        testHand.applyRotation(0.5 * M_PI, 0, M_PI, Hand::RIGHT);
		testHand.applyTranslation(Eigen::Vector3f(0.03, 0.11, 1.8), Hand::RIGHT);

		//transform MANO to OpenPose and Project to 2D given camera intrinsics
		std::cout << "Transforming to OpenPose in 2D..." << std::endl;
		std::array<std::array<double, 2>, NUM_OPENPOSE_KEYPOINTS> hand_projected = testHand.get2DJointLocations(left_or_right, denniscam);

		std::cout << "Projected hand keypoints: ";
		for (int i = 0; i < NUM_OPENPOSE_KEYPOINTS; i++)
		{
			std::cout << hand_projected[i][0] << " ";
		}
		std::cout << std::endl;

		//simple, weighted L1 norm
		//std::cout << "Computing the residual with: weight: " << weight << " keypoint: " << pointX << " predicted: " << hand_projected[0][0] << std::endl;

		residual[0] = T(0);
		//residual[0] += T(pose[0]) - T(100);
		for (int i = 0; i < NUM_KEYPOINTS; i++) {
		   residual[0] += T(weight[i]) * (T(hand_projected[i][0]) - T(pointX[i]) + (T(hand_projected[i][1]) - T(pointY[i])));
		    std::cout << "Computing the residual with: weight: " << weight[i] << " keypoint: " << pointX[i] << " predicted: " << hand_projected[i][0] << std::endl;
		    std::cout << T(weight[i]) * (T(hand_projected[i][0]) - T(pointX[i]) + (T(hand_projected[i][1]) - T(pointY[i]))) << std::endl;
		}

		std::cout << "Residual: " << residual[0] << std::endl;

		//reset hand shape
		testHand.reset();

		return true;
	}

private:
	std::array<double, NUM_KEYPOINTS> pointX;
	std::array<double, NUM_KEYPOINTS> pointY;
	std::array<double, NUM_KEYPOINTS> weight;
	HandModel hands_to_optimize;
	Hand left_or_right; 
};

void runEnergy()
{
	// Read OpenPose keypoints
	std::string filename;
	//filename = "samples/artificial/01/keypoints01.json";
	filename = "samples/pictures/onehand1_keypoints.json";


	std::array<std::array<double, NUM_KEYPOINTS * 3>, 2> keypoints = Parser::readJsonCV(filename);

	std::array<double, NUM_KEYPOINTS * 3> left_keypoints = keypoints[0];
	std::array<double, NUM_KEYPOINTS * 3> right_keypoints = keypoints[1];

	std::array<double, NUM_KEYPOINTS> weights;
	std::array<double, NUM_KEYPOINTS> x_values;
	std::array<double, NUM_KEYPOINTS> y_values;

	for (int i = 0; i < NUM_KEYPOINTS; i++) {
	  x_values[i] = right_keypoints[3 * i];
	  y_values[i] = right_keypoints[3 * i + 1];
	  weights[i] = right_keypoints[3 * i + 2];
	}


	// Define initial values for parameters of pose and shape 
	std::array<double, MANO_THETA_SIZE> poseInitial = std::array<double, MANO_THETA_SIZE>();;
	std::array<double, MANO_BETA_SIZE> shapeInitial = std::array<double, MANO_BETA_SIZE>();;
	//HandModel::fillRandom(&poseInitial, 0.5f);
	//HandModel::fillRandom(&shapeInitial, 0.5f);

	// Assign initial values to parameters
	std::array<double, MANO_THETA_SIZE> pose = poseInitial;
	std::array<double, MANO_BETA_SIZE> shape = shapeInitial;

	for (int i = 0; i < MANO_THETA_SIZE; i++) {
	  pose[i] = (i + 10) / 2.0;
	}

	for (int i = 0; i < MANO_BETA_SIZE; i++) {
	  shape[i] = (i + 10) / 2.0;
	}

	//create initial HandModel for further optimization
	HandModel hands_to_optimize("mano/model/mano_right.json", "mano/model/mano_left.json");

	// FOR TESTING: we use only the right hand 
	Hand left_or_right = Hand::RIGHT;
	pose = hands_to_optimize.getMeanShape(left_or_right);
	pose[5] = 0.8;
	shape[5] = 0.4;

	ceres::Problem problem;

	// Residual block for right hand 

	ceres::CostFunction* cost_function =
	  new ceres::AutoDiffCostFunction<EnergyCostFunction, 1, 10, 48>(
	     new EnergyCostFunction(x_values, y_values, weights, hands_to_optimize,
				    left_or_right));
	problem.AddResidualBlock(cost_function, nullptr, &shape[0], &pose[0]);


	ceres::Solver::Options options;
	options.max_num_iterations = 50;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	std::cout << "STARTTTTTTTTTTT!" << std::endl;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << std::endl;

	// Output the initial and final pose
	std::cout << "Initial pose: ";
	for (int i = 0; i < MANO_THETA_SIZE; i++)
	{
		std::cout << poseInitial[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "Final pose: ";
	for (int i = 0; i < MANO_THETA_SIZE; i++)
	{
		std::cout << pose[i] << " ";
	}
	std::cout << std::endl;

	system("pause");
}
