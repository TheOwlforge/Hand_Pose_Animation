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


//Some poor attempts to include interfacing
/*struct setModelParametersFunctor {
	bool operator()(const double* pose, const double* shape, HandModel* hand) const
	{
		hand->setModelParameters(pose, shape, Hand::RIGHT);
		return true;
	}
};*/

/*struct doTheThingFunctor {
	bool operator()(const double* pose, const double* shape, double* output) const
	{
		std::array<std::array<double, 2>, NUM_OPENPOSE_KEYPOINTS> array_of_keypoints = doTheManoKeypointMagic(shape, pose);
		output = (double*)&array_of_keypoints;
		return true;
	}
};*/


struct EnergyCostFunction
{
	EnergyCostFunction(double pointX_, double pointY_, double weight_, const int iteration_, Hand LorR_)
		: pointX(pointX_), pointY(pointY_), weight(weight_), i(iteration_), left_or_right(LorR_)
	{
		//Some more attempts to include interfacing
		//set_model_params.reset(new ceres::CostFunctionToFunctor<1, 48, 10>(
		//	new ceres::NumericDiffCostFunction<setModelParametersFunctor, ceres::CENTRAL, 1, 48, 10>(new setModelParametersFunctor)));
		//mano_magic.reset(new ceres::CostFunctionToFunctor<1, 48, 10>(
		//	new ceres::NumericDiffCostFunction<doTheThingFunctor, ceres::CENTRAL, 1, 48, 10>(new doTheThingFunctor)));
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

		//camera settings for real images 
		SimpleCamera denniscam;

		//Initialization of a HandModel
		HandModel testHand("mano/model/mano_right.json", "mano/model/mano_left.json");

		//create MANO surface thorugh setting shape and pose parameters for predefined Hand Model 
		std::cout << "Setting the model parameters pose and shape..." << std::endl;
		testHand.setModelParameters((double*)shape, (double*)pose, left_or_right);

		//an attempt to replace the function above with a functor
		//(*set_model_params)(pose, shape, &testHand);

		//Translate and rotate (proper for onehand1 dataset and similar only!)
		testHand.applyRotation(0.5 * M_PI, 0, M_PI, Hand::RIGHT);
		testHand.applyTranslation(Eigen::Vector3f(0.03, 0.11, 1.8), Hand::RIGHT);

		//transform MANO to OpenPose and Project to 2D given camera intrinsics
		std::cout << "Transforming to OpenPose in 2D..." << std::endl;
		std::array<std::array<double, 2>, NUM_OPENPOSE_KEYPOINTS> hand_projected = testHand.get2DJointLocations(left_or_right, denniscam);
		
		//an attempt to replace the function above with a functor
		//(*mano_magic)(pose, shape, hand_projected);

		//simple, weighted L1 norm
		std::cout << "Computing the residual with: weight: " << weight << " keypoint: " << pointX << " predicted: " << hand_projected[i][0] << std::endl;
		residual[0] = T(weight) * (T(hand_projected[i][0]) - T(pointX) + (T(hand_projected[i][1]) - T(pointY)));

		//reset hand shape
		testHand.reset();

		return true;
	}

private:
	double pointX;
	double pointY;
	double weight;
	const int i;
	Hand left_or_right;
	//std::unique_ptr<ceres::CostFunctionToFunctor<1, 48, 10> > set_model_params;
	//std::unique_ptr<ceres::CostFunctionToFunctor<1, 48, 10> > mano_magic;
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

	// Define initial values for parameters of pose and shape 
	std::array<double, MANO_THETA_SIZE> poseInitial = std::array<double, MANO_THETA_SIZE>();;
	std::array<double, MANO_BETA_SIZE> shapeInitial = std::array<double, MANO_BETA_SIZE>();;
	HandModel::fillRandom(&poseInitial, 0.5f);
	HandModel::fillRandom(&shapeInitial, 0.5f);

	// Assign initial values to parameters
	std::array<double, MANO_THETA_SIZE> pose = poseInitial;
	std::array<double, MANO_BETA_SIZE> shape = shapeInitial;

	// FOR TESTING: we use only the right hand 
	Hand left_or_right = Hand::RIGHT;

	ceres::Problem problem;

	// Residual block for right hand 
	for (int i = 0; i < NUM_KEYPOINTS; ++i)
	{
		ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<EnergyCostFunction, 1, 10, 48>(
				new EnergyCostFunction(right_keypoints[3 * i], right_keypoints[3 * i + 1], right_keypoints[3 * i + 2], i, left_or_right));
		problem.AddResidualBlock(cost_function, nullptr, &shape[0], &pose[0]);
	}

	ceres::Solver::Options options;
	options.max_num_iterations = 25;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

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

//a poor attempt to interface mano functions into ceres
std::array<std::array<double, 2>, NUM_OPENPOSE_KEYPOINTS> doTheManoKeypointMagic(const double * s, const double * p)
{
	HandModel* myHand = new HandModel("mano/model/mano_right.json", "mano/model/mano_left.json");
	myHand->setModelParameters(s, p, Hand::RIGHT);

	SimpleCamera cam;

	myHand->applyRotation(0.5 * M_PI, 0, M_PI, Hand::RIGHT);
	myHand->applyTranslation(Eigen::Vector3f(0.03, 0.11, 1.8), Hand::RIGHT);

	std::array<std::array<double, 2>, NUM_OPENPOSE_KEYPOINTS> output = myHand->get2DJointLocations(Hand::RIGHT, cam);

	delete myHand;

	return output;
}