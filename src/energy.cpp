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
#include <filesystem>

#define VIDEO 0

#define PRIOR_COEFF_EST    0.5  // Balancing term for surface estimation difference prior
#define PRIOR_COEFF_MEAN  0.5  // Balancing term for difference between pose/shape parameters and mean estimation of pose/shape parameters
//#define PRIOR_COEFF_POSE   0.5  // Balancing term for pose prior
#define PRIOR_COEFF_TEMP   0.5  // Balancing term for temporal regularization

std::array<std::array<double, 2>, NUM_OPENPOSE_KEYPOINTS> prev_surface_est;
std::array<double, MANO_THETA_SIZE + MANO_BETA_SIZE> optimize_params(std::string filename, int num_sequences, std::array<double, MANO_BETA_SIZE> mean_shape, std::array<double, MANO_THETA_SIZE> mean_pose, std::array<double, MANO_THETA_SIZE> prev_pose, std::array<double, MANO_BETA_SIZE> prev_shape);

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
  EnergyCostFunction(double pointX_, double pointY_, double weight_, const int iteration_, Hand LorR_, const std::array<double, MANO_BETA_SIZE> mean_shape_, const std::array<double, MANO_THETA_SIZE> mean_pose_, const int num_sequences_, const std::array<double, MANO_BETA_SIZE> prev_shape_, const std::array<double, MANO_THETA_SIZE> prev_pose_)
    : pointX(pointX_), pointY(pointY_), weight(weight_), i(iteration_), left_or_right(LorR_), mean_shape(mean_shape_), mean_pose(mean_pose_), num_sequences(num_sequences_), prev_shape(prev_shape_), prev_pose(prev_pose_)
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

		
		//DIFFERENT IDEAS FOR THE PRIORS
		
		// Prior 1: Make previous hand projections closer to the previous one.
		
		if (num_sequences > 0) {
		  residual[0] += PRIOR_COEFF_EST * pow((T(hand_projected[i][0]) - T(prev_surface_est[i][0])) + (T(hand_projected[i][1]) - T(prev_surface_est[i][0])), 2);
		}

		// Prior 2: Make pose/shape parameters closer to mean

		for (int i = 0; i < MANO_BETA_SIZE; i++) {
		  residual[0] += PRIOR_COEFF_MEAN * pow(T(shape[i]) - T(mean_shape[i]), 2);
		}

		for (int i = 0; i < MANO_THETA_SIZE; i++) {
		  residual[0] += PRIOR_COEFF_MEAN * pow(T(pose[i]) - T(mean_pose[i]), 2);
		}

		//  Optional:  Temporal Regularizer: zero-velocity prior (Real-time Pose and Shape Reconstruction of Two Interacting Hands With a Single Depth Camera)

		for (int i = 0; i < MANO_BETA_SIZE; i++) {
		  residual[0] += PRIOR_COEFF_TEMP * pow(T(shape[i]) - T(prev_shape[i]), 2);
		}

		for (int i = 0; i < MANO_THETA_SIZE; i++) {
		  residual[0] += PRIOR_COEFF_TEMP * pow(T(pose[i]) - T(prev_pose[i]), 2);
		}

		if (i == NUM_KEYPOINTS - 1) {
		  prev_surface_est = hand_projected;  // save current estimations for next iteration
		}

		testHand.reset(); 

		return true;
	}
private:
	double pointX;
	double pointY;
	double weight;
	const int i;
	const Hand left_or_right;

        const std::array<double, MANO_BETA_SIZE> mean_shape;
        const std::array<double, MANO_THETA_SIZE> mean_pose;
        const int num_sequences;
        const std::array<double, MANO_THETA_SIZE> prev_pose;
        const std::array<double, MANO_BETA_SIZE> prev_shape;
};

void runEnergy()
{
	// Read OpenPose keypoints
	std::string path;

	if (VIDEO == 1) {
		path = "samples/webcam_examples";
	}
	else {
		path = "samples/pictures/onehand1_keypoints.json";
	}


	if (VIDEO == 1) {
		int num_sequences = 0;
		std::array<double, MANO_BETA_SIZE> mean_shape = std::array<double, MANO_BETA_SIZE>();
		std::array<double, MANO_THETA_SIZE> mean_pose = std::array<double, MANO_THETA_SIZE>();

		std::array<double, MANO_THETA_SIZE> prev_pose = std::array<double, MANO_THETA_SIZE>();
		std::array<double, MANO_BETA_SIZE> prev_shape = std::array<double, MANO_BETA_SIZE>();
		std::array<double, MANO_THETA_SIZE> pose = std::array<double, MANO_THETA_SIZE>();
		std::array<double, MANO_BETA_SIZE> shape = std::array<double, MANO_BETA_SIZE>();
                
		for (const auto& entry : std::filesystem::directory_iterator(path)) {
		        std::cout << entry.path() << std::endl;

    		        std::array<double, MANO_THETA_SIZE + MANO_BETA_SIZE> params = optimize_params(entry.path().string(), num_sequences, mean_shape, mean_pose, prev_pose, prev_shape);


			// Assign pose and shape parameters from output of optimize_params
			
			for (int i = 0; i < MANO_THETA_SIZE; i++) {
			  pose[i] = params[i];
			}
			for (int i = 0; i < MANO_BETA_SIZE; i++) {
			  shape[i] = params[MANO_THETA_SIZE + i];
			}

			// Assign current pose/shape to prev pose/shape
			
			prev_pose = pose;
			prev_shape = shape;


			// Calculate mean pose/shape

			for (int i = 0; i < MANO_BETA_SIZE; i++) {
			  mean_shape[i] = (mean_shape[i] * num_sequences + shape[i]) / (num_sequences + 1);
			}

			for (int i = 0; i < MANO_THETA_SIZE; i++) {
			  mean_pose[i] = (mean_pose[i] * num_sequences + pose[i]) / (num_sequences + 1);
			}
			
			num_sequences++;

		}
	}	
	else {
		int num_sequences = 0;
		std::array<double, MANO_BETA_SIZE> mean_shape = std::array<double, MANO_BETA_SIZE>();
		std::array<double, MANO_THETA_SIZE> mean_pose = std::array<double, MANO_THETA_SIZE>();

		std::array<double, MANO_THETA_SIZE> prev_pose = std::array<double, MANO_THETA_SIZE>();
		std::array<double, MANO_BETA_SIZE> prev_shape = std::array<double, MANO_BETA_SIZE>();
		std::array<double, MANO_THETA_SIZE> pose = std::array<double, MANO_THETA_SIZE>();
		std::array<double, MANO_BETA_SIZE> shape = std::array<double, MANO_BETA_SIZE>();

		// Get params (pose + shape parameters)

		std::array<double, MANO_THETA_SIZE + MANO_BETA_SIZE> params = optimize_params(path, num_sequences, mean_shape, mean_pose, prev_pose, prev_shape);

		// Decode params into pose and shape arrays
		
		for (int i = 0; i < MANO_THETA_SIZE; i++) {
		  pose[i] = params[i];
		}
		for (int i = 0; i < MANO_BETA_SIZE; i++) {
		  shape[i] = params[MANO_THETA_SIZE + i];
		}
	}
}

std::array<double, MANO_THETA_SIZE + MANO_BETA_SIZE> optimize_params(std::string filename, int num_sequences, std::array<double, MANO_BETA_SIZE> mean_shape, std::array<double, MANO_THETA_SIZE> mean_pose, std::array<double, MANO_THETA_SIZE> prev_pose, std::array<double, MANO_BETA_SIZE> prev_shape) 
{

	std::array<std::array<double, NUM_KEYPOINTS * 3>, 2> keypoints = Parser::readJsonCV(filename);

	std::array<double, NUM_KEYPOINTS * 3> left_keypoints = keypoints[0];
	std::array<double, NUM_KEYPOINTS * 3> right_keypoints = keypoints[1];

	// Define initial values for parameters of pose and shape 
	std::array<double, MANO_THETA_SIZE> poseInitial = std::array<double, MANO_THETA_SIZE>();
	std::array<double, MANO_BETA_SIZE> shapeInitial = std::array<double, MANO_BETA_SIZE>();
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
										       new EnergyCostFunction(right_keypoints[3 * i], right_keypoints[3 * i + 1], right_keypoints[3 * i + 2], i, left_or_right, mean_shape, mean_pose, num_sequences, prev_shape, prev_pose));
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

	// Store all pose and shape params into params array

	std::array<double, MANO_THETA_SIZE + MANO_BETA_SIZE> params;
	
	for (int i = 0; i < MANO_THETA_SIZE; i++) {
	  params[i] = pose[i];
	}
	for (int i = 0; i < MANO_BETA_SIZE; i++) {
	  params[MANO_THETA_SIZE + i] = shape[i];
	}

	return params;
	
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
