#pragma once
#include "pch.h"
#include "transopt.h"
#include "ceres/ceres.h"
#include "parser.hpp"

constexpr int FINGERTIPS[5] = { 4, 8, 12, 16, 20 };

double* getOPMeanPose(double* keypoints)
{
	double meanPos_OP[2] = { 0.0, 0.0 };
	for (int i = 0; i < NUM_OPENPOSE_KEYPOINTS; i++)
	{
		meanPos_OP[0] += ((double*)keypoints)[i * 3];
		meanPos_OP[1] += ((double*)keypoints)[i * 3 + 1];
	}
	meanPos_OP[0] /= NUM_OPENPOSE_KEYPOINTS;
	meanPos_OP[1] /= NUM_OPENPOSE_KEYPOINTS;

	return meanPos_OP;
}

double* getMANOMeanPose2D(HandModel& hands, const double* translation, Hand hand, SimpleCamera camera, int frame_width, int frame_height)
{
	// get 2D joint locations of the mano hand
	hands.applyTranslation(Eigen::Vector3f((float)translation[0],
		(float)translation[1],
		(float)translation[2]), hand);
	std::array<std::array<double, 2>, NUM_OPENPOSE_KEYPOINTS> j = hands.get2DJointLocations(hand, camera);
	// adjust them to image dimensions
	for (int i = 0; i < NUM_OPENPOSE_KEYPOINTS; i++)
	{
		//((int)j[i][0] / c.image_width * frame.cols, (int)j[i][1] / c.image_height * frame.rows);
		j[i][0] = j[i][0] / camera.image_width * frame_width;
		j[i][1] = j[i][0] / camera.image_width * frame_height;
	}

	// mean position of mano joints
	double meanPos_MANO[2] = { 0.0, 0.0 };
	for (int i = 0; i < NUM_OPENPOSE_KEYPOINTS; i++)
	{
		meanPos_MANO[0] += j[i][0];
		meanPos_MANO[1] += j[i][1];
	}
	meanPos_MANO[0] /= NUM_OPENPOSE_KEYPOINTS;
	meanPos_MANO[1] /= NUM_OPENPOSE_KEYPOINTS;

	return meanPos_MANO;
}

double* getMANOMeanPose3D(HandModel& hands, const double* translation, Hand hand)
{
	std::array<std::array<double, 3>, NUM_OPENPOSE_KEYPOINTS> j = hands.getFullJoints(hand);

	// mean position of mano joints
	double meanPos_MANO[3] = { 0.0, 0.0, 0.0 };
	for (int i = 0; i < NUM_OPENPOSE_KEYPOINTS; i++)
	{
		meanPos_MANO[0] += j[i][0];
		meanPos_MANO[1] += j[i][1];
		meanPos_MANO[2] += j[i][2];
	}
	meanPos_MANO[0] /= NUM_OPENPOSE_KEYPOINTS;
	meanPos_MANO[1] /= NUM_OPENPOSE_KEYPOINTS;
	meanPos_MANO[2] /= NUM_OPENPOSE_KEYPOINTS;

	return meanPos_MANO;
}

/*struct TranslationCostFunction
{
	TranslationCostFunction(HandModel* hands_,
							Hand hand_,
							std::array<std::array<double, NUM_KEYPOINTS * 3>, 2> keypoints_,
							SimpleCamera& camera_,
							int frameWidth_,
							int frameHeight_)
		: hands(hands_), hand(hand_), camera(camera_), frame_width(frameWidth_), frame_height(frameHeight_)
	{
		switch (hand)
		{
		case Hand::RIGHT:
			keypoints = keypoints_[1];
			break;
		case Hand::LEFT:
			keypoints = keypoints_[0];
			break;
		}
	}

	template<typename T>
	bool operator()(const T* const translation, T* residual) const
	{
		// TODO: Cost Function for translation
		
		// mean position of keypoints
		T* meanPos_OP = getOPMeanPose<T>((T*)keypoints.data());
		T* meanPos_MANO = getMANOMeanPose<T>(hands, translation, hand, camera, frame_width, frame_height);

		std::cout << meanPos_OP[0] << " " << meanPos_OP[1] << std::endl;
		std::cout << meanPos_MANO[0] << " " << meanPos_MANO[1] << std::endl;

		// silhouette extend of keypoints
		double distance_OP = -1;
		for (int i : FINGERTIPS)
		{
			distance_OP = std::max(distance_OP, sqrt(pow(keypoints[0] - keypoints[i * 3], 2) + pow(keypoints[1] - keypoints[i * 3 + 1], 2)));
		}

		// silhouette extend of mano joints
		double distance_MANO = -1;
		for (int i : FINGERTIPS)
		{
			distance_MANO = std::max(distance_MANO, sqrt(pow(j[0][0] - j[i][0], 2) + pow(j[0][1] - j[i][1], 2)));
		}

		// Cost function
		residual[0] = (T)(sqrt((meanPos_OP[0] - meanPos_MANO[0]))); // / frame_width));
		residual[1] = (T)(sqrt((meanPos_OP[1] - meanPos_MANO[1]))); // / frame_height));
		//residual[2] = (T)(distance_MANO - distance_OP);

		// reset hand
		hands->reset();

		return true;
	}

private:
	HandModel* hands;
	Hand hand;
	std::array<double, NUM_OPENPOSE_KEYPOINTS * 3> keypoints;
	SimpleCamera& camera;
	int frame_width;
	int frame_height;
}; */

double* optimizeTranslation(HandModel& hands,
						 Hand hand,
						 std::array<std::array<double, NUM_OPENPOSE_KEYPOINTS * 3>, 2> keypoints,
						 SimpleCamera& camera,
						 int frame_width,
						 int frame_height)
{
	/*ceres::Problem problem;

	// Residual block
	ceres::CostFunction* cost_function =
		new ceres::AutoDiffCostFunction<TranslationCostFunction, 2, 3>(
			new TranslationCostFunction(hands, hand, keypoints, camera, frame_width, frame_height));
	problem.AddResidualBlock(cost_function, nullptr, translation);

	ceres::Solver::Options options;
	options.max_num_iterations = 25;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << std::endl;*/

	double* j_OP;
	switch (hand)
	{
	case Hand::RIGHT:
		j_OP = keypoints[1].data();
		break;
	case Hand::LEFT:
		j_OP = keypoints[0].data();
		break;
	}

	//hardcode depth
	double translation[3] = {0, 0, DEPTH};

	double* meanPos_OP_2D = getOPMeanPose(j_OP);
	Eigen::Matrix3f K = camera.getK();
	double x = ((meanPos_OP_2D[0] / frame_width) * camera.image_width - K(0,2)) / K(0,0);
	double y = ((meanPos_OP_2D[1] / frame_height) * camera.image_height - K(0,2)) / K(1,1);
	double z = translation[2];

	hands.applyTranslation(Eigen::Vector3f((float)translation[0],
		(float)translation[1],
		(float)translation[2]), hand);
	double* meanPos_MANO = getMANOMeanPose3D(hands, translation, hand);
	hands.reset();

	translation[0] = (x - meanPos_MANO[0]);
	translation[1] = -(y - meanPos_MANO[1]);

	return translation;
}

void testTransOptimization()
{
	HandModel* hands = new HandModel("mano/model/mano_right.json", "mano/model/mano_left.json");
	std::array<std::array<double, NUM_KEYPOINTS * 3>, 2> keypoints_OP = Parser::readJsonCV("samples/pictures/onehand1_keypoints.json");

	std::array<double, MANO_THETA_SIZE> theta = std::array<double, MANO_THETA_SIZE>(); //hands->getMeanShape(Hand::RIGHT);
	std::array<double, MANO_BETA_SIZE> beta = std::array<double, MANO_BETA_SIZE>();

	hands->setModelParameters(theta.data(), beta.data(), Hand::RIGHT);
	hands->applyRotation(0.5 * M_PI, 0, M_PI, Hand::RIGHT);

	SimpleCamera c;
	double* trans = optimizeTranslation(*hands, Hand::RIGHT, keypoints_OP, c, 458, 258);
	
	hands->applyRotation(0.5 * M_PI, 0, M_PI, Hand::RIGHT); //apply rotation again for display
	hands->applyTranslation(Eigen::Vector3f(trans[0], trans[1], trans[2]), Hand::RIGHT);

	hands->display("samples/pictures/onehand1.png", Hand::RIGHT, c);

	delete hands;
}