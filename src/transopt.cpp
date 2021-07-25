#pragma once
#include "pch.h"
#include "transopt.h"
#include "ceres/ceres.h"

struct TranslationCostFunction
{
	TranslationCostFunction(HandModel* hands_, Hand hand_)
		: hands(hands_), hand(hand_)
	{

	}

	template<typename T>
	bool operator()(const T* const translation, T* residual) const
	{
		// TODO: Cost Function for translation

		return true;
	}

private:
	HandModel* hands;
	Hand hand;
};

void optimizeTranslation(HandModel* hands, double* translation, Hand hand)
{
	ceres::Problem problem;

	// Residual block
	ceres::CostFunction* cost_function =
		new ceres::AutoDiffCostFunction<TranslationCostFunction, 1, 3>(
			new TranslationCostFunction(hands, hand));
	problem.AddResidualBlock(cost_function, nullptr, &translation[0]);

	ceres::Solver::Options options;
	options.max_num_iterations = 25;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << std::endl;
}

void testTransOptimization()
{
	HandModel* hands = new HandModel("mano/model/mano_right.json", "mano/model/mano_left.json");

	std::array<double, MANO_THETA_SIZE> theta = hands->getMeanShape(Hand::RIGHT);
	std::array<double, MANO_BETA_SIZE> beta = std::array<double, MANO_BETA_SIZE>();

	hands->setModelParameters(theta.data(), beta.data(), Hand::RIGHT);

	double trans[3] = {0.0, 0.0, 0.0};
	optimizeTranslation(hands, &trans[0], Hand::RIGHT);

	hands->applyTranslation(Eigen::Vector3f(trans[0], trans[1], trans[2]), Hand::RIGHT);

	hands->display("samples/pictures/onehand1.png", Hand::RIGHT);

	delete hands;
}