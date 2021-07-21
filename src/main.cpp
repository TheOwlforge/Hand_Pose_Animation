#pragma once
#include "pch.h"
#include "mano.h"
#include "parser.hpp"

// declare static variables for parser once globally
// cannot be done in parser.hpp to avoid errors when including in multiple files
std::vector<cv::VideoCapture> Parser::caps;
std::vector<cv::VideoWriter> Parser::videos;

int main(int argc, char* argv[])
{
	Parser::testJson();

	std::cout << sizeof(float) << ", " << sizeof(ManoHand) << std::endl;

	HandModel* hands = new HandModel("mano/model/mano_right.json", "mano/model/mano_left.json");

	std::array<float, MANO_THETA_SIZE> rnd1 = std::array<float, MANO_THETA_SIZE>();
	std::array<float, MANO_BETA_SIZE> rnd2 = std::array<float, MANO_BETA_SIZE>();

	HandModel::fillRandom(&rnd1, 0.5);
	//rnd1[0] = 1;
	//rnd1[3] = 1;

	/*for (uint16_t i = 0; i < (NUM_MANO_JOINTS - 1) * 3; i++)
	{
		rnd1[i + 3] = hands->rightHand->hands_mean[i];
	}*/

	hands->setModelParameters(rnd1, rnd2, Hand::RIGHT);
	hands->applyTransformation(Eigen::Vector3f(0, 0, 0.5), Hand::LEFT);
	hands->saveVertices();
	hands->saveMANOJoints();
	hands->saveOPJoints();

	delete hands;
}