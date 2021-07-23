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

	std::array<double, MANO_THETA_SIZE> rnd1 = std::array<double, MANO_THETA_SIZE>();
	std::array<double, MANO_THETA_SIZE> rnd3 = std::array<double, MANO_THETA_SIZE>();
	std::array<double, MANO_BETA_SIZE> rnd2 = std::array<double, MANO_BETA_SIZE>();

	//HandModel::fillRandom(&rnd1, 0.5);
	//rnd1[0] = 2;
	/*rnd1[5] = 2;
	rnd1[8] = 0.2;
	rnd1[11] = 0.1;
	rnd1[32] = 2;
	rnd1[23] = 2;
	rnd1[41] = 2;*/

	/*for (uint16_t i = 0; i < (NUM_MANO_JOINTS - 1) * 3; i++)
	{
		rnd1[i + 3] = hands->rightHand->hands_mean[i];
	}*/

	//rnd1.fill(0.2);

	hands->setModelParameters(rnd1, rnd2, Hand::RIGHT);
	//hands->reset();
	//hands->setModelParameters(rnd1, rnd2, Hand::RIGHT);
	//hands->applyTransformation(Eigen::Vector3f(0, 0, 0.2), Hand::LEFT);


	//hands->applyScale(0.70, Hand::RIGHT);

	hands->applyRotation(0.5 * M_PI, 0, M_PI, Hand::RIGHT);
	hands->applyTranslation(Eigen::Vector3f(0.03, 0.11, 1.8), Hand::RIGHT);
	hands->isVisible_left = false;

	hands->saveVertices();
	hands->saveMANOJoints();
	hands->saveOPJoints();


	hands->display("samples/pictures/onehand1.png", Hand::RIGHT);

	delete hands;
}