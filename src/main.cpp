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

	HandModel hands("mano/model/mano_right.json", "mano/model/mano_left.json");

	std::array<double, MANO_THETA_SIZE> rnd = {};
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distrib(0, 1);
	for (auto& val : rnd) {
		val = distrib(gen) * 4 - 2;
	}

	hands.setTheta(rnd, Hand::RIGHT);
	hands.applyTransformation(Eigen::Vector3d(0, 0, 0.5), Hand::LEFT);
	hands.saveToObj("./test.obj");
}