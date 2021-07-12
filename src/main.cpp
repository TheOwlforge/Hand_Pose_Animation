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

	HandModel hands("mano/model/mano_right.json", "mano/model/mano_left.json");
	hands.applyTransformation(Eigen::Vector3f(0, 0, 1), Hand::LEFT);
	hands.saveToObj("./test.obj");
}