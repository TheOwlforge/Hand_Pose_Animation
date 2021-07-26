#pragma once
#include "pch.h"
#include "mano.h"
#include "parser.hpp"
#include "transopt.h"
#include "energy.h"

// declare static variables for parser once globally
// cannot be done in parser.hpp to avoid errors when including in multiple files
std::vector<cv::VideoCapture> Parser::caps;
std::vector<cv::VideoWriter> Parser::videos;

int main(int argc, char* argv[])
{
	Parser::testJson();

	runEnergy();

	HandModel::test();
	testTransOptimization();
}