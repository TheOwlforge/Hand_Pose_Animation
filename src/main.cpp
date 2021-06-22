#include "pch.h"
#include "parser.hpp"

int main(int argc, char* argv[])
{
	std::cout << "Hello World!" << std::endl;
	//Parser::test();
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	op::opLog("Starting OpenPose demo...", op::Priority::High);
}