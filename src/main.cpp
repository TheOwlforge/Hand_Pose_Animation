#include "pch.h"
#include "parser.hpp"
#include "parse_json.hpp"

int main(int argc, char* argv[])
{
	std::string filename = "samples/webcam_examples/000000000000_keypoints.json";

	std::array<std::array<float, NUM_KEYPOINTS * 3>, 2> keypoints = json_parse(filename);

	print_keypoints(keypoints);
}