#include "json.hpp"
#include <iostream>
#include <fstream>

#define NUM_KEYPOINTS 21

using json = nlohmann::json;

std::array<std::array<float, NUM_KEYPOINTS * 3>, 2> json_parse(std::string filename) {
  /*
     Input: Filename (.json)
     Output: Left and Right Keypoints (#keypoints = 20) (Format: x, y, c)
  */

  // Read and parse json file

  std::ifstream keypoints_file(filename);
  json js = json::parse(keypoints_file);

  // Assign keypoints to float array

  std::array<float, NUM_KEYPOINTS * 3> left_keypoints = js["people"][0]["hand_left_keypoints_2d"];
  std::array<float, NUM_KEYPOINTS * 3> right_keypoints = js["people"][0]["hand_right_keypoints_2d"];

  std::array<std::array<float, NUM_KEYPOINTS * 3>, 2> keypoints = {left_keypoints, right_keypoints};

  return keypoints;
}


void test_json_parse(void) {
  std::string filename = "samples/webcam_examples/000000000000_keypoints.json";

  std::array<std::array<float, NUM_KEYPOINTS * 3>, 2> keypoints = json_parse(filename);

  std::array<float, NUM_KEYPOINTS * 3> left_keypoints = keypoints[0];
  std::array<float, NUM_KEYPOINTS * 3> right_keypoints = keypoints[1];
  
  // Print each of left/right keypoints: x0, y0, c0, x1, y1, c1, x2, y2, c2, ..., x20, y20, c20

  std::cout << "Left keypoints:" << std::endl;

  for (int i = 0; i < NUM_KEYPOINTS; i++) {
    std::cout << i << " : " << left_keypoints[3*i] << "," << left_keypoints[3*i+1] << "," << left_keypoints[3*i+2] << std::endl; // (x,y, confidence_score)
  }
  std::cout << std::endl << "Right keypoints:" << std::endl;
  
  for (int i = 0; i < NUM_KEYPOINTS; i++) {
    std::cout << i << " : " << right_keypoints[3*i] << "," << right_keypoints[3*i+1] << "," << right_keypoints[3*i+2] << std::endl; // (x,y, confidence_score)
  }
}
