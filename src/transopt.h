#pragma once
#include "mano.h"

constexpr double DEPTH = 1.7;

double* optimizeTranslation(HandModel& hands,
						    Hand hand,
						    std::array<std::array<double, NUM_OPENPOSE_KEYPOINTS * 3>, 2> keypoints,
						    SimpleCamera& camera,
						    int frame_width,
						    int frame_height);
void testTransOptimization();