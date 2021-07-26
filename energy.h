#pragma once


void runEnergy();
std::array<std::array<double, 2>, NUM_OPENPOSE_KEYPOINTS> doTheManoKeypointMagic(const double* s, const double* p);