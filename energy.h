#pragma once
#include "pch.h"
#include "mano.h"


void runEnergy();
std::array<std::array<double, 2>, NUM_OPENPOSE_KEYPOINTS> doTheManoKeypointMagic(const double* s, const double* p);
void optimize_params(std::string filename, int num_sequences, std::array<double, MANO_BETA_SIZE> mean_shape, std::array<double, MANO_THETA_SIZE> prev_pose, std::array<double, MANO_BETA_SIZE> prev_shape);