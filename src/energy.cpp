#include "pch.h"

void energy()
{
	//compute L1 norm between 
	// THING! Model converted to json form, projected with use of our camera intrinsics  
	//output from openpose 

	// E_openpose = L1(project(sample_joints_corresponding_to_openpose(smpl(beta, theta))), openpose_network_predicted_2d_joints)

	/*
	Let's consider an OpenPose based energy: E_openpose = L1(project(sample_joints_corresponding_to_openpose(smpl(beta, theta))), openpose_network_predicted_2d_joints), here:
	openpose_network_predicted_2d_joints - should be taken from json files produced by https://github.com/CMU-Perceptual-Computing-Lab/openpose
	smpl - can be either implemented from paper or taken from some github (look for c/c++ implementation)
	sample_joints_corresponding_to_openpose - how to regress 3D openpose conforming joints from smpl surface can be checked here https://github.com/vchoutas/smplx/blob/master/smplx/vertex_ids.py
	project - is a pinhole camera projection with intrinsics from your camera setup.

	*/
}