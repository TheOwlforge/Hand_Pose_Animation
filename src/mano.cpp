#pragma once
#include "pch.h"
#include "mano.h"
#include "parser.hpp"

HandModel::HandModel(std::string rightHand_filename, std::string leftHand_filename)
{
	rightHand = Parser::readJsonMANO(rightHand_filename);
	leftHand = Parser::readJsonMANO(leftHand_filename);

	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		rightHand->vertices[i] = Eigen::Vector4d(rightHand->T[i].x(), rightHand->T[i].y(), rightHand->T[i].z(), 1);
		leftHand->vertices[i] = Eigen::Vector4d(leftHand->T[i].x(), leftHand->T[i].y(), leftHand->T[i].z(), 1);
	}
	for (int j = 0; j < NUM_MANO_JOINTS; j++)
	{
		rightHand->joints[j] = rightHand->J[j];
		leftHand->joints[j] = leftHand->J[j];
	}

	isVisible_right = true;
	isVisible_left = true;

	// init rest post with zero
	rightHand->theta = std::array<double, MANO_THETA_SIZE>();
	leftHand->theta = std::array<double, MANO_THETA_SIZE>();

	rightHand->beta = Eigen::Vector<double, MANO_BETA_SIZE>::Zero();
	leftHand->beta = Eigen::Vector<double, MANO_BETA_SIZE>::Zero();

	for (int k = 0; k < NUM_MANO_JOINTS; k++)
	{
		rightHand->inv_G_zero[k] = computeG(rightHand, k).inverse();
		leftHand->inv_G_zero[k] = computeG(leftHand, k).inverse();
	}
	for (int k = 0; k < NUM_MANO_JOINTS - 1; k++)
	{
		rightHand->R_zero[k] = computeR(rightHand, k);
		leftHand->R_zero[k] = computeR(leftHand, k);
	}
}

HandModel::~HandModel()
{
	free(rightHand);
	free(leftHand);
}

void HandModel::setTheta(std::array<double, MANO_THETA_SIZE> theta, Hand hand)
{
	std::cout << "Applying Model Pose Parameters" << std::endl;

	ManoHand* h;
	switch (hand)
	{
	case Hand::RIGHT:
		h = rightHand;
		break;
	case Hand::LEFT:
		h = leftHand;
		break;
	}

	h->theta = theta;

	auto start = std::chrono::high_resolution_clock::now();

	/*concurrency::parallel_for(size_t(0), (size_t)NUM_MANO_VERTICES, [&](size_t i)
		{
			Eigen::Vector4d new_vertex = Eigen::Vector4d();
			for (int k = 0; k < NUM_MANO_JOINTS; k++)
			{
				Eigen::Matrix4d G_k = computeG(h, k);
				Eigen::Matrix4d G_k_prime = G_k * h->inv_G_zero[k];
				new_vertex += h->W(k, i) * G_k_prime * h->vertices[i];
			}
		});*/

	//compute shape offsets
	Eigen::Vector<double, NUM_MANO_VERTICES * 3> BS_temp = h->shape_blend_shapes * h->beta;
	std::array<Eigen::Vector3d, NUM_MANO_VERTICES> BS = std::array<Eigen::Vector3d, NUM_MANO_VERTICES>();
	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		BS[i] = Eigen::Vector3d(BS_temp(3 * i), BS_temp(3 * i + 1), BS_temp(3 * i + 2));
	}

	//compute pose offsets
	Eigen::Vector<double, MANO_R_SIZE> R;
	for (int j = 0; j < (NUM_MANO_JOINTS - 1); j++)
	{
		Eigen::Vector<double, 9> R_current = computeR(h, j) - h->R_zero[j];
		for (int i = 0; i < 9; i++)
		{
			R(9 * j + i) = R_current(i);
		}
	}

	Eigen::Vector<double, NUM_MANO_VERTICES * 3> BP_temp = h->pose_blend_shapes * R;
	std::array<Eigen::Vector3d, NUM_MANO_VERTICES> BP = std::array<Eigen::Vector3d, NUM_MANO_VERTICES>();
	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		BP[i] = Eigen::Vector3d(BP_temp(3 * i), BP_temp(3 * i + 1), BP_temp(3 * i + 2));
	}

	//update joints
	for (int j = 0; j < NUM_MANO_JOINTS; j++)
	{
		Eigen::Vector3d new_joint = Eigen::Vector3d::Zero();
		for (int i = 0; i < NUM_MANO_VERTICES; i++)
		{
			new_joint += h->joint_regressor(j, i) * (h->T[i] + BS[i]);
		}
		h->joints[j] = new_joint;
	}

	//update vertices
	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		Eigen::Vector4d new_vertex = Eigen::Vector4d::Zero();
		for (int k = 0; k < NUM_MANO_JOINTS; k++)
		{
			Eigen::Matrix4d G_k = computeG(h, k);
			Eigen::Matrix4d G_k_prime = G_k * h->inv_G_zero[k];
			Eigen::Vector4d bs = Eigen::Vector4d(BS[i].x(), BS[i].y(), BS[i].z(), 0);
			Eigen::Vector4d bp = Eigen::Vector4d(BP[i].x(), BP[i].y(), BP[i].z(), 0);
			new_vertex += h->W(i, k) * G_k_prime * (h->vertices[i] + bs + bp);
		}
		h->vertices[i] = new_vertex;
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	std::cout << "...took " << duration.count() << " ms." << std::endl << std::endl;
}

void HandModel::setBeta(std::array<double, MANO_BETA_SIZE> theta, Hand hand)
{

}

std::array<Eigen::Vector3d, NUM_OPENPOSE_KEYPOINTS> HandModel::getJointLocations(Hand hand)
{
	ManoHand* h;
	switch (hand)
	{
	case Hand::RIGHT:
		h = rightHand;
		break;
	case Hand::LEFT:
		h = leftHand;
		break;
	}

	std::array<Eigen::Vector3d, NUM_OPENPOSE_KEYPOINTS> result = std::array<Eigen::Vector3d, NUM_OPENPOSE_KEYPOINTS>();
	for (int j = 0; j < NUM_MANO_JOINTS; j++)
	{
		result[j] = h->joints[j];
	}
	// TODO: add 5 additional joints for openpose
	return result;
}

void HandModel::applyTransformation(Eigen::Vector3d t, Hand hand)
{
	std::cout << "Applying Transformation" << std::endl;

	Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
	transform.block<3, 1>(0, 3) = t;
	ManoHand* h;
	switch (hand)
	{
	case Hand::RIGHT:
		h = rightHand;
		break;
	case Hand::LEFT:
		h = leftHand;
		break;
	}
	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		h->vertices[i] = transform * h->vertices[i];
	}
}

bool HandModel::saveToObj(std::string output_filename)
{
	std::cout << "Saving Hands Model to " << output_filename << std::endl;

	std::ofstream outFile(output_filename);
	if (!outFile.is_open()) return false;

	// right Hand vertices
	if (isVisible_right)
	{
		for (int i = 0; i < NUM_MANO_VERTICES; i++)
		{
			Eigen::Vector4d v = rightHand->vertices[i];
			outFile << "v " << v.x() << " " << v.y() << " " << v.z() << std::endl;
		}
	}
	// left Hand vertices
	if (isVisible_left)
	{
		for (int i = 0; i < NUM_MANO_VERTICES; i++)
		{
			Eigen::Vector4d v = leftHand->vertices[i];
			outFile << "v " << v.x() << " " << v.y() << " " << v.z() << std::endl;
		}

	}
	// right Hand faces
	if (isVisible_right)
	{
		for (int i = 0; i < NUM_MANO_FACES; i++)
		{
			Eigen::Vector3i f = rightHand->face_indices[i];
			outFile << "f " << f.x() << " " << f.y() << " " << f.z() << std::endl;
		}
	}
	// left Hand faces
	if (isVisible_left)
	{
		for (int i = 0; i < NUM_MANO_FACES; i++)
		{
			// add number of vertices to correct face indices to save both in the same obj files
			Eigen::Vector3i f = leftHand->face_indices[i] + (NUM_MANO_VERTICES)*Eigen::Vector3i::Ones();
			outFile << "f " << f.x() << " " << f.y() << " " << f.z() << std::endl;
		}
	}

	outFile.close();
	return true;
}

bool HandModel::hasAncestor(ManoHand* h, unsigned int joint_index)
{
	if (!(*h->kinematic_tree).count(joint_index))
		return false;
	return (*h->kinematic_tree).at(joint_index) < NUM_MANO_JOINTS;
}

unsigned int HandModel::getAncestor(ManoHand* h, unsigned int joint_index)
{
	if (!(*h->kinematic_tree).count(joint_index))
		return -1;
	return (*h->kinematic_tree).at(joint_index);
}

Eigen::Matrix4d HandModel::computeG(ManoHand* h, unsigned int joint_index)
{
	Eigen::Matrix4d G_k = Eigen::Matrix4d::Identity();
	unsigned int current_idx = joint_index;
	while (hasAncestor(h, current_idx))
	{
		Eigen::Matrix4d A = Eigen::Matrix4d::Identity();
		unsigned int j = getAncestor(h, current_idx);

		A.block<3, 3>(0, 0) = rodrigues(Eigen::Vector3d(h->theta[j * 3], h->theta[j * 3 + 1], h->theta[j * 3 + 2]));
		A.block<3, 1>(0, 3) = h->joints[j];

		G_k *= A;
		current_idx = j;
	}
	return G_k;
}

Eigen::Vector<double, 9> HandModel::computeR(ManoHand* h, unsigned int j)
{
	Eigen::Matrix3d R_mat = rodrigues(Eigen::Vector3d(rightHand->theta[j * 3], rightHand->theta[j * 3 + 1], rightHand->theta[j * 3 + 2]));
	R_mat.transposeInPlace();
	Eigen::VectorXd R_vec(Eigen::Map<Eigen::VectorXd>(R_mat.data(), R_mat.cols() * R_mat.rows()));
	return R_vec;
}

Eigen::Matrix3d HandModel::rodrigues(Eigen::Vector3d w)
{
	float w_length = w.norm();
	Eigen::Vector3d w_normalized = w.normalized();
	Eigen::Matrix3d skew;
	skew << 0, -w_normalized.z(), w_normalized.y(),
		w_normalized.z(), 0, -w_normalized.x(),
		-w_normalized.y(), w_normalized.x(), 0;

	Eigen::Matrix3d result = Eigen::Matrix3d::Identity() + skew * sin(w_length) + (skew * skew) * cos(w_length);
	return result;
}