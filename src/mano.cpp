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
		rightHand->vertices[i] = Eigen::Vector4f(rightHand->T[i].x(), rightHand->T[i].y(), rightHand->T[i].z(), 1);
		leftHand->vertices[i] = Eigen::Vector4f(leftHand->T[i].x(), leftHand->T[i].y(), leftHand->T[i].z(), 1);
	}
	for (int j = 0; j < NUM_MANO_JOINTS; j++)
	{
		rightHand->joints[j] = Eigen::Vector4f(rightHand->J[j].x(), rightHand->J[j].y(), rightHand->J[j].z(), 1);
		leftHand->joints[j] = Eigen::Vector4f(leftHand->J[j].x(), leftHand->J[j].y(), leftHand->J[j].z(), 1);
	}

	isVisible_right = true;
	isVisible_left = true;

	// init rest post with zero
	rightHand->theta = {};
	leftHand->theta = {};

	rightHand->beta = {};
	leftHand->beta = {};

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
	delete rightHand;
	delete leftHand;
}

/*concurrency::parallel_for(size_t(0), (size_t)NUM_MANO_VERTICES, [&](size_t i)
	{
		Eigen::Vector4f new_vertex = Eigen::Vector4f();
		for (int k = 0; k < NUM_MANO_JOINTS; k++)
		{
			Eigen::Matrix4f G_k = computeG(h, k);
			Eigen::Matrix4f G_k_prime = G_k * h->inv_G_zero[k];
			new_vertex += h->W(k, i) * G_k_prime * h->vertices[i];
		}
	});*/

void HandModel::setModelParameters(std::array<float, MANO_THETA_SIZE> theta, std::array<float, MANO_BETA_SIZE> beta, Hand hand)
{
	std::cout << "Applying Model Parameters" << std::endl;

	ManoHand* h;
	switch (hand)
	{
	case Hand::RIGHT:
		h = rightHand;
		break;
	case Hand::LEFT:
		h = leftHand;
		break;
	default:
		return;
	}

	h->theta = theta;
	h->beta = beta;

	auto start = std::chrono::high_resolution_clock::now();

	//compute shape offsets via Bs = sum(beta_n * S_n)
	Eigen::Vector<float, NUM_MANO_VERTICES * 3> BS_temp = h->shape_blend_shapes * Eigen::Map<Eigen::Vector<float, MANO_BETA_SIZE>>(h->beta.data());
	std::array<Eigen::Vector3f, NUM_MANO_VERTICES> BS{};
	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		BS[i] = Eigen::Vector3f(BS_temp(3 * i), BS_temp(3 * i + 1), BS_temp(3 * i + 2));
	}

	//compute pose offsets via Bp = sum((R_theta - R_theta_zero) * P_n)
	Eigen::Vector<float, MANO_R_SIZE> R;
	for (int j = 0; j < (NUM_MANO_JOINTS - 1); j++)
	{
		Eigen::Vector<float, 9> R_current = computeR(h, j) - h->R_zero[j];
		for (int i = 0; i < 9; i++)
		{
			R(9 * j + i) = R_current(i);
		}
	}

	Eigen::Vector<float, NUM_MANO_VERTICES * 3> BP_temp = h->pose_blend_shapes * R;
	std::array<Eigen::Vector3f, NUM_MANO_VERTICES> BP{};
	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		BP[i] = Eigen::Vector3f(BP_temp(3 * i), BP_temp(3 * i + 1), BP_temp(3 * i + 2));
	}

	//update joints according to shape by calculating J_reg * (T + Bs)
	for (int j = 0; j < NUM_MANO_JOINTS; j++)
	{
		Eigen::Vector3f new_joint = Eigen::Vector3f::Zero();
		for (int i = 0; i < NUM_MANO_VERTICES; i++)
		{
			new_joint += h->joint_regressor(j, i) * (h->T[i] + BS[i]);
		}
		h->joints[j] = Eigen::Vector4f(new_joint.x(), new_joint.y(), new_joint.z(), 1);
	}

	//update vertices
	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		Eigen::Vector4f new_vertex = Eigen::Vector4f::Zero();
		for (int k = 0; k < NUM_MANO_JOINTS; k++)
		{
			Eigen::Matrix4f G_k = computeG(h, k);
			Eigen::Matrix4f G_k_prime = G_k * h->inv_G_zero[k];
			Eigen::Vector4f bs = Eigen::Vector4f(BS[i].x(), BS[i].y(), BS[i].z(), 0);
			Eigen::Vector4f bp = Eigen::Vector4f(BP[i].x(), BP[i].y(), BP[i].z(), 0);
			new_vertex += h->W(i, k) * G_k_prime * (h->vertices[i] + bs + bp);
		}
		h->vertices[i] = new_vertex;
	}

	//update joints by calculating J_Reg * Vertices
	Eigen::MatrixXf v_matrix(NUM_MANO_VERTICES, 3);
	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		v_matrix(i, 0) = h->vertices[i].x();
		v_matrix(i, 1) = h->vertices[i].y();
		v_matrix(i, 2) = h->vertices[i].z();
	}
	Eigen::MatrixXf joints_matrix = h->joint_regressor * v_matrix; // dimension NUM_MANO_JOINTS * 3
	for (int j = 0; j < NUM_MANO_JOINTS; j++)
	{
		h->joints[j] = Eigen::Vector4f(joints_matrix(j, 0), joints_matrix(j, 1), joints_matrix(j, 2), 1);
	}

	//print out time
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	std::cout << "...took " << duration.count() << " ms." << std::endl << std::endl;
}

std::array<std::array<float, 2>, NUM_OPENPOSE_KEYPOINTS> HandModel::get2DJointLocations(Hand hand)
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

	std::array<std::array<float, 2>, NUM_OPENPOSE_KEYPOINTS> result{};
	for (int j = 0; j < NUM_MANO_JOINTS; j++)
	{
		result[j] = { h->joints[j].x(), h->joints[j].y() };
	}
	// add 5 additional joints for openpose
	for (int j = 0; j < 5; j++)
	{
		result[NUM_MANO_JOINTS + j] = { h->vertices[ADDITIONAL_JOINTS[j]].x(), h->vertices[ADDITIONAL_JOINTS[j]].y() };
	}
	return result;
}

void HandModel::applyTransformation(Eigen::Vector3f t, Hand hand)
{
	std::cout << "Applying Transformation" << std::endl;

	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
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
	for (int i = 0; i < NUM_MANO_JOINTS; i++)
	{
		h->joints[i] = transform * h->joints[i];
	}
}

bool HandModel::saveVertices()
{
	std::string output_filename = "vertices.obj";
	std::cout << "Saving Hands Model to " << output_filename << std::endl;

	std::ofstream outFile(output_filename);
	if (!outFile.is_open()) return false;

	// right Hand vertices
	if (isVisible_right)
	{
		for (int i = 0; i < NUM_MANO_VERTICES; i++)
		{
			Eigen::Vector4f v = rightHand->vertices[i];
			outFile << "v " << v.x() << " " << v.y() << " " << v.z() << std::endl;
		}
	}
	// left Hand vertices
	if (isVisible_left)
	{
		for (int i = 0; i < NUM_MANO_VERTICES; i++)
		{
			Eigen::Vector4f v = leftHand->vertices[i];
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
			Eigen::Vector3i f = leftHand->face_indices[i] + (isVisible_right * NUM_MANO_VERTICES) * Eigen::Vector3i::Ones();
			outFile << "f " << f.x() << " " << f.y() << " " << f.z() << std::endl;
		}
	}

	outFile.close();

	return true;
}

bool HandModel::saveMANOJoints()
{
	std::string output_filename = "jointsMANO.off";
	std::cout << "Saving Joints to " << output_filename << std::endl;

	std::ofstream jointsFile(output_filename);
	if (!jointsFile.is_open()) return false;

	jointsFile << "OFF" << std::endl;
	jointsFile << NUM_MANO_JOINTS * (isVisible_left + isVisible_right) * 4 << " " << NUM_MANO_JOINTS * (isVisible_left + isVisible_right) << " 0" << std::endl;

	float offset = 0.001;
	// right Hand vertices
	if (isVisible_right)
	{
		for (int i = 0; i < NUM_MANO_JOINTS; i++)
		{
			Eigen::Vector4f v = rightHand->joints[i];
			jointsFile << v.x() - offset << " " << v.y() - offset << " " << v.z() << std::endl;
			jointsFile << v.x() + offset << " " << v.y() - offset << " " << v.z() << std::endl;
			jointsFile << v.x() - offset << " " << v.y() + offset << " " << v.z() << std::endl;
			jointsFile << v.x() + offset << " " << v.y() + offset << " " << v.z() << std::endl;
		}
	}
	// left Hand vertices
	if (isVisible_left)
	{
		for (int i = 0; i < NUM_MANO_JOINTS; i++)
		{
			Eigen::Vector4f v = leftHand->joints[i];
			jointsFile << v.x() - offset << " " << v.y() - offset << " " << v.z() << std::endl;
			jointsFile << v.x() + offset << " " << v.y() - offset << " " << v.z() << std::endl;
			jointsFile << v.x() - offset << " " << v.y() + offset << " " << v.z() << std::endl;
			jointsFile << v.x() + offset << " " << v.y() + offset << " " << v.z() << std::endl;
		}

	}
	// right Hand faces
	if (isVisible_right)
	{
		for (int i = 0; i < NUM_MANO_JOINTS; i++)
		{
			jointsFile << 4 << " " << i * 4 << " " << i * 4 + 1 << " " << i * 4 + 3 << " " << i * 4 + 2 << " " << 255 << " " << 0 << " " << 0 << std::endl;
		}
	}
	// left Hand faces
	if (isVisible_left)
	{
		int o = isVisible_right * NUM_MANO_JOINTS * 4;
		for (int i = 0; i < NUM_MANO_JOINTS; i++)
		{
			jointsFile << 4 << " " << o + i * 4 << " " << o + i * 4 + 1 << " " << o + i * 4 + 3 << " " << o + i * 4 + 2 << " " << 0 << " " << 0 << " " << 255 << std::endl;
		}
	}

	jointsFile.close();

	return true;
}

bool HandModel::saveOPJoints()
{
	std::string output_filename = "jointsOP.off";
	std::cout << "Saving Joints to " << output_filename << std::endl;

	std::ofstream jointsFile(output_filename);
	if (!jointsFile.is_open()) return false;

	jointsFile << "OFF" << std::endl;
	jointsFile << NUM_OPENPOSE_KEYPOINTS * (isVisible_left + isVisible_right) * 4 << " " << NUM_OPENPOSE_KEYPOINTS * (isVisible_left + isVisible_right) << " 0" << std::endl;

	float offset = 0.001;
	// right Hand vertices
	jointsFile << "# right hand vertices" << std::endl;
	if (isVisible_right)
	{
		jointsFile << "# MANO" << std::endl;
		for (int i = 0; i < NUM_MANO_JOINTS; i++)
		{
			Eigen::Vector4f v = rightHand->joints[i];
			jointsFile << v.x() - offset << " " << v.y() - offset << " " << v.z() << std::endl;
			jointsFile << v.x() + offset << " " << v.y() - offset << " " << v.z() << std::endl;
			jointsFile << v.x() - offset << " " << v.y() + offset << " " << v.z() << std::endl;
			jointsFile << v.x() + offset << " " << v.y() + offset << " " << v.z() << std::endl;
		}
		// add 5 additional joints for openpose
		jointsFile << "# OpenPose" << std::endl;
		for (int j = 0; j < 5; j++)
		{
			Eigen::Vector4f v = rightHand->vertices[ADDITIONAL_JOINTS[j]];
			jointsFile << v.x() - offset << " " << v.y() - offset << " " << v.z() << std::endl;
			jointsFile << v.x() + offset << " " << v.y() - offset << " " << v.z() << std::endl;
			jointsFile << v.x() - offset << " " << v.y() + offset << " " << v.z() << std::endl;
			jointsFile << v.x() + offset << " " << v.y() + offset << " " << v.z() << std::endl;
		}
	}
	// left Hand vertices
	jointsFile << "# left hand vertices" << std::endl;
	if (isVisible_left)
	{
		jointsFile << "# MANO" << std::endl;
		for (int i = 0; i < NUM_MANO_JOINTS; i++)
		{
			Eigen::Vector4f v = leftHand->joints[i];
			jointsFile << v.x() - offset << " " << v.y() - offset << " " << v.z() << std::endl;
			jointsFile << v.x() + offset << " " << v.y() - offset << " " << v.z() << std::endl;
			jointsFile << v.x() - offset << " " << v.y() + offset << " " << v.z() << std::endl;
			jointsFile << v.x() + offset << " " << v.y() + offset << " " << v.z() << std::endl;
		}
		// add 5 additional joints for openpose
		jointsFile << "# OpenPose" << std::endl;
		for (int j = 0; j < 5; j++)
		{
			Eigen::Vector4f v = leftHand->vertices[ADDITIONAL_JOINTS[j]];
			jointsFile << v.x() - offset << " " << v.y() - offset << " " << v.z() << std::endl;
			jointsFile << v.x() + offset << " " << v.y() - offset << " " << v.z() << std::endl;
			jointsFile << v.x() - offset << " " << v.y() + offset << " " << v.z() << std::endl;
			jointsFile << v.x() + offset << " " << v.y() + offset << " " << v.z() << std::endl;
		}
	}
	// right Hand faces
	jointsFile << "# right hand faces" << std::endl;
	if (isVisible_right)
	{
		for (int i = 0; i < NUM_OPENPOSE_KEYPOINTS; i++)
		{
			jointsFile << 4 << " " << i * 4 << " " << i * 4 + 1 << " " << i * 4 + 3 << " " << i * 4 + 2 << " " << 255 << " " << 0 << " " << 0 << std::endl;
		}
	}
	// left Hand faces
	jointsFile << "# left hand faces" << std::endl;
	if (isVisible_left)
	{
		int o = isVisible_right * NUM_OPENPOSE_KEYPOINTS * 4;
		for (int i = 0; i < NUM_OPENPOSE_KEYPOINTS; i++)
		{
			jointsFile << 4 << " " << o + i * 4 << " " << o + i * 4 + 1 << " " << o + i * 4 + 3 << " " << o + i * 4 + 2 << " " << 0 << " " << 0 << " " << 255 << std::endl;
		}
	}

	jointsFile.close();

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

Eigen::Matrix4f HandModel::computeG(ManoHand* h, unsigned int joint_index)
{
	Eigen::Matrix4f G_k = Eigen::Matrix4f::Identity();

	if (joint_index == 0)
	{
		G_k.block<3, 3>(0, 0) = rodrigues(Eigen::Vector3f(h->theta[joint_index * 3], h->theta[joint_index * 3 + 1], h->theta[joint_index * 3 + 2]));
		G_k.block<3, 1>(0, 3) = Eigen::Vector3f(h->joints[joint_index].x(), h->joints[joint_index].y(), h->joints[joint_index].z());
		return G_k;
	}

	unsigned int current_idx = joint_index;
	std::vector<unsigned int> ancestors;
	while (hasAncestor(h, current_idx))
	{
		unsigned int j = getAncestor(h, current_idx);
		ancestors.push_back(j);
		current_idx = j;
	}

	//size is larger 1 as root is handled beforehand
	for (int j = ancestors[ancestors.size() - 1]; j >= 0; j--)
	{
		Eigen::Matrix4f A = Eigen::Matrix4f::Identity();

		A.block<3, 3>(0, 0) = rodrigues(Eigen::Vector3f(h->theta[j * 3], h->theta[j * 3 + 1], h->theta[j * 3 + 2]));
		A.block<3, 1>(0, 3) = Eigen::Vector3f(h->joints[j].x(), h->joints[j].y(), h->joints[j].z());

		G_k *= A;
	}
	return G_k;
}

Eigen::Vector<float, 9> HandModel::computeR(ManoHand* h, unsigned int j)
{
	Eigen::Matrix3f R_mat = rodrigues(Eigen::Vector3f(rightHand->theta[j * 3], rightHand->theta[j * 3 + 1], rightHand->theta[j * 3 + 2]));
	R_mat.transposeInPlace();
	Eigen::VectorXf R_vec(Eigen::Map<Eigen::VectorXf>(R_mat.data(), R_mat.cols() * R_mat.rows()));
	return R_vec;
}

Eigen::Matrix3f HandModel::rodrigues(Eigen::Vector3f w)
{
	float w_length = w.norm();
	Eigen::Vector3f w_normalized = w.normalized();
	Eigen::Matrix3f skew;
	skew << 0, -w_normalized.z(), w_normalized.y(),
		w_normalized.z(), 0, -w_normalized.x(),
		-w_normalized.y(), w_normalized.x(), 0;
	//skew *= -1;
	//skew.transposeInPlace();

	Eigen::Matrix3f result = Eigen::Matrix3f::Identity() + skew * sin(w_length) + (skew * skew) * cos(w_length);
	return result;
}