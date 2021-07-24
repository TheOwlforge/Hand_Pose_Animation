#pragma once
#include "pch.h"
#include "mano.h"
#include "parser.hpp"

HandModel::HandModel(std::string rightHand_filename, std::string leftHand_filename)
{
	rightHand = Parser::readJsonMANO(rightHand_filename);
	leftHand = Parser::readJsonMANO(leftHand_filename);

	reset();
}

HandModel::~HandModel()
{
	//delete rightHand;
	//delete leftHand;
}

void HandModel::reset()
{
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
	for (int k = 1; k < NUM_MANO_JOINTS; k++)
	{
		rightHand->R_zero[k - 1] = computeR(rightHand, k);
		leftHand->R_zero[k - 1] = computeR(leftHand, k);
	}
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

template<typename T>
void HandModel::setModelParameters(const T*const theta, const T*const beta, Hand hand)
{
	std::cout << "Applying Model Parameters" << std::endl;

	std::shared_ptr<ManoHand> h;
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
	Eigen::Vector<float, NUM_MANO_VERTICES * 3> BS_temp = h->shape_blend_shapes * Eigen::Map<Eigen::Vector<double, MANO_BETA_SIZE>>(h->beta.data()).cast<float>();
	std::array<Eigen::Vector3f, NUM_MANO_VERTICES> BS{};
	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		BS[i] = Eigen::Vector3f(BS_temp(3 * i), BS_temp(3 * i + 1), BS_temp(3 * i + 2));
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

	//compute pose offsets via Bp = sum((R_theta - R_theta_zero) * P_n)
	Eigen::Vector<float, MANO_R_SIZE> R;
	std::array<Eigen::Vector<float, 9>, NUM_MANO_JOINTS - 1> R_new;
	for (int j = 1; j < NUM_MANO_JOINTS; j++)
	{
		Eigen::Vector<float, 9> R_current = computeR(h, j) - h->R_zero[j-1];
		R_new[j - 1] = R_current;
		for (int i = 0; i < 9; i++)
		{
			R(9 * (j-1) + i) = R_current(i);
		}
	}

	Eigen::Vector<float, NUM_MANO_VERTICES * 3> BP_temp = h->pose_blend_shapes * R;
	std::array<Eigen::Vector3f, NUM_MANO_VERTICES> BP{};
	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		BP[i] = Eigen::Vector3f(BP_temp(3 * i), BP_temp(3 * i + 1), BP_temp(3 * i + 2));
	}

	//precompute transformations
	std::array<Eigen::Matrix4f, NUM_MANO_JOINTS> G;
	for (int k = 0; k < NUM_MANO_JOINTS; k++)
	{
		Eigen::Matrix4f G_k = computeG(h, k);
		/*Eigen::Vector4f test = G_k * Eigen::Vector4f(h->J[k].x(), h->J[k].y(), h->J[k].z(), 0);
		Eigen::Matrix4f G_k_test = Eigen::Matrix4f::Identity();
		G_k_test.block<3, 1>(0, 3) = Eigen::Vector3f(test.x(), test.y(), test.z());
		Eigen::Matrix4f G_k_prime = G_k - G_k_test;*/
		Eigen::Matrix4f G_k_prime = G_k * h->inv_G_zero[k]; // remove the transformation due to the rest pose
		G[k] = G_k_prime;
	}

	//update vertices
	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		Eigen::Vector4f new_vertex = Eigen::Vector4f::Zero();
		for (int k = 0; k < NUM_MANO_JOINTS; k++)
		{
			Eigen::Vector4f bs = Eigen::Vector4f(BS[i].x(), BS[i].y(), BS[i].z(), 0);
			Eigen::Vector4f bp = Eigen::Vector4f(BP[i].x(), BP[i].y(), BP[i].z(), 0);
			new_vertex += h->W(i, k) * G[k] * (h->vertices[i] + bs + bp);
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

	Eigen::MatrixXf joints_matrix = h->joint_regressor * v_matrix; // dimension (NUM_MANO_JOINTS-1) * 3
	for (int j = 0; j < NUM_MANO_JOINTS; j++)
	{
		h->joints[j] = Eigen::Vector4f(joints_matrix(j, 0), joints_matrix(j, 1), joints_matrix(j, 2), 1);
	}

	//update rest parameters
	//h->inv_G_zero = G;
	//h->R_zero = R_new;

	//print out time
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;
	std::cout << "...took " << duration.count() << " ms." << std::endl << std::endl;
}

Eigen::Vector2f computeProjection(Eigen::Vector4f point, SimpleCamera camera)
{
	Eigen::Vector3f result = camera.getK() * Eigen::Vector3f(point.x() / point.z(), point.y() / point.z(), point.z() / point.z());
	//return Eigen::Vector2f(((point.x() * camera.f_x) / (point.z() / camera.aspect)) + camera.m_x,
	//					   ((point.y() * camera.f_y) / point.z()) + camera.m_y);
	return Eigen::Vector2f(result.x(), result.y());
}

std::array<std::array<double, 2>, NUM_OPENPOSE_KEYPOINTS> HandModel::get2DJointLocations(Hand hand, SimpleCamera camera)
{
	std::shared_ptr<ManoHand> h;
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
	std::array<int, NUM_MANO_JOINTS> opIndex = { 0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3 };
	for (int j = 0; j < NUM_MANO_JOINTS; j++)
	{
		Eigen::Vector2f proj = computeProjection(h->joints[j], camera);
		result[opIndex[j]] = { proj.x(), proj.y() };
	}
	// add 5 additional joints for openpose
	std::array<int, 5> opIndexAdd = { 4, 8, 12, 16, 20};
	for (int j = 0; j < 5; j++)
	{
		Eigen::Vector2f proj = computeProjection(h->vertices[ADDITIONAL_JOINTS[j]], camera);
		result[opIndexAdd[j]] = { proj.x(), proj.y() };
	}
	return result;
}

std::array<std::array<float, 2>, NUM_MANO_VERTICES> HandModel::get2DVertexLocations(Hand hand, SimpleCamera camera)
{
	std::shared_ptr<ManoHand> h;
	switch (hand)
	{
	case Hand::RIGHT:
		h = rightHand;
		break;
	case Hand::LEFT:
		h = leftHand;
		break;
	}

	std::array<std::array<float, 2>, NUM_MANO_VERTICES> result{};
	for (int j = 0; j < NUM_MANO_VERTICES; j++)
	{
		Eigen::Vector2f proj = computeProjection(h->vertices[j], camera);
		result[j] = { proj.x(), proj.y() };
	}
	return result;
}

void HandModel::applyTransformation(Eigen::Matrix4f transform, Hand hand)
{
	std::shared_ptr<ManoHand> h;
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

void HandModel::applyTranslation(Eigen::Vector3f t, Hand hand)
{
	std::cout << "Applying Transformation" << std::endl;

	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	transform.block<3, 1>(0, 3) = t;

	applyTransformation(transform, hand);
}

void HandModel::applyScale(float factor, Hand hand)
{
	std::cout << "Applying Scale" << std::endl;

	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity() * factor;
	transform(3, 3) = 1;

	applyTransformation(transform, hand);
}

void HandModel::applyRotation(float alpha, float beta, float gamma, Hand hand)
{
	std::cout << "Applying Rotation" << std::endl;

	Eigen::Matrix3f rot;
	rot = Eigen::AngleAxisf(alpha, Eigen::Vector3f::UnitX())* Eigen::AngleAxisf(beta, Eigen::Vector3f::UnitY())* Eigen::AngleAxisf(gamma, Eigen::Vector3f::UnitZ());
	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	transform.block<3, 3>(0, 0) = rot;

	applyTransformation(transform, hand);
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
			outFile << "v " << v.x() << " " << v.y() << " " << v.z() << " " << rightHand->W(i, 4) * 255 << " " << rightHand->W(i, 5) * 255 << " " << rightHand->W(i, 6) * 255 << std::endl;
			//outFile << "v " << v.x() << " " << v.y() << " " << v.z() << " " << rightHand->W(i, 1) * rightHand->theta[3] * 255 << " " << rightHand->W(i, 1) * rightHand->theta[4] * 255 << " " << rightHand->W(i, 1) * rightHand->theta[5] * 255 << std::endl;
		}
	}
	// left Hand vertices
	if (isVisible_left)
	{
		for (int i = 0; i < NUM_MANO_VERTICES; i++)
		{
			Eigen::Vector4f v = leftHand->vertices[i];
			outFile << "v " << v.x() << " " << v.y() << " " << v.z() << " " << leftHand->W(i, 4)* 255 << " " << leftHand->W(i, 5) * 255 << " " << leftHand->W(i, 6) * 255 << std::endl;
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

void writeCubeVertices(std::ofstream& file, Eigen::VectorXf v, float offset = 0.001)
{

	file << v.x() - offset << " " << v.y() - offset << " " << v.z() + offset << std::endl;
	file << v.x() + offset << " " << v.y() - offset << " " << v.z() + offset << std::endl;
	file << v.x() - offset << " " << v.y() + offset << " " << v.z() + offset << std::endl;
	file << v.x() + offset << " " << v.y() + offset << " " << v.z() + offset << std::endl;
	file << v.x() - offset << " " << v.y() - offset << " " << v.z() - offset << std::endl;
	file << v.x() + offset << " " << v.y() - offset << " " << v.z() - offset << std::endl;
	file << v.x() - offset << " " << v.y() + offset << " " << v.z() - offset << std::endl;
	file << v.x() + offset << " " << v.y() + offset << " " << v.z() - offset << std::endl;
}

void writeCubeFaces(std::ofstream& file, int i, Eigen::Vector3i color, int offset = 0)
{
	int idx = i * 8 + offset;
	file << 4 << " " << idx << " " << idx + 1 << " " << idx + 3 << " " << idx + 2 << " " << color.x() << " " << color.y() << " " << color.z() << std::endl;
	file << 4 << " " << idx + 4 << " " << idx + 5 << " " << idx + 7 << " " << idx + 6 << " " << color.x() << " " << color.y() << " " << color.z() << std::endl;
	file << 4 << " " << idx << " " << idx + 4 << " " << idx + 6 << " " << idx + 2 << " " << color.x() << " " << color.y() << " " << color.z() << std::endl;
	file << 4 << " " << idx + 2 << " " << idx + 3 << " " << idx + 7 << " " << idx + 6 << " " << color.x() << " " << color.y() << " " << color.z() << std::endl;
	file << 4 << " " << idx + 1 << " " << idx + 5 << " " << idx + 7 << " " << idx + 3 << " " << color.x() << " " << color.y() << " " << color.z() << std::endl;
	file << 4 << " " << idx << " " << idx + 1 << " " << idx + 5 << " " << idx + 4 << " " << color.x() << " " << color.y() << " " << color.z() << std::endl;
}

bool HandModel::saveMANOJoints()
{
	std::string output_filename = "jointsMANO.off";
	std::cout << "Saving Joints to " << output_filename << std::endl;

	std::ofstream jointsFile(output_filename);
	if (!jointsFile.is_open()) return false;

	jointsFile << "OFF" << std::endl;
	jointsFile << NUM_MANO_JOINTS * (isVisible_left + isVisible_right) * 8 << " " << NUM_MANO_JOINTS * (isVisible_left + isVisible_right) * 6 << " 0" << std::endl;

	float vertex_offset = std::max((rightHand->joints[0] - rightHand->joints[1]).norm(), (leftHand->joints[0] - leftHand->joints[1]).norm()) / 30;
	// right Hand vertices
	if (isVisible_right)
	{
		for (int i = 0; i < NUM_MANO_JOINTS; i++)
		{
			Eigen::Vector4f v = rightHand->joints[i];
			writeCubeVertices(jointsFile, v, vertex_offset);
		}
	}
	// left Hand vertices
	if (isVisible_left)
	{
		for (int i = 0; i < NUM_MANO_JOINTS; i++)
		{
			Eigen::Vector4f v = leftHand->joints[i];
			writeCubeVertices(jointsFile, v, vertex_offset);
		}

	}
	// right Hand faces
	if (isVisible_right)
	{
		for (int i = 0; i < NUM_MANO_JOINTS; i++)
		{
			writeCubeFaces(jointsFile, i, { 255, 0, 0 });
		}
	}
	// left Hand faces
	if (isVisible_left)
	{
		int offset = isVisible_right * NUM_MANO_JOINTS * 8;
		for (int i = 0; i < NUM_MANO_JOINTS; i++)
		{
			writeCubeFaces(jointsFile, i, {0, 0, 255}, offset);
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
	jointsFile << NUM_OPENPOSE_KEYPOINTS * (isVisible_left + isVisible_right) * 8 << " " << NUM_OPENPOSE_KEYPOINTS * (isVisible_left + isVisible_right) * 6 << " 0" << std::endl;

	float vertex_offset = std::max((rightHand->joints[0] - rightHand->joints[1]).norm(), (leftHand->joints[0] - leftHand->joints[1]).norm()) / 30;

	// right Hand vertices
	jointsFile << "# right hand vertices" << std::endl;
	if (isVisible_right)
	{
		jointsFile << "# MANO" << std::endl;
		for (int i = 0; i < NUM_MANO_JOINTS; i++)
		{
			Eigen::Vector4f v = rightHand->joints[i];
			writeCubeVertices(jointsFile, v, vertex_offset);
		}
		// add 5 additional joints for openpose
		jointsFile << "# OpenPose" << std::endl;
		for (int j = 0; j < 5; j++)
		{
			Eigen::Vector4f v = rightHand->vertices[ADDITIONAL_JOINTS[j]];
			writeCubeVertices(jointsFile, v, vertex_offset);
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
			writeCubeVertices(jointsFile, v, vertex_offset);
		}
		// add 5 additional joints for openpose
		jointsFile << "# OpenPose" << std::endl;
		for (int j = 0; j < 5; j++)
		{
			Eigen::Vector4f v = leftHand->vertices[ADDITIONAL_JOINTS[j]];
			writeCubeVertices(jointsFile, v, vertex_offset);
		}
	}
	// right Hand faces
	jointsFile << "# right hand faces" << std::endl;
	if (isVisible_right)
	{
		for (int i = 0; i < NUM_OPENPOSE_KEYPOINTS; i++)
		{
			writeCubeFaces(jointsFile, i, { 255, 0, 0 });
		}
	}
	// left Hand faces
	jointsFile << "# left hand faces" << std::endl;
	if (isVisible_left)
	{
		int offset = isVisible_right * NUM_OPENPOSE_KEYPOINTS * 8;
		for (int i = 0; i < NUM_OPENPOSE_KEYPOINTS; i++)
		{
			writeCubeFaces(jointsFile, i, { 0, 0, 255 }, offset);
		}
	}

	jointsFile.close();

	return true;
}

bool HandModel::hasAncestor(std::shared_ptr<ManoHand> h, unsigned int joint_index)
{
	if (!(*h->kinematic_tree).count(joint_index))
		return false;
	return (*h->kinematic_tree).at(joint_index) < NUM_MANO_JOINTS;
}

unsigned int HandModel::getAncestor(std::shared_ptr<ManoHand> h, unsigned int joint_index)
{
	if (!(*h->kinematic_tree).count(joint_index))
		return -1;
	return (*h->kinematic_tree).at(joint_index);
}

Eigen::Matrix4f HandModel::computeG(std::shared_ptr<ManoHand> h, unsigned int joint_index)
{
	Eigen::Matrix4f G_k = Eigen::Matrix4f::Identity();

	if (joint_index == 0)
	{
		// rotation defined in axis-angle format
		G_k.block<3, 3>(0, 0) = rodrigues(Eigen::Vector3f(h->theta[0], h->theta[1], h->theta[2]));
		G_k.block<3, 1>(0, 3) = Eigen::Vector3f(h->joints[0].x(), h->joints[0].y(), h->joints[0].z());
		return G_k;
	}

	Eigen::Vector4f child = h->joints[joint_index];
	Eigen::Vector4f parent = h->joints[getAncestor(h, joint_index)];

	G_k.block<3, 3>(0, 0) = rodrigues(Eigen::Vector3f(h->theta[joint_index * 3], h->theta[joint_index * 3 + 1], h->theta[joint_index * 3 + 2]));
	G_k.block<3, 1>(0, 3) = Eigen::Vector3f(child.x(), child.y(), child.z()) - Eigen::Vector3f(parent.x(), parent.y(), parent.z());

	return computeG(h, getAncestor(h, joint_index)) * G_k;

	/*Eigen::Matrix4f G_k = Eigen::Matrix4f::Identity();

	if (joint_index == 0)
	{
		G_k.block<3, 3>(0, 0) = rodrigues(Eigen::Vector3f(h->theta[0], h->theta[1], h->theta[2]));
		G_k.block<3, 1>(0, 3) = Eigen::Vector3f(h->joints[0].x(), h->joints[0].y(), h->joints[0].z());
		return G_k;
	}

	unsigned int current_idx = joint_index;
	std::vector<unsigned int> ancestors;
	ancestors.push_back(current_idx);
	while (hasAncestor(h, current_idx))
	{
		unsigned int j = getAncestor(h, current_idx);
		ancestors.push_back(j);
		current_idx = j;
	}

	//size is larger 1 as root is handled beforehand
	for (int idx = ancestors.size() - 1; idx >= 0; idx--)
	{
		int j = ancestors[idx];
		Eigen::Matrix4f A = Eigen::Matrix4f::Identity();

		A.block<3, 3>(0, 0) = rodrigues(Eigen::Vector3f(h->theta[j * 3], h->theta[j * 3 + 1], h->theta[j * 3 + 2]));
		A.block<3, 1>(0, 3) = Eigen::Vector3f(h->joints[j].x(), h->joints[j].y(), h->joints[j].z());

		G_k *= A;
	}
	return G_k;*/
}

Eigen::Vector<float, 9> HandModel::computeR(std::shared_ptr<ManoHand> h, unsigned int j)
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

	Eigen::Matrix3f result = Eigen::Matrix3f::Identity() + skew * sin(w_length) + (skew * skew) * (1 - cos(w_length));
	return result;
}

void HandModel::display(const char* filename, Hand hand)
{
	SimpleCamera c = SimpleCamera();
	std::array<std::array<float, 2>, NUM_OPENPOSE_KEYPOINTS> j = get2DJointLocations(Hand::RIGHT, c);
	std::array<std::array<float, 2>, NUM_MANO_VERTICES> v = get2DVertexLocations(Hand::RIGHT, c);

	cv::Mat frame = Parser::readImageCV(filename);

	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		//cv::Point pos(int((v[i][0] / 2 + 0.5f) * frame.cols), int((v[i][1] / 2 + 0.5f) * frame.rows));
		cv::Point pos((int)v[i][0] / c.image_width * frame.cols, (int)v[i][1] / c.image_height * frame.rows);
		cv::circle(frame, pos, 1, cv::Scalar(0, 255, 0), -1, cv::FILLED);
	}

	for (int i = 0; i < NUM_OPENPOSE_KEYPOINTS; i++)
	{
		//cv::Point pos(int((k[i][0] / 2 + 0.5f) * frame.cols), int((k[i][1] / 2 + 0.5f) * frame.rows));
		cv::Point pos((int)j[i][0] / c.image_width * frame.cols, (int)j[i][1] / c.image_height * frame.rows);
		cv::circle(frame, pos, 1, cv::Scalar(0, 0, 255), -1, cv::FILLED);
		cv::putText(frame, std::to_string(i), pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
		//std::cout << pos << std::endl;
	}

	//print a gitter for testing
	/*for (int j = 0; j < 5; j++)
	{
		for (int i = 0; i < 5; i++)
		{
			Eigen::Vector3f result = c.getK() * Eigen::Vector3f(i/10.0, j/10.0, 1);
			cv::Point pos(result.x() / c.image_width * frame.cols, result.y() / c.image_height * frame.rows);
			std::cout << pos << std::endl;
			cv::circle(frame, pos, 1, cv::Scalar(255, 0, 0), -1, cv::FILLED);
			cv::putText(frame, std::to_string(i) + ", " + std::to_string(j), pos, cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(255, 0, 0));
		}
	}*/

	cv::imshow("Hand", frame);
	cv::waitKey(0);
}