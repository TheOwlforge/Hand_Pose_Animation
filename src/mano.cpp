#pragma once
#include "pch.h"
#include "mano.h"
#include "parser.hpp"

HandModel::HandModel(std::string rightHand_filename, std::string leftHand_filename)
{
	rightHand = Parser::readJsonMANO(rightHand_filename);
	leftHand = Parser::readJsonMANO(leftHand_filename);

	for (int i = 0; i < NUM_MANO_FACES; i++)
	{
		leftHand->face_indices[i] = leftHand->face_indices[i] + (NUM_MANO_VERTICES) * Eigen::Vector3i::Ones();
	}

	rightHand->vertices = rightHand->T;
	leftHand->vertices = leftHand->T;
}

void HandModel::applyTransformation(Eigen::Vector3f t, Hand hand)
{
	Eigen::Translation3f translation = Eigen::Translation3f(t);
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
		h->vertices[i] = translation * h->vertices[i];
	}
}

bool HandModel::saveToObj(std::string output_filename)
{
	std::cout << "Saving Hands Model to " << output_filename << std::endl;
	std::ofstream outFile(output_filename);
	if (!outFile.is_open()) return false;

	// right Hand vertices
	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		Eigen::Vector3f v = rightHand->vertices[i];
		outFile << "v " << v.x() << " " << v.y() << " " << v.z() << std::endl;
	}
	// left Hand vertices
	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		Eigen::Vector3f v = leftHand->vertices[i];
		outFile << "v " << v.x() << " " << v.y() << " " << v.z() << std::endl;
	}
	// right Hand faces
	for (int i = 0; i < NUM_MANO_FACES; i++)
	{
		Eigen::Vector3i f = rightHand->face_indices[i];
		outFile << "f " << f.x() << " " << f.y() << " " << f.z() << std::endl;
	}
	// left Hand faces
	for (int i = 0; i < NUM_MANO_FACES; i++)
	{
		Eigen::Vector3i f = leftHand->face_indices[i];
		outFile << "f " << f.x() << " " << f.y() << " " << f.z() << std::endl;
	}
		

	outFile.close();
	return true;
}