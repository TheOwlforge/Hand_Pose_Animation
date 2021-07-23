#pragma once
#include "pch.h"
#include "mano.h"
#include "parser.hpp"

// declare static variables for parser once globally
// cannot be done in parser.hpp to avoid errors when including in multiple files
std::vector<cv::VideoCapture> Parser::caps;
std::vector<cv::VideoWriter> Parser::videos;

int main(int argc, char* argv[])
{
	Parser::testJson();

	std::cout << sizeof(float) << ", " << sizeof(ManoHand) << std::endl;

	HandModel* hands = new HandModel("mano/model/mano_right.json", "mano/model/mano_left.json");

	std::array<double, MANO_THETA_SIZE> rnd1 = std::array<double, MANO_THETA_SIZE>();
	std::array<double, MANO_THETA_SIZE> rnd3 = std::array<double, MANO_THETA_SIZE>();
	std::array<double, MANO_BETA_SIZE> rnd2 = std::array<double, MANO_BETA_SIZE>();

	//HandModel::fillRandom(&rnd1, 0.5);
	//rnd1[0] = 2;
	/*rnd1[5] = 2;
	rnd1[8] = 0.2;
	rnd1[11] = 0.1;
	rnd1[32] = 2;
	rnd1[23] = 2;
	rnd1[41] = 2;*/

	/*for (uint16_t i = 0; i < (NUM_MANO_JOINTS - 1) * 3; i++)
	{
		rnd1[i + 3] = hands->rightHand->hands_mean[i];
	}*/

	hands->setModelParameters(rnd1, rnd2, Hand::RIGHT);
	//hands->reset();
	//hands->setModelParameters(rnd1, rnd2, Hand::RIGHT);
	//hands->applyTransformation(Eigen::Vector3f(0, 0, 0.2), Hand::LEFT);


	//hands->applyScale(0.70, Hand::RIGHT);

	//hands->applyRotation(-0.5 * M_PI, 0, 0, Hand::RIGHT);
	hands->applyTranslation(Eigen::Vector3f(0, 0, 5), Hand::RIGHT);
	hands->isVisible_left = false;

	hands->saveVertices();
	hands->saveMANOJoints();
	hands->saveOPJoints();

	std::array<std::array<float, 2>, NUM_OPENPOSE_KEYPOINTS> k = hands->get2DJointLocations(Hand::RIGHT, SimpleCamera());
	std::array<std::array<float, 2>, NUM_MANO_VERTICES> v = hands->get2DVertexLocations(Hand::RIGHT, SimpleCamera());

	cv::Mat frame = Parser::readImageCV("samples/pictures/onehand1.png");

	for (int i = 0; i < NUM_MANO_VERTICES; i++)
	{
		//cv::Point pos(int((v[i][0] / 2 + 0.5f) * frame.cols), int((v[i][1] / 2 + 0.5f) * frame.rows));
		cv::Point pos((int)v[i][0] / 2, (int)v[i][1] / 2);
		cv::circle(frame, pos, 1, cv::Scalar(0, 255, 0), -1, cv::FILLED);
	}

	for (int i = 0; i < NUM_OPENPOSE_KEYPOINTS; i++)
	{
		//cv::Point pos(int((k[i][0] / 2 + 0.5f) * frame.cols), int((k[i][1] / 2 + 0.5f) * frame.rows));
		cv::Point pos((int)k[i][0], (int)k[i][1]);
		cv::circle(frame, pos, 1, cv::Scalar(0, 0, 255), -1, cv::FILLED);
		cv::putText(frame, std::to_string(i), pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
		std::cout << pos << std::endl;
	}

	for (int j = 0; j < 5; j++)
	{
		for (int i = 0; i < 5; i++)
		{
			SimpleCamera c = SimpleCamera();
			Eigen::Vector3f result = c.getK() * Eigen::Vector3f(i/20.0, j/20.0, 1);
			cv::Point pos(result.x() / c.image_width * frame.cols, result.y() / c.image_height * frame.rows);
			std::cout << pos << std::endl;
			cv::circle(frame, pos, 1, cv::Scalar(255, 0, 0), -1, cv::FILLED);
			cv::putText(frame, std::to_string(i) + ", " + std::to_string(j), pos, cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(255, 0, 0));
		}
	}

	/*float x = 0.3f;
	float y = 0.3f;
	float z = 3.0f;
	cv::Point pos(int((x * 1.92 / z) * frame.cols), int((y * 1.92 / z) * frame.rows));
	cv::circle(frame, pos, 1, (0, 255, 255), -1, cv::FILLED);

	std::cout << pos << std::endl;*/

	cv::imshow("test", frame);
	cv::waitKey(0);

	delete hands;
}