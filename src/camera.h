#pragma once

struct SimpleCamera {
	float f_x = 1.92;
	float f_y = 1.92;
	float sensor_width = 1.33107;
	float sensor_height = 0.7487;
	float image_width = 1280;
	float image_height = 720;
	float m_x = image_width / 2;
	float m_y = image_height / 2;

	Eigen::Matrix3f getK() {
		Eigen::Matrix3f K;
		K << f_x * image_width / sensor_width, 0,   m_x,
			 0,   f_y * image_height / sensor_height, m_y,
			 0,   0,   1;
		return K;
	}
};