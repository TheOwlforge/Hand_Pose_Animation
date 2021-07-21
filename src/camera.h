#pragma once

struct SimpleCamera {
	float f_x;
	float f_y;
	float m_x;
	float m_y;

	Eigen::Matrix3f getK() {
		Eigen::Matrix3f K;
		K << f_x, 0,   m_x,
			 0,   f_y, m_y,
			 0,   0,   1;
		return K;
	}
};