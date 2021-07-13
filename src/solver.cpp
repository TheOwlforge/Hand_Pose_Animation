#include "parser.hpp"
#include "Eigen.h"
#include "ceres/ceres.h"
#include <iostream>
#include <fstream>

#define POSEDIM      12;
#define SHAPEDIM     10;
#define GLOBALROTDIM 3;

#define MAX_ITERATIONS 50;


struct EnergyFunction
{
  EnergyFunction(const float weight, const float x, const float y)
	{
	}

	template<typename T>
	bool operator()(const T* const pose, const T* const shape, T* globalRot) const
	{
	  /*
	    TODO: Implement energy function

	    1. Get surface from MANO model given pose, shape
	    2. Sample joints from MANO surface corresponding to OpenPose
            3. Project joints with camera intrinsincs
	  */

  	  T energyFunction = 0.0;

	  // Prior 1: Try Gaussian Prior

	  T posePrior = pow(pose, 2).sum();
	  T shapePrior = pow(shape, 2).sum();

	  T prior = posePrior + shapePrior;

	  residual[0] += prior;

	  return true;
	}
};
 

int main(int argc, char **argv) {
  // Read keypoints

  if (argc > 1) {
    const std::string filename = argv[1];
  }
  else {
      const std::string filename = "samples/webcam_examples/000000000000_keypoints.json";
  }

  std::array<std::array<float, NUM_KEYPOINTS * 3>, 2> keypoints = Parser::readJsonCV(filename);

  std::array<float, NUM_KEYPOINTS * 3> left_keypoints = keypoints[0];
  std::array<float, NUM_KEYPOINTS * 3> right_keypoints = keypoints[1];

  // Define initial values for parameters of pose, shape, global rotation

  const VectorXf poseInitial = VectorXf::Random(POSEDIM);
  const VectorXf shapeInitial = VectorXf::Random(SHAPEDIM);
  const VectorXf globalRotInitial = VectorXf::Random(GLOBALROTDIM);

  // Assign initial values to parameters

  VectorXf pose = poseInitial;
  VectorXf shape = shapeInitial;
  VectorXf globalRotation = globalRotInitial;

  // Define Ceres problem and add residual blocks

  ceres::Problem problem;

  for (int i = 0; i < NUM_KEYPOINTS; i++) {
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<EnergyFunction, 1, 1, 1, 1>(
										new EnergyFunction(keypoints[0][3*i+2], keypoints[0][3*i], keypoints[0][3*i+1])),
	  nullptr, &pose, &shape, &globalRotation
      );
  }

  // Ceres settings
  
  ceres::Solver::Options options;
  options.max_num_iterations = MAX_ITERATIONS;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << std::endl;

  // Output the final pose, shape and global rotation parameters
  
  std::cout << "Initial pose: " << poseInitial << "shape: " << shapeInitial << "globalRot: " <<globalRotInitial << std::endl;
  std::cout << "Final pose: " << pose << "shape: " << shape << "globalRot: " << globalRot << std::endl;

  system("pause");
  return 0;
}
