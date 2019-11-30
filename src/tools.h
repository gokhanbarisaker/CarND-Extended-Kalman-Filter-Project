#ifndef TOOLS_H_
#define TOOLS_H_

#include <vector>
#include "Eigen/Dense"

class Tools
{
public:
  /**
   * Constructor.
   */
  Tools();

  /**
   * Destructor.
   */
  virtual ~Tools();

  /**
   * A helper method to calculate RMSE.
   */
  Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations,
                                const std::vector<Eigen::VectorXd> &ground_truth);

  /**
   * A helper method to calculate Jacobians.
   */
  Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd &x_state);

  /**
   * 
   */
  void initF(Eigen::MatrixXd &F, double dt);

  /**
   * 
   */
  void initQ(Eigen::MatrixXd &Q, double dt, double covariance_ax, double covariance_ay);

  /**
   * Converts cartesian (px,py,vx,vy) to polar (rho, pho, rho_dot)
   */
  Eigen::VectorXd ConvertToPolar(Eigen::VectorXd &x);

private:
  double normalizeAngle(double angle);
};

#endif // TOOLS_H_
