#include <iostream>
#include "kalman_filter.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in)
{
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

/**
 * Recap
 * 
 * x: Estimate
 * P: Uncertainty covariance
 * F: State transition matrix
 * u: motion vector
 * Z: measurement
 * H: measurement function
 * R: measurement covariance matrix
 * y: error
 * S: System uncertainty
 * K: Kalman gain
 * I: identity matrix
 */

void KalmanFilter::Predict()
{
  // x' = F.x + u
  x_ = F_ * x_;
  // P' = F.P.F_transpose
  P_ = (F_ * P_ * F_.transpose()) + Q_;
}

void KalmanFilter::Update(const VectorXd &z)
{
  /**
   * update the state by using Kalman Filter equations
   */
  MatrixXd H_transpose = H_.transpose();

  // y = Z - H.x
  MatrixXd y = z - (H_ * x_);
  // S = H.P.H_transpose
  MatrixXd S = (H_ * P_ * H_transpose) + R_;
  // K = P.H_transpose.S_inverse
  MatrixXd K = P_ * H_transpose * S.inverse();
  // x' = x + (K.y)
  x_ = x_ + (K * y);
  // P = (I - K.H) . P
  P_ = (MatrixXd::Identity(K.rows(), K.rows()) - (K * H_)) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z)
{
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
  Tools tools;
  H_ = tools.CalculateJacobian(x_);
  MatrixXd H_transpose = H_.transpose();

  // y = Z - H.x
  MatrixXd y = z - tools.ConvertToPolar(x_);
  // S = H.P.H_transpose
  MatrixXd S = (H_ * P_ * H_transpose) + R_;
  // K = P.H_transpose.S_inverse
  MatrixXd K = P_ * H_transpose * S.inverse();
  // x' = x + (K.y)
  x_ = x_ + (K * y);
  // P = (I - K.H) . P
  P_ = (MatrixXd::Identity(K.rows(), K.rows()) - (K * H_)) * P_;
}
