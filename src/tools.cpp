#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
   VectorXd rmse(4);
   rmse << 0., 0., 0., 0.;

   int size = estimations.size();

   if (size == 0)
   {
      std::cout << "Detected estimation with size 0!!!";
      return rmse;
   }

   if (size != ground_truth.size())
   {
      std::cout << "Estimation size does not match the ground truth!!!";
      return rmse;
   }

   for (size_t i = 0; i < size; i++)
   {
      // VectorXd residual_squared = (estimations[i] - ground_truth[i]).array().pow(2);
      VectorXd residual = estimations[i] - ground_truth[i];
      VectorXd residual_squared = (residual.array() * residual.array());

      rmse += residual_squared;
   }

   // Calculate mean
   rmse /= size;

   return rmse.array().sqrt();
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state)
{
   Eigen::MatrixXd Hj(3, 4);

   double px = x_state(0);
   double py = x_state(1);
   double vx = x_state(2);
   double vy = x_state(3);
   double px_2_py_2_sum = (px * px) + (py * py);
   double vxpy_vypx_diff = (vx * py) - (vy * px);

   if (fabs(px_2_py_2_sum) < 0.0001)
   {
      std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
      return Hj;
   }

   double range = sqrt(px_2_py_2_sum);
   double px2_py2_32 = std::pow(px_2_py_2_sum, 1.5);

   Hj << (px / range), (py / range), 0., 0.,
       -(py / px_2_py_2_sum), (px / px_2_py_2_sum), 0., 0.,
       ((py * vxpy_vypx_diff) / (px_2_py_2_sum * range)), ((px * vxpy_vypx_diff) / (px_2_py_2_sum * range)), (px / range), (py / range);

   return Hj;
}

void Tools::initF(Eigen::MatrixXd &F, double dt)
{
   F(0, 2) = dt;
   F(1, 3) = dt;
}

void Tools::initQ(Eigen::MatrixXd &Q, double dt, double covariance_ax, double covariance_ay)
{
   double dt_2 = dt * dt;
   double dt_3 = dt_2 * dt;
   double dt_4 = dt_3 * dt;

   Q << (dt_4 * covariance_ax * 0.25), 0., (dt_3 * covariance_ax * 0.5), 0.,
       0., (dt_4 * covariance_ay * 0.25), 0., (dt_3 * covariance_ay * 0.5),
       (dt_3 * covariance_ax * 0.5), 0., (dt_2 * covariance_ax), 0.,
       0., (dt_3 * covariance_ay * 0.5), 0., (dt_2 * covariance_ay);
}

Eigen::VectorXd Tools::ConvertToPolar(Eigen::VectorXd &x)
{
   Eigen::VectorXd x_polar(3);

   double px = x(0);
   double py = x(1);
   double vx = x(2);
   double vy = x(3);

   if (px == 0)
   {
      std::cout << "Unable to convert to polar. px is 0!!!";
      return x_polar;
   }

   // rho
   double range = sqrt((px * px) + (py * py));

   if (fabs(range) == 0)
   {
      std::cout << "Unable to convert to polar. range is 0!!!";
      return x_polar;
   }

   // phi
   double bearing = atan2(py, px);

   // rho dot
   double range_rate = ((px * vx) + (py * vy)) / range;

   x_polar << range,
       bearing,
       range_rate;
   return x_polar;
}
