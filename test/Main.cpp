#include <gtest/gtest.h>
// #include "../src/kalman_filter.h"
#include "../src/Eigen/Dense"
#include "../src/tools.h"
#include "../src/kalman_filter.h"
#include <math.h>

TEST(KalmanFilter, RootMeanSquaredError)
{
    std::vector<Eigen::VectorXd> estimations;
    std::vector<Eigen::VectorXd> ground_truth;
    Eigen::VectorXd rmse_expected(4);
    rmse_expected << 0.1, 0.1, 0.1, 0.1;

    // the input list of estimations
    Eigen::VectorXd e(4);
    e << 1, 1, 0.2, 0.1;
    estimations.push_back(e);
    e << 2, 2, 0.3, 0.2;
    estimations.push_back(e);
    e << 3, 3, 0.4, 0.3;
    estimations.push_back(e);

    // the corresponding list of ground truth values
    Eigen::VectorXd g(4);
    g << 1.1, 1.1, 0.3, 0.2;
    ground_truth.push_back(g);
    g << 2.1, 2.1, 0.4, 0.3;
    ground_truth.push_back(g);
    g << 3.1, 3.1, 0.5, 0.4;
    ground_truth.push_back(g);

    Tools tools;
    Eigen::VectorXd rmse_actual = tools.CalculateRMSE(estimations, ground_truth);

    ASSERT_TRUE(rmse_actual.isApprox(rmse_expected)) << "Expected: " << rmse_expected << std::endl
                                                     << "Actual: " << rmse_actual << std::endl;
}

TEST(KalmanFilter, Jacobian)
{
    Eigen::VectorXd x_predicted(4);
    x_predicted << 1, 2, 0.2, 0.4;

    Tools tools;
    Eigen::MatrixXd Hj_actual = tools.CalculateJacobian(x_predicted);

    Eigen::MatrixXd Hj_expected(3, 4);
    Hj_expected << 0.447214, 0.894427, 0, 0,
        -0.4, 0.2, 0, 0,
        0, 0, 0.447214, 0.894427;

    ASSERT_TRUE(Hj_actual.isApprox(Hj_expected, 0.001)) << "Expected: " << Hj_expected << std::endl
                                                        << "Actual: " << Hj_actual << std::endl;
}

// TEST(KalmanFilter, FusionEKF)
// {
//     ASSERT_TRUE(false) << "TODO: Implement";
// }

TEST(KalmanFilter, KalmanFilter1DConstantMotion)
{
    // design the KF with 1D motion
    Eigen::VectorXd x = Eigen::VectorXd(2);
    x << 0., 0.;

    Eigen::MatrixXd P = Eigen::MatrixXd(2, 2);
    P << 1000., 0., 0., 1000.;

    Eigen::MatrixXd F = Eigen::MatrixXd(2, 2);
    F << 1., 1., 0., 1.;

    Eigen::MatrixXd H = Eigen::MatrixXd(1, 2);
    H << 1., 0.;

    Eigen::MatrixXd R = Eigen::MatrixXd(1, 1);
    R << 1.;

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(2, 2);

    Eigen::MatrixXd Q = Eigen::MatrixXd(2, 2);
    Q << 0., 0., 0., 0.;

    KalmanFilter kf;

    kf.Init(x, P, F, H, R, Q);

    Eigen::VectorXd measurement(1);
    Eigen::VectorXd x_expected = Eigen::VectorXd(2);
    Eigen::MatrixXd P_expected = Eigen::MatrixXd(2, 2);

    measurement << 1.;
    x_expected << 0.999001, 0.;
    P_expected << 1001., 1000., 1000., 1000.;

    kf.Update(measurement);
    kf.Predict();

    ASSERT_TRUE(kf.x_.isApprox(x_expected, 0.1)) << "x_actual: " << kf.x_;
    ASSERT_TRUE(kf.P_.isApprox(P_expected, 0.1)) << "P_actual: " << kf.P_;

    measurement << 2.;
    x_expected << 2.998, 0.999002;
    P_expected << 4.99002, 2.99302, 2.99302, 1.99501;

    kf.Update(measurement);
    kf.Predict();

    ASSERT_TRUE(kf.x_.isApprox(x_expected, 0.1)) << "x_actual: " << kf.x_;
    ASSERT_TRUE(kf.P_.isApprox(P_expected, 0.1)) << "P_actual: " << kf.P_;

    measurement << 3.;
    x_expected << 3.99967, 1.;
    P_expected << 2.33189, 0.999168, 0.999168, 0.499501;

    kf.Update(measurement);
    kf.Predict();

    ASSERT_TRUE(kf.x_.isApprox(x_expected, 0.1)) << "x_actual: " << kf.x_;
    ASSERT_TRUE(kf.P_.isApprox(P_expected, 0.1)) << "P_actual: " << kf.P_;
}

TEST(KalmanFilter, KalmanFilter2DCartesian)
{
    // design the KF with 1D motion
    Eigen::VectorXd x = Eigen::VectorXd(4);
    x << 0.463227, 0.607415, 0., 0.;

    Eigen::MatrixXd P = Eigen::MatrixXd(4, 4);
    P << 1., 0., 0., 0.,
        0., 1., 0., 0.,
        0., 0., 1000., 0.,
        0., 0., 0., 1000.;

    Eigen::MatrixXd F = Eigen::MatrixXd(4, 4);
    F << 1., 0., 0., 0.,
        0., 1., 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 1.;

    Eigen::MatrixXd H = Eigen::MatrixXd(2, 4);
    H << 1., 0., 0., 0.,
        0., 1., 0., 0.;

    Eigen::MatrixXd R = Eigen::MatrixXd(2, 2);
    R << 0.0225, 0.,
        0., 0.0225;

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(4, 4);

    Eigen::MatrixXd Q = Eigen::MatrixXd(4, 4);
    Q << 0., 0., 0., 0.,
        0., 0., 0., 0.,
        0., 0., 0., 0.,
        0., 0., 0., 0.;

    KalmanFilter kf;

    kf.Init(x, P, F, H, R, Q);

    double noise_ax = 5;
    double noise_ay = 5;

    Tools tools;

    double timestamp = 1477010443000000;
    double dt = 0.;
    Eigen::VectorXd measurement(2);
    Eigen::VectorXd x_expected = Eigen::VectorXd(4);
    Eigen::MatrixXd P_expected = Eigen::MatrixXd(4, 4);

    dt = (1477010443100000 - timestamp) / 1000000.0;
    timestamp = 1477010443100000;
    measurement << 0.968521, 0.40545;
    x_expected << 0.96749, 0.405862, 4.58427, -1.83232;
    P_expected << 0.0224541, 0, 0.204131, 0,
        0, 0.0224541, 0, 0.204131,
        0.204131, 0, 92.7797, 0,
        0, 0.204131, 0, 92.7797;

    tools.initF(kf.F_, dt);
    tools.initQ(kf.Q_, dt, noise_ax, noise_ay);

    kf.Predict();
    kf.Update(measurement);

    ASSERT_TRUE(kf.x_.isApprox(x_expected, 0.1)) << "x_actual: " << kf.x_;
    ASSERT_TRUE(kf.P_.isApprox(P_expected, 0.1)) << "P_actual: " << kf.P_;

    dt = (1477010443200000 - timestamp) / 1000000.0;
    timestamp = 1477010443200000;
    measurement << 0.947752, 0.636824;
    x_expected << 0.958365, 0.627631, 0.110368, 2.04304;
    P_expected << 0.0220006, 0, 0.210519, 0,
        0, 0.0220006, 0, 0.210519,
        0.210519, 0, 4.08801, 0,
        0, 0.210519, 0, 4.08801;

    tools.initF(kf.F_, dt);
    tools.initQ(kf.Q_, dt, noise_ax, noise_ay);

    kf.Predict();
    kf.Update(measurement);

    ASSERT_TRUE(kf.x_.isApprox(x_expected, 0.1)) << "x_actual: " << kf.x_;
    ASSERT_TRUE(kf.P_.isApprox(P_expected, 0.1)) << "P_actual: " << kf.P_;

    dt = (1477010443300000 - timestamp) / 1000000.0;
    timestamp = 1477010443300000;
    measurement << 1.42287, 0.264328;
    x_expected << 1.34291, 0.364408, 2.32002, -0.722813;
    P_expected << 0.0185328, 0, 0.109639, 0,
        0, 0.0185328, 0, 0.109639,
        0.109639, 0, 1.10798, 0,
        0, 0.109639, 0, 1.10798;

    tools.initF(kf.F_, dt);
    tools.initQ(kf.Q_, dt, noise_ax, noise_ay);

    kf.Predict();
    kf.Update(measurement);

    ASSERT_TRUE(kf.x_.isApprox(x_expected, 0.1)) << "x_actual: " << kf.x_;
    ASSERT_TRUE(kf.P_.isApprox(P_expected, 0.1)) << "P_actual: " << kf.P_;
}

// TEST(KalmanFilter, KalmanFilter2DPolar)
// {
//     /**
//      * Unable to test this one. We're lacking a reference input
//      */

//     ASSERT_TRUE(false) << "TODO: Implement";

//     // // design the KF with 1D motion
//     // Eigen::VectorXd x = Eigen::VectorXd(4);
//     // x << 0.463227, 0.607415, 0., 0.;

//     // Eigen::MatrixXd P = Eigen::MatrixXd(4, 4);
//     // P << 1., 0., 0., 0.,
//     //     0., 1., 0., 0.,
//     //     0., 0., 1000., 0.,
//     //     0., 0., 0., 1000.;

//     // Eigen::MatrixXd F = Eigen::MatrixXd(4, 4);
//     // F << 1., 0., 0., 0.,
//     //     0., 1., 0., 0.,
//     //     0., 0., 1., 0.,
//     //     0., 0., 0., 1.;

//     // Eigen::MatrixXd H = Eigen::MatrixXd(2, 4);
//     // H << 1., 0., 0., 0.,
//     //     0., 1., 0., 0.;

//     // Eigen::MatrixXd R = Eigen::MatrixXd(3, 3);
//     // R << 0.0225, 0., 0.,
//     //     0., 0.0225, 0.,
//     //     0., 0., 0.0225;

//     // Eigen::MatrixXd I = Eigen::MatrixXd::Identity(4, 4);

//     // Eigen::MatrixXd Q = Eigen::MatrixXd(4, 4);
//     // Q << 0., 0., 0., 0.,
//     //     0., 0., 0., 0.,
//     //     0., 0., 0., 0.,
//     //     0., 0., 0., 0.;

//     // KalmanFilter kf;

//     // kf.Init(x, P, F, H, R, Q);

//     // double noise_ax = 5;
//     // double noise_ay = 5;

//     // Tools tools;

//     // double timestamp = 1477010443000000;
//     // double dt = 0.;
//     // Eigen::VectorXd measurement(2);
//     // Eigen::VectorXd x_expected = Eigen::VectorXd(4);
//     // Eigen::MatrixXd P_expected = Eigen::MatrixXd(4, 4);

//     // dt = (1477010443100000 - timestamp) / 1000000.0;
//     // timestamp = 1477010443100000;
//     // measurement << 0.968521, 0.40545;
//     // x_expected << 0.96749, 0.405862, 4.58427, -1.83232;
//     // P_expected << 0.0224541, 0, 0.204131, 0,
//     //     0, 0.0224541, 0, 0.204131,
//     //     0.204131, 0, 92.7797, 0,
//     //     0, 0.204131, 0, 92.7797;

//     // tools.initF(kf.F_, dt);
//     // tools.initQ(kf.Q_, dt, noise_ax, noise_ay);

//     // kf.Predict();
//     // kf.UpdateEKF(measurement);

//     // ASSERT_TRUE(kf.x_.isApprox(x_expected, 0.1)) << "x_actual: " << kf.x_;
//     // ASSERT_TRUE(kf.P_.isApprox(P_expected, 0.1)) << "P_actual: " << kf.P_;

//     // dt = (1477010443200000 - timestamp) / 1000000.0;
//     // timestamp = 1477010443200000;
//     // measurement << 0.947752, 0.636824;
//     // x_expected << 0.958365, 0.627631, 0.110368, 2.04304;
//     // P_expected << 0.0220006, 0, 0.210519, 0,
//     //     0, 0.0220006, 0, 0.210519,
//     //     0.210519, 0, 4.08801, 0,
//     //     0, 0.210519, 0, 4.08801;

//     // tools.initF(kf.F_, dt);
//     // tools.initQ(kf.Q_, dt, noise_ax, noise_ay);

//     // kf.Predict();
//     // kf.UpdateEKF(measurement);

//     // ASSERT_TRUE(kf.x_.isApprox(x_expected, 0.1)) << "x_actual: " << kf.x_;
//     // ASSERT_TRUE(kf.P_.isApprox(P_expected, 0.1)) << "P_actual: " << kf.P_;

//     // dt = (1477010443300000 - timestamp) / 1000000.0;
//     // timestamp = 1477010443300000;
//     // measurement << 1.42287, 0.264328;
//     // x_expected << 1.34291, 0.364408, 2.32002, -0.722813;
//     // P_expected << 0.0185328, 0, 0.109639, 0,
//     //     0, 0.0185328, 0, 0.109639,
//     //     0.109639, 0, 1.10798, 0,
//     //     0, 0.109639, 0, 1.10798;

//     // tools.initF(kf.F_, dt);
//     // tools.initQ(kf.Q_, dt, noise_ax, noise_ay);

//     // kf.Predict();
//     // kf.UpdateEKF(measurement);

//     // ASSERT_TRUE(kf.x_.isApprox(x_expected, 0.1)) << "x_actual: " << kf.x_;
//     // ASSERT_TRUE(kf.P_.isApprox(P_expected, 0.1)) << "P_actual: " << kf.P_;
// }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
