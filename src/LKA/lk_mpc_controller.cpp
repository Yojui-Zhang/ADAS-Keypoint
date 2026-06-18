#include "lk_mpc_controller.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "lk_math.h"

namespace lane_keeping {
namespace internal {
namespace {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;

double Dot3(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 Add3(const Vec3& a, const Vec3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3 Sub3(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 Scale3(const Vec3& a, double scale) {
    return {a[0] * scale, a[1] * scale, a[2] * scale};
}

Vec3 MatVec3(const Mat3& a, const Vec3& x) {
    Vec3 out{};
    for (std::size_t i = 0; i < 3; ++i) {
        out[i] = a[i][0] * x[0] + a[i][1] * x[1] + a[i][2] * x[2];
    }
    return out;
}

Mat3 MatMul3(const Mat3& a, const Mat3& b) {
    Mat3 out{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                acc += a[i][k] * b[k][j];
            }
            out[i][j] = acc;
        }
    }
    return out;
}

Mat3 Transpose3(const Mat3& a) {
    Mat3 out{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            out[i][j] = a[j][i];
        }
    }
    return out;
}

Mat3 AddMat3(const Mat3& a, const Mat3& b) {
    Mat3 out{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    return out;
}

Mat3 ScaleMat3(const Mat3& a, double scale) {
    Mat3 out{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            out[i][j] = a[i][j] * scale;
        }
    }
    return out;
}

Mat3 Outer3(const Vec3& a, const Vec3& b) {
    Mat3 out{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            out[i][j] = a[i] * b[j];
        }
    }
    return out;
}

}  // namespace

MpcSteerResult ComputeMpcSteering(double cte_m,
                                  double heading_err_rad,
                                  double curvature_m_inv,
                                  double last_steer_rad,
                                  const ControlConfig& cfg) {
    MpcSteerResult out;

    const double v = std::max(0.0, static_cast<double>(cfg.velocity_mps));
    const double dt = std::max(1e-3, static_cast<double>(cfg.dt_s));
    const double wheelbase = std::max(1e-3, static_cast<double>(cfg.wheel_base_m));
    const double v_gain = (v * dt) / wheelbase;
    const double curvature = cfg.enable_feedforward ? curvature_m_inv : 0.0;

    // State convention matches LKA: cte/head_err are path-relative errors in
    // vehicle coordinates (x forward, y left), and steer is the LKA raw steer.
    const Mat3 A{{
        {{1.0, v * dt, 0.0}},
        {{0.0, 1.0, -v_gain}},
        {{0.0, 0.0, 1.0}},
    }};
    const Vec3 B{0.0, -v_gain, 1.0};
    const Vec3 c{0.0, v * curvature * dt, 0.0};

    Mat3 Q{};
    Q[0][0] = std::max(0.0, static_cast<double>(cfg.mpc_q_cte));
    Q[1][1] = std::max(0.0, static_cast<double>(cfg.mpc_q_heading));
    Q[2][2] = std::max(0.0, static_cast<double>(cfg.mpc_q_steer));

    Mat3 P = Q;
    P[0][0] *= 4.0;
    P[1][1] *= 4.0;
    P[2][2] *= 2.0;
    Vec3 s{0.0, 0.0, 0.0};

    const double R = std::max(1e-6, static_cast<double>(cfg.mpc_r_steer_rate));
    Vec3 first_K{0.0, 0.0, 0.0};
    double first_k = 0.0;
    const std::size_t horizon =
        static_cast<std::size_t>(std::max(1, cfg.mpc_horizon));

    for (std::size_t step = 0; step < horizon; ++step) {
        const Mat3 P_prev = P;
        const Vec3 s_prev = s;
        const Vec3 PB = MatVec3(P_prev, B);
        const double G = R + Dot3(B, PB);
        if (G < 1e-9) {
            return out;
        }

        const Mat3 PA = MatMul3(P_prev, A);
        Vec3 BtPA{};
        for (std::size_t j = 0; j < 3; ++j) {
            BtPA[j] = B[0] * PA[0][j] + B[1] * PA[1][j] + B[2] * PA[2][j];
        }
        const Vec3 K = Scale3(BtPA, 1.0 / G);

        const Vec3 Pc_plus_s = Add3(MatVec3(P_prev, c), s_prev);
        const double k_ff = Dot3(B, Pc_plus_s) / G;

        first_K = K;
        first_k = k_ff;

        const Mat3 A_t = Transpose3(A);
        const Mat3 A_t_P_A = MatMul3(A_t, PA);
        const Mat3 A_t_P_B_K = Outer3(MatVec3(A_t, PB), K);
        P = AddMat3(Q, AddMat3(A_t_P_A, ScaleMat3(A_t_P_B_K, -1.0)));

        const Vec3 P_B_k = Scale3(PB, k_ff);
        const Vec3 s_term = Sub3(Pc_plus_s, P_B_k);
        s = MatVec3(A_t, s_term);
    }

    const Vec3 z0{cte_m, heading_err_rad, last_steer_rad};
    out.delta_u_rad = -(Dot3(first_K, z0) + first_k);
    out.steer_rad = last_steer_rad + out.delta_u_rad;
    out.valid = std::isfinite(out.steer_rad);
    return out;
}

} // namespace internal
} // namespace lane_keeping
