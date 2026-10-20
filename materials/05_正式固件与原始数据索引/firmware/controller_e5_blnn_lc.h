#ifndef CONTROLLER_E5_BLNN_LC_H
#define CONTROLLER_E5_BLNN_LC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Controller configuration parameters for E5 Coordinated BLNN
 */
typedef struct {
    float Ts;                 /* Control sampling period (s), default 0.001f */
    float Le;                 /* Gantry track span (m), default 0.600f */
    float Kf_L;               /* Thrust coefficient left (N/count), default 0.006198f */
    float Kf_R;               /* Thrust coefficient right (N/count), default 0.006198f */
    float Imax_L;             /* Current limit left (counts) */
    float Imax_R;             /* Current limit right (counts) */
    float slew_rate_max;      /* Max motor-space current change per step (counts/step), e.g. 75.0f */
    
    /* C2b Sliding Mode and Nominal Dynamics Gains */
    float Gamma[2];           /* [gamma_y, gamma_alpha], default [25.0, 18.0] */
    float K1[2];              /* [K1_y, K1_alpha], default [40.0, 25.0] */
    float K2[2];              /* [K2_y, K2_alpha], default [10.0, 5.0] */
    float phi[2];             /* Boundary layer thickness [phi_y, phi_alpha], default [0.025, 0.012] */
    float eps_v;              /* Velocity smoothing eps for Sf_y, default 0.01f */
    float eps_alpha;          /* Angular velocity smoothing eps for Sf_alpha, default 0.01f */
    
    float M_hat[2];           /* Nominal mass & inertia [mG_nom, J_alpha_nom], default [13.1, 0.393] */
    float B_hat[2];           /* Nominal damping [2*b_nom, 50.0], default [70.0, 50.0] */
    float K_hat[2];           /* Nominal stiffness [0.0, K_alpha], default [0.0, 2000.0] */
    float A_hat[2];           /* Nominal Coulomb friction [2*fc_nom, 0.0], default [16.0, 0.0] */
    
    /* Anti-windup parameters */
    float K_aw;               /* Anti-windup gain, default 20.0f */
    float lambda_aw;          /* Anti-windup decay rate, default 20.0f */
    
    /* BLNN and LC parameters */
    bool enable_blnn;         /* Master switch for neural network */
    bool enable_lc;           /* Switch for Learning Coordination (LC) */
    float gamma_lc;           /* LC modulation steepness parameter, default 100.0f */
    float nn_eps[2];          /* Learning rates [eps_y, eps_alpha], default [25.0, 18.0] */
    float nn_eta[2];          /* Leakage coefficients [eta_y, eta_alpha], default [0.01, 0.01] */
    float scale_y[5];         /* Normalization scales for y: [qddot, qdot, q, e, edot] */
    float scale_alpha[5];     /* Normalization scales for alpha: [qddot, qdot, q, e, edot] */
} e5_controller_cfg_t;

/**
 * @brief Controller persistent state structure
 */
typedef struct {
    float z_aw[2];            /* Dynamic anti-windup state: [z_aw_L, z_aw_R] in motor current counts */
    float v_prev[2];          /* Previous cycle motor-space neural compensation [v_prev_L, v_prev_R] */
    float W_y[4][3];          /* Translation neural network weights (4 nodes x 3 features) */
    float W_alpha[4][3];      /* Rotation neural network weights (4 nodes x 3 features) */
    uint32_t step_count;      /* Step counter */
} e5_controller_state_t;

/**
 * @brief Controller step diagnostics information
 */
typedef struct {
    float iL_cmd;             /* Output current command to Left Motor (counts) */
    float iR_cmd;             /* Output current command to Right Motor (counts) */
    float e_y;                /* Position tracking error yG - yd (m) */
    float e_sync;             /* Synchronization error Le * alpha (m) */
    float sigma[2];           /* Sliding surface values [sigma_y, sigma_alpha] */
    float v_nn_raw_modal[2];  /* Raw neural request [FG_raw, Talpha_raw] */
    float v_nn_raw_motor[2];  /* Raw neural request in motor space [v_raw_L, v_raw_R] */
    float v_applied_motor[2]; /* Realized neural increment in motor space */
    float delta_v_total[2];   /* Total discrepancy: v_applied - v_raw (motor space) */
    float rho_lc[2];          /* LC modulation factors [rho_y, rho_alpha] */
    float norm_W_y;           /* Frobenius norm of translation weights */
    float norm_W_alpha;       /* Frobenius norm of rotation weights */
    float F_cmd;              /* Pre-allocation translation force command (N) */
    float T_cmd;              /* Pre-allocation rotation torque command (N*m) */
    bool sat_alloc;           /* Allocator saturation flag */
    bool is_sat_thrust;       /* Thrust channel saturation flag */
    bool is_sat_torque;       /* Torque channel saturation flag */
    bool numerical_fault;     /* NaN or Inf detected in inputs, states, or outputs */
} e5_controller_info_t;

/**
 * @brief Initialize controller configuration with default nominal values
 */
void e5_controller_init_config(e5_controller_cfg_t *cfg);

/**
 * @brief Reset controller state
 */
void e5_controller_reset_state(e5_controller_state_t *state);

/**
 * @brief Main controller execution step (1.0 ms cycle)
 *
 * @param q_meas     Measured modal position [yG, alpha] (m, rad)
 * @param qdot_meas  Measured modal velocity [yG_dot, alpha_dot] (m/s, rad/s)
 * @param qddot_est  Filtered modal acceleration estimate [yG_ddot, alpha_ddot] (m/s^2, rad/s^2)
 * @param qd         Desired modal position [yd, 0.0] (m, rad)
 * @param qdot_d     Desired modal velocity [yd_dot, 0.0] (m/s, rad/s)
 * @param qddot_d    Desired modal acceleration [yd_ddot, 0.0] (m/s^2, rad/s^2)
 * @param cfg        Controller configuration pointer
 * @param state      Controller state pointer (updated in-place)
 * @param info       Diagnostics info output pointer
 */
void e5_controller_step(
    const float q_meas[2],
    const float qdot_meas[2],
    const float qddot_est[2],
    const float qd[2],
    const float qdot_d[2],
    const float qddot_d[2],
    const e5_controller_cfg_t *cfg,
    e5_controller_state_t *state,
    e5_controller_info_t *info);

#ifdef __cplusplus
}
#endif

#endif /* CONTROLLER_E5_BLNN_LC_H */
