#include "controller_e5_blnn_lc.h"
#include <math.h>
#include <string.h>

/* Centers matrix (4 nodes x 5 inputs) matching MATLAB */
static const float BLNN_CENTERS[4][5] = {
    {-1.0f, -0.5f,  0.0f,  0.5f,  1.0f},
    {-0.5f,  0.0f,  0.5f,  1.0f, -1.0f},
    { 0.0f,  0.5f,  1.0f, -1.0f, -0.5f},
    { 0.5f,  1.0f, -1.0f, -0.5f,  0.0f}
};

static inline bool is_valid_float(float x)
{
    return !isnan(x) && !isinf(x);
}

static float clamp_f(float val, float min_v, float max_v)
{
    if (isnan(val)) return 0.0f;
    if (val < min_v) return min_v;
    if (val > max_v) return max_v;
    return val;
}

void e5_controller_init_config(e5_controller_cfg_t *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(e5_controller_cfg_t));
    
    cfg->Ts = 0.001f;
    cfg->Le = 0.600f;
    cfg->Kf_L = 0.0061979167f;
    cfg->Kf_R = 0.0061979167f;
    cfg->Imax_L = 1800.0f;
    cfg->Imax_R = 1800.0f;
    cfg->slew_rate_max = 75.0f;
    
    cfg->Gamma[0] = 25.0f; cfg->Gamma[1] = 18.0f;
    cfg->K1[0] = 40.0f;    cfg->K1[1] = 25.0f;
    cfg->K2[0] = 10.0f;    cfg->K2[1] = 5.0f;
    cfg->phi[0] = 0.025f;  cfg->phi[1] = 0.012f;
    cfg->eps_v = 0.01f;
    cfg->eps_alpha = 0.01f;
    
    cfg->M_hat[0] = 13.1f;  cfg->M_hat[1] = 0.393f;
    cfg->B_hat[0] = 70.0f;  cfg->B_hat[1] = 50.0f;
    cfg->K_hat[0] = 0.0f;   cfg->K_hat[1] = 2000.0f;
    cfg->A_hat[0] = 16.0f;  cfg->A_hat[1] = 0.0f;
    
    cfg->K_aw = 20.0f;
    cfg->lambda_aw = 20.0f;
    
    cfg->enable_blnn = true;
    cfg->enable_lc = true;
    cfg->gamma_lc = 100.0f;
    cfg->nn_eps[0] = 25.0f; cfg->nn_eps[1] = 18.0f;
    cfg->nn_eta[0] = 0.01f; cfg->nn_eta[1] = 0.01f;
    
    /* Input scales matching MATLAB ctrl.nn.input_scales */
    cfg->scale_y[0] = 2.0f;  cfg->scale_alpha[0] = 1.0f;
    cfg->scale_y[1] = 0.7f;  cfg->scale_alpha[1] = 0.2f;
    cfg->scale_y[2] = 1.0f;  cfg->scale_alpha[2] = 0.02f;
    cfg->scale_y[3] = 0.08f; cfg->scale_alpha[3] = 0.01f;
    cfg->scale_y[4] = 0.7f;  cfg->scale_alpha[4] = 0.2f;
}

void e5_controller_reset_state(e5_controller_state_t *state)
{
    if (!state) return;
    memset(state, 0, sizeof(e5_controller_state_t));
}

/**
 * @brief Forward mapping: motor currents to generalized forces
 * FG = Kf_L * iL - Kf_R * iR
 * Talpha = -0.5 * Le * (Kf_L * iL + Kf_R * iR)
 */
static void map_motor_to_modal(
    float iL, float iR, float Le, float Kf_L, float Kf_R,
    float *FG, float *Talpha)
{
    float FL = Kf_L * iL;
    float FR = -Kf_R * iR;
    *FG = FL + FR;
    *Talpha = 0.5f * Le * (FR - FL);
}

/**
 * @brief Inverse mapping: generalized forces to ideal motor currents
 */
static void map_modal_to_motor(
    float FG, float Talpha, float Le, float Kf_L, float Kf_R,
    float *iL, float *iR)
{
    /* FL = 0.5 * FG - (Talpha / Le) = Kf_L * iL */
    /* FR = 0.5 * FG + (Talpha / Le) = -Kf_R * iR */
    float FL = 0.5f * FG - (Talpha / Le);
    float FR = 0.5f * FG + (Talpha / Le);
    *iL = FL / Kf_L;
    *iR = -FR / Kf_R;
}

/**
 * @brief Closed-form thrust-space synchronization-priority allocator
 */
static void thrust_allocator_sync_c(
    float FG_des, float Talpha_des, float Le, float Kf_L, float Kf_R,
    float Imax_L, float Imax_R,
    float *iL_act, float *iR_act,
    float *iL_raw, float *iR_raw,
    bool *is_sat_thrust, bool *is_sat_torque)
{
    float FL_max = Kf_L * Imax_L;
    float FR_max = Kf_R * Imax_R;
    float DeltaF_des = (2.0f * Talpha_des) / Le;
    float DeltaF_max = FR_max + FL_max;
    
    /* Priority 1: Torqure correction delta thrust */
    float DeltaF_alloc = clamp_f(DeltaF_des, -DeltaF_max, DeltaF_max);
    
    /* Priority 2: Feasible bounds for FL */
    float FL_lower = fmaxf(-FL_max, -FR_max - DeltaF_alloc);
    float FL_upper = fminf( FL_max,  FR_max - DeltaF_alloc);
    
    float FL_target = 0.5f * FG_des - 0.5f * DeltaF_alloc;
    float FL_alloc = clamp_f(FL_target, FL_lower, FL_upper);
    float FR_alloc = FL_alloc + DeltaF_alloc;
    
    *iL_raw = FL_alloc / Kf_L;
    *iR_raw = -FR_alloc / Kf_R;
    
    *iL_act = clamp_f(*iL_raw, -Imax_L, Imax_L);
    *iR_act = clamp_f(*iR_raw, -Imax_R, Imax_R);
    
    float FL_real = Kf_L * (*iL_act);
    float FR_real = -Kf_R * (*iR_act);
    float FG_alloc = FL_real + FR_real;
    float Talpha_alloc = 0.5f * Le * (FR_real - FL_real);
    
    *is_sat_torque = (fabsf(Talpha_alloc - Talpha_des) > 1e-5f);
    *is_sat_thrust = (fabsf(FG_alloc - FG_des) > 1e-5f);
}

void e5_controller_step(
    const float q_meas[2],
    const float qdot_meas[2],
    const float qddot_est[2],
    const float qd[2],
    const float qdot_d[2],
    const float qddot_d[2],
    const e5_controller_cfg_t *cfg,
    e5_controller_state_t *state,
    e5_controller_info_t *info)
{
    if (!info) return;
    info->numerical_fault = false;

    /* 0. Input and internal state NaN / Inf Sanity Check */
    if (!is_valid_float(q_meas[0]) || !is_valid_float(q_meas[1]) ||
        !is_valid_float(qdot_meas[0]) || !is_valid_float(qdot_meas[1]) ||
        !is_valid_float(qddot_est[0]) || !is_valid_float(qddot_est[1]) ||
        !is_valid_float(qd[0]) || !is_valid_float(qd[1]) ||
        !is_valid_float(qdot_d[0]) || !is_valid_float(qdot_d[1]) ||
        !is_valid_float(qddot_d[0]) || !is_valid_float(qddot_d[1]) ||
        !is_valid_float(state->z_aw[0]) || !is_valid_float(state->z_aw[1]) ||
        !is_valid_float(state->v_prev[0]) || !is_valid_float(state->v_prev[1]))
    {
        info->numerical_fault = true;
        info->iL_cmd = 0.0f;
        info->iR_cmd = 0.0f;
        return;
    }

    /* 1. Tracking errors and generalized sliding surfaces */
    float e[2], edot[2], sigma[2];
    e[0] = q_meas[0] - qd[0];
    e[1] = q_meas[1] - qd[1];
    edot[0] = qdot_meas[0] - qdot_d[0];
    edot[1] = qdot_meas[1] - qdot_d[1];
    sigma[0] = edot[0] + cfg->Gamma[0] * e[0];
    sigma[1] = edot[1] + cfg->Gamma[1] * e[1];
    
    state->step_count++;
    
    /* 2. Baseline C2b nominal dynamics and robust feedback */
    float Sf[2];
    Sf[0] = tanhf(qdot_meas[0] / cfg->eps_v);
    Sf[1] = tanhf(qdot_meas[1] / cfg->eps_alpha);
    
    float va[2];
    va[0] = cfg->B_hat[0] * qdot_meas[0] + cfg->K_hat[0] * q_meas[0] + cfg->A_hat[0] * Sf[0] + cfg->M_hat[0] * qddot_d[0];
    va[1] = cfg->B_hat[1] * qdot_meas[1] + cfg->K_hat[1] * q_meas[1] + cfg->A_hat[1] * Sf[1] + cfg->M_hat[1] * qddot_d[1];
    
    float vr[2];
    vr[0] = -cfg->K1[0] * sigma[0] - cfg->K2[0] * tanhf(sigma[0] / cfg->phi[0]) - cfg->M_hat[0] * cfg->Gamma[0] * edot[0];
    vr[1] = -cfg->K1[1] * sigma[1] - cfg->K2[1] * tanhf(sigma[1] / cfg->phi[1]) - cfg->M_hat[1] * cfg->Gamma[1] * edot[1];
    
    float v_base[2];
    v_base[0] = va[0] + vr[0];
    v_base[1] = va[1] + vr[1];
    
    /* 3. Base actuator footprint with previous anti-windup state (motor current space) */
    float v_aw_modal[2];
    map_motor_to_modal(state->z_aw[0], state->z_aw[1], cfg->Le, cfg->Kf_L, cfg->Kf_R,
                       &v_aw_modal[0], &v_aw_modal[1]);
    float v_base_total[2];
    v_base_total[0] = v_base[0] + v_aw_modal[0];
    v_base_total[1] = v_base[1] + v_aw_modal[1];
    
    float i_base_act_L, i_base_act_R, i_base_raw_L, i_base_raw_R;
    bool dummy_sat_t, dummy_sat_q;
    thrust_allocator_sync_c(
        v_base_total[0], v_base_total[1], cfg->Le, cfg->Kf_L, cfg->Kf_R,
        cfg->Imax_L, cfg->Imax_R,
        &i_base_act_L, &i_base_act_R, &i_base_raw_L, &i_base_raw_R,
        &dummy_sat_t, &dummy_sat_q);
    
    /* 4. BLNN Neural network feature mapping (4 nodes x 3 features) */
    float v_nn_raw_modal[2] = {0.0f, 0.0f};
    float S_y[4][3], S_a[4][3];
    
    if (cfg->enable_blnn) {
        /* Normalized inputs */
        float chi_y[5] = {
            clamp_f(qddot_est[0] / cfg->scale_y[0], -2.0f, 2.0f),
            clamp_f(qdot_meas[0] / cfg->scale_y[1], -2.0f, 2.0f),
            clamp_f(q_meas[0]    / cfg->scale_y[2], -2.0f, 2.0f),
            clamp_f(e[0]         / cfg->scale_y[3], -2.0f, 2.0f),
            clamp_f(edot[0]      / cfg->scale_y[4], -2.0f, 2.0f)
        };
        float chi_a[5] = {
            clamp_f(qddot_est[1] / cfg->scale_alpha[0], -2.0f, 2.0f),
            clamp_f(qdot_meas[1] / cfg->scale_alpha[1], -2.0f, 2.0f),
            clamp_f(q_meas[1]    / cfg->scale_alpha[2], -2.0f, 2.0f),
            clamp_f(e[1]         / cfg->scale_alpha[3], -2.0f, 2.0f),
            clamp_f(edot[1]      / cfg->scale_alpha[4], -2.0f, 2.0f)
        };
        
        for (int j = 0; j < 4; j++) {
            float d2_y = 0.0f, d2_a = 0.0f;
            for (int k = 0; k < 5; k++) {
                float diff_y = BLNN_CENTERS[j][k] - chi_y[k];
                d2_y += diff_y * diff_y;
                float diff_a = BLNN_CENTERS[j][k] - chi_a[k];
                d2_a += diff_a * diff_a;
            }
            float h_y = expf(-d2_y);
            S_y[j][0] = h_y;
            S_y[j][1] = cosf(h_y);
            S_y[j][2] = sinf(h_y);
            
            float h_a = expf(-d2_a);
            S_a[j][0] = h_a;
            S_a[j][1] = cosf(h_a);
            S_a[j][2] = sinf(h_a);
            
            for (int m = 0; m < 3; m++) {
                v_nn_raw_modal[0] -= state->W_y[j][m] * S_y[j][m];
                v_nn_raw_modal[1] -= state->W_alpha[j][m] * S_a[j][m];
            }
        }
    }
    
    /* Map raw neural request to motor space */
    float v_nn_raw_motor[2];
    map_modal_to_motor(v_nn_raw_modal[0], v_nn_raw_modal[1], cfg->Le, cfg->Kf_L, cfg->Kf_R,
                       &v_nn_raw_motor[0], &v_nn_raw_motor[1]);
    
    /* 5. Slew rate limiter on motor space */
    float v_slew_motor[2];
    if (cfg->slew_rate_max > 0.0f) {
        float dv_L = v_nn_raw_motor[0] - state->v_prev[0];
        float dv_R = v_nn_raw_motor[1] - state->v_prev[1];
        v_slew_motor[0] = state->v_prev[0] + clamp_f(dv_L, -cfg->slew_rate_max, cfg->slew_rate_max);
        v_slew_motor[1] = state->v_prev[1] + clamp_f(dv_R, -cfg->slew_rate_max, cfg->slew_rate_max);
    } else {
        v_slew_motor[0] = v_nn_raw_motor[0];
        v_slew_motor[1] = v_nn_raw_motor[1];
    }
    state->v_prev[0] = v_slew_motor[0];
    state->v_prev[1] = v_slew_motor[1];
    
    /* 6. Form total command and allocate */
    float v_applied_modal[2];
    map_motor_to_modal(v_slew_motor[0], v_slew_motor[1], cfg->Le, cfg->Kf_L, cfg->Kf_R,
                       &v_applied_modal[0], &v_applied_modal[1]);
    
    float FG_cmd = v_base_total[0] + v_applied_modal[0];
    float Talpha_cmd = v_base_total[1] + v_applied_modal[1];
    
    float i_ideal_req_L, i_ideal_req_R;
    map_modal_to_motor(FG_cmd, Talpha_cmd, cfg->Le, cfg->Kf_L, cfg->Kf_R,
                       &i_ideal_req_L, &i_ideal_req_R);
    
    float iL_actual, iR_actual, iL_alloc_raw, iR_alloc_raw;
    bool is_sat_thrust = false, is_sat_torque = false;
    thrust_allocator_sync_c(
        FG_cmd, Talpha_cmd, cfg->Le, cfg->Kf_L, cfg->Kf_R,
        cfg->Imax_L, cfg->Imax_R,
        &iL_actual, &iR_actual, &iL_alloc_raw, &iR_alloc_raw,
        &is_sat_thrust, &is_sat_torque);
    
    /* 7. Increments and discrepancies */
    float v_applied_motor[2] = {
        iL_actual - i_base_act_L,
        iR_actual - i_base_act_R
    };
    float delta_v_total[2] = {
        v_applied_motor[0] - v_nn_raw_motor[0],
        v_applied_motor[1] - v_nn_raw_motor[1]
    };
    float delta_v_total_modal[2];
    map_motor_to_modal(delta_v_total[0], delta_v_total[1], cfg->Le, cfg->Kf_L, cfg->Kf_R,
                       &delta_v_total_modal[0], &delta_v_total_modal[1]);
    
    /* 8. Learning Coordination (LC) */
    float rho_y = 1.0f, rho_alpha = 1.0f;
    if (cfg->enable_blnn && cfg->enable_lc) {
        float max_Imax = fmaxf(cfg->Imax_L, cfg->Imax_R);
        float chi_y = (fabsf(delta_v_total_modal[0]) / (cfg->Kf_L + cfg->Kf_R)) / max_Imax;
        rho_y = 1.0f / (1.0f + cfg->gamma_lc * (chi_y * chi_y));
        
        float chi_alpha = (fabsf(delta_v_total_modal[1]) / (0.5f * cfg->Le * (cfg->Kf_L + cfg->Kf_R))) / max_Imax;
        rho_alpha = 1.0f / (1.0f + cfg->gamma_lc * (chi_alpha * chi_alpha));
    }
    
    /* Weight adaptive update */
    if (cfg->enable_blnn) {
        for (int j = 0; j < 4; j++) {
            for (int m = 0; m < 3; m++) {
                float W_dot_y = cfg->nn_eps[0] * (rho_y * sigma[0] * S_y[j][m] - cfg->nn_eta[0] * state->W_y[j][m]);
                float W_dot_a = cfg->nn_eps[1] * (rho_alpha * sigma[1] * S_a[j][m] - cfg->nn_eta[1] * state->W_alpha[j][m]);
                state->W_y[j][m] += cfg->Ts * W_dot_y;
                state->W_alpha[j][m] += cfg->Ts * W_dot_a;
            }
        }
    }
    
    /* 9. Dynamic anti-windup state update (in motor counts) */
    float Delta_i_alloc_L = iL_actual - i_ideal_req_L;
    float Delta_i_alloc_R = iR_actual - i_ideal_req_R;
    
    state->z_aw[0] += cfg->Ts * (cfg->K_aw * Delta_i_alloc_L - cfg->lambda_aw * state->z_aw[0]);
    state->z_aw[1] += cfg->Ts * (cfg->K_aw * Delta_i_alloc_R - cfg->lambda_aw * state->z_aw[1]);
    
    /* 10. Populate diagnostics info */
    info->iL_cmd = iL_actual;
    info->iR_cmd = iR_actual;
    info->e_y = e[0];
    info->e_sync = cfg->Le * q_meas[1];
    info->sigma[0] = sigma[0];
    info->sigma[1] = sigma[1];
    info->v_nn_raw_modal[0] = v_nn_raw_modal[0];
    info->v_nn_raw_modal[1] = v_nn_raw_modal[1];
    info->v_nn_raw_motor[0] = v_nn_raw_motor[0];
    info->v_nn_raw_motor[1] = v_nn_raw_motor[1];
    info->v_applied_motor[0] = v_applied_motor[0];
    info->v_applied_motor[1] = v_applied_motor[1];
    info->delta_v_total[0] = delta_v_total[0];
    info->delta_v_total[1] = delta_v_total[1];
    info->rho_lc[0] = rho_y;
    info->rho_lc[1] = rho_alpha;
    info->sat_alloc = is_sat_thrust || is_sat_torque;
    info->is_sat_thrust = is_sat_thrust;
    info->is_sat_torque = is_sat_torque;
    info->F_cmd = FG_cmd;
    info->T_cmd = Talpha_cmd;
    
    float norm_Wy_sq = 0.0f, norm_Wa_sq = 0.0f;
    for (int j = 0; j < 4; j++) {
        for (int m = 0; m < 3; m++) {
            norm_Wy_sq += state->W_y[j][m] * state->W_y[j][m];
            norm_Wa_sq += state->W_alpha[j][m] * state->W_alpha[j][m];
        }
    }
    info->norm_W_y = sqrtf(norm_Wy_sq);
    info->norm_W_alpha = sqrtf(norm_Wa_sq);

    /* 11. Output command and state NaN/Inf sanity check */
    if (!is_valid_float(info->iL_cmd) || !is_valid_float(info->iR_cmd) ||
        !is_valid_float(state->z_aw[0]) || !is_valid_float(state->z_aw[1]) ||
        !is_valid_float(info->norm_W_y) || !is_valid_float(info->norm_W_alpha))
    {
        info->numerical_fault = true;
        info->iL_cmd = 0.0f;
        info->iR_cmd = 0.0f;
    }
}
