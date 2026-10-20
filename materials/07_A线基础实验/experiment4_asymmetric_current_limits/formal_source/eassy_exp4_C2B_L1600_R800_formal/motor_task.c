#include "motor_task.h"

extern motor_measure_t motor_3508[2];

volatile uint8_t g_exp3_abort = 0U;
experiment3_state_t g_exp3;

#define EXP3_FAULT_CAN_LEFT       0x01U
#define EXP3_FAULT_CAN_RIGHT      0x02U
#define EXP3_FAULT_TEMPERATURE    0x04U
#define EXP3_FAULT_SYNCHRONISM    0x08U
#define EXP3_FAULT_TRAVEL         0x10U
#define EXP3_FAULT_USER_ABORT     0x20U

static float smoothstep5(float u)
{
    float u2;
    float u3;

    if (u <= 0.0f)
    {
        return 0.0f;
    }
    if (u >= 1.0f)
    {
        return 1.0f;
    }

    u2 = u * u;
    u3 = u2 * u;
    return u3 * (10.0f + u * (-15.0f + 6.0f * u));
}

static void trajectory_reference(uint32_t elapsed_ms,
                                 uint8_t *phase,
                                 float *position_m,
                                 float *velocity_mps,
                                 float *acceleration_mps2)
{
    uint32_t forward_end_ms;
    uint32_t hold_end_ms;
    uint32_t return_end_ms;
    float u;
    float duration_s;
    float s;
    float ds_du;
    float d2s_du2;

    *position_m = 0.0f;
    *velocity_mps = 0.0f;
    *acceleration_mps2 = 0.0f;

    if (elapsed_ms < EXP3_PRE_HOLD_MS)
    {
        *phase = EXP3_PHASE_PRE_HOLD;
        return;
    }

    forward_end_ms = EXP3_PRE_HOLD_MS + EXP3_FORWARD_MS;
    hold_end_ms = forward_end_ms + EXP3_HOLD_END_MS;
    return_end_ms = hold_end_ms + EXP3_RETURN_MS;

    if (elapsed_ms < forward_end_ms)
    {
        *phase = EXP3_PHASE_FORWARD;
        u = (float)(elapsed_ms - EXP3_PRE_HOLD_MS) / (float)EXP3_FORWARD_MS;
        duration_s = (float)EXP3_FORWARD_MS * 0.001f;
        s = smoothstep5(u);
        ds_du = 30.0f * u * u * (1.0f - u) * (1.0f - u);
        d2s_du2 = 60.0f * u * (1.0f - u) * (1.0f - 2.0f * u);
        *position_m = EXP3_STROKE_M * s;
        *velocity_mps = EXP3_STROKE_M * ds_du / duration_s;
        *acceleration_mps2 = EXP3_STROKE_M * d2s_du2 / (duration_s * duration_s);
        return;
    }

    if (elapsed_ms < hold_end_ms)
    {
        *phase = EXP3_PHASE_HOLD_END;
        *position_m = EXP3_STROKE_M;
        return;
    }

    if (elapsed_ms < return_end_ms)
    {
        *phase = EXP3_PHASE_RETURN;
        u = (float)(elapsed_ms - hold_end_ms) / (float)EXP3_RETURN_MS;
        duration_s = (float)EXP3_RETURN_MS * 0.001f;
        s = smoothstep5(u);
        ds_du = 30.0f * u * u * (1.0f - u) * (1.0f - u);
        d2s_du2 = 60.0f * u * (1.0f - u) * (1.0f - 2.0f * u);
        *position_m = EXP3_STROKE_M * (1.0f - s);
        *velocity_mps = -EXP3_STROKE_M * ds_du / duration_s;
        *acceleration_mps2 = -EXP3_STROKE_M * d2s_du2 / (duration_s * duration_s);
        return;
    }

    if (elapsed_ms < EXP3_TOTAL_MS)
    {
        *phase = EXP3_PHASE_RECOVERY_HOLD;
        return;
    }

    *phase = EXP3_PHASE_DONE;
}

static float clamp_float(float value, float limit)
{
    if (value > limit)
    {
        return limit;
    }
    if (value < -limit)
    {
        return -limit;
    }
    return value;
}

static void c2b_controller(float reference_m,
                           float reference_velocity_mps,
                           float reference_acceleration_mps2)
{
    float position_left_m;
    float position_right_m;
    float velocity_left_mps;
    float velocity_right_mps;
    float y_m;
    float alpha_rad;
    float y_velocity_mps;
    float alpha_velocity_radps;
    float error_y;
    float error_alpha;
    float error_velocity_y;
    float error_velocity_alpha;
    float sigma_y;
    float sigma_alpha;
    float va_y;
    float va_alpha;
    float vr_y;
    float vr_alpha;
    float force_y;
    float torque_alpha;
    float request_left;
    float request_right;
    float allocated_left;
    float allocated_right;
    float actual_left;
    float actual_right;
    float residual_left;
    float residual_right;
    float z_aw_left;
    float z_aw_right;

    position_left_m = ((float)motor_3508[0].total_ecd - chassis_ang_init[0]) *
                      EXP3_METERS_PER_MOTOR_COUNT;
    position_right_m = -((float)motor_3508[1].total_ecd - chassis_ang_init[1]) *
                       EXP3_METERS_PER_MOTOR_COUNT;
    velocity_left_mps = ((float)motor_3508[0].speed_rpm / 60.0f) *
                        (2.0f * EXP3_PI * EXP3_DRIVE_RADIUS_M / EXP3_GEAR_RATIO);
    velocity_right_mps = -((float)motor_3508[1].speed_rpm / 60.0f) *
                         (2.0f * EXP3_PI * EXP3_DRIVE_RADIUS_M / EXP3_GEAR_RATIO);

    y_m = 0.5f * (position_left_m + position_right_m);
    alpha_rad = (position_right_m - position_left_m) / EXP3_C2_LE_M;
    y_velocity_mps = 0.5f * (velocity_left_mps + velocity_right_mps);
    alpha_velocity_radps = (velocity_right_mps - velocity_left_mps) / EXP3_C2_LE_M;

    error_y = y_m - reference_m;
    error_alpha = alpha_rad;
    error_velocity_y = y_velocity_mps - reference_velocity_mps;
    error_velocity_alpha = alpha_velocity_radps;
    sigma_y = error_velocity_y + EXP3_C2_GAMMA_Y * error_y;
    sigma_alpha = error_velocity_alpha + EXP3_C2_GAMMA_ALPHA * error_alpha;

    va_y = EXP3_C2_B_Y * y_velocity_mps +
           EXP3_C2_A_Y * tanhf(y_velocity_mps / EXP3_C2_EPS_V) +
           EXP3_C2_M_Y * reference_acceleration_mps2;
    va_alpha = EXP3_C2_B_ALPHA * alpha_velocity_radps +
               EXP3_C2_K_ALPHA * alpha_rad;

    vr_y = -EXP3_C2_K1_Y * sigma_y -
           EXP3_C2_K2_Y * tanhf(sigma_y / EXP3_C2_PHI_Y) -
           EXP3_C2_M_Y * EXP3_C2_GAMMA_Y * error_velocity_y;
    vr_alpha = -EXP3_C2_K1_ALPHA * sigma_alpha -
               EXP3_C2_K2_ALPHA * tanhf(sigma_alpha / EXP3_C2_PHI_ALPHA) -
               EXP3_C2_M_ALPHA * EXP3_C2_GAMMA_ALPHA * error_velocity_alpha;

    /* C2b uses the previous-cycle dynamic state in generalized-force space.
     * T_act maps motor-current counts to [gantry force; skew torque].
     */
    z_aw_left = g_exp3.z_aw_left;
    z_aw_right = g_exp3.z_aw_right;
    force_y = va_y + vr_y +
              EXP3_C2_KF_LEFT_N_PER_COUNT * z_aw_left -
              EXP3_C2_KF_RIGHT_N_PER_COUNT * z_aw_right;
    torque_alpha = va_alpha + vr_alpha -
                   0.5f * EXP3_C2_LE_M *
                   (EXP3_C2_KF_LEFT_N_PER_COUNT * z_aw_left +
                    EXP3_C2_KF_RIGHT_N_PER_COUNT * z_aw_right);
    request_left = force_y / (2.0f * EXP3_C2_KF_LEFT_N_PER_COUNT) -
                   torque_alpha / (EXP3_C2_LE_M * EXP3_C2_KF_LEFT_N_PER_COUNT);
    request_right = -(force_y / (2.0f * EXP3_C2_KF_RIGHT_N_PER_COUNT) +
                      torque_alpha / (EXP3_C2_LE_M * EXP3_C2_KF_RIGHT_N_PER_COUNT));

    allocated_left = clamp_float(request_left, EXP3_REQUEST_LIMIT);
    allocated_right = clamp_float(request_right, EXP3_REQUEST_LIMIT);
    actual_left = clamp_float(allocated_left, EXP4_CURRENT_LIMIT_LEFT);
    actual_right = clamp_float(allocated_right, EXP4_CURRENT_LIMIT_RIGHT);
    residual_left = actual_left - allocated_left;
    residual_right = actual_right - allocated_right;

    g_exp3.request_left = request_left;
    g_exp3.request_right = request_right;
    g_exp3.allocated_left = allocated_left;
    g_exp3.allocated_right = allocated_right;
    g_exp3.applied_left = (int16_t)actual_left;
    g_exp3.applied_right = (int16_t)actual_right;
    g_exp3.sat_left = (fabsf(residual_left) > 0.5f) ? 1U : 0U;
    g_exp3.sat_right = (fabsf(residual_right) > 0.5f) ? 1U : 0U;
    g_exp3.sigma_y = sigma_y;
    g_exp3.sigma_alpha = sigma_alpha;
    g_exp3.z_aw_left = z_aw_left + EXP3_CONTROL_TS_S *
                       (EXP3_C2_K_AW * residual_left -
                        EXP3_C2_LAMBDA_AW * z_aw_left);
    g_exp3.z_aw_right = z_aw_right + EXP3_CONTROL_TS_S *
                        (EXP3_C2_K_AW * residual_right -
                         EXP3_C2_LAMBDA_AW * z_aw_right);
    g_exp3.delta_alloc_left = allocated_left - request_left;
    g_exp3.delta_alloc_right = allocated_right - request_right;
    g_exp3.delta_total_left = actual_left - request_left;
    g_exp3.delta_total_right = actual_right - request_right;
}

static uint32_t can_age_ms(uint8_t index, uint32_t now_ms)
{
    if (motor_3508[index].rx_count == 0U)
    {
        return 0xFFFFFFFFU;
    }
    return now_ms - motor_3508[index].last_rx_ms;
}

static uint8_t protection_check(uint32_t now_ms)
{
    uint8_t fault = 0U;
    float pos_left_m;
    float pos_right_m;

    if (can_age_ms(0U, now_ms) > EXP3_CAN_TIMEOUT_MS)
    {
        fault |= EXP3_FAULT_CAN_LEFT;
    }
    if (can_age_ms(1U, now_ms) > EXP3_CAN_TIMEOUT_MS)
    {
        fault |= EXP3_FAULT_CAN_RIGHT;
    }
    if ((motor_3508[0].temperate >= EXP3_TEMPERATURE_LIMIT_C) ||
        (motor_3508[1].temperate >= EXP3_TEMPERATURE_LIMIT_C))
    {
        fault |= EXP3_FAULT_TEMPERATURE;
    }

    pos_left_m = ((float)motor_3508[0].total_ecd - chassis_ang_init[0]) *
                 EXP3_METERS_PER_MOTOR_COUNT;
    pos_right_m = -((float)motor_3508[1].total_ecd - chassis_ang_init[1]) *
                  EXP3_METERS_PER_MOTOR_COUNT;

    if (fabsf(pos_left_m - pos_right_m) > EXP3_SYNC_FAULT_M)
    {
        fault |= EXP3_FAULT_SYNCHRONISM;
    }
    if ((fabsf(pos_left_m) > EXP3_TRAVEL_FAULT_M) ||
        (fabsf(pos_right_m) > EXP3_TRAVEL_FAULT_M))
    {
        fault |= EXP3_FAULT_TRAVEL;
    }
    if (g_exp3_abort != 0U)
    {
        fault |= EXP3_FAULT_USER_ABORT;
    }

    return fault;
}

static void update_tracking_gate(float reference_m)
{
    float pos_left_m;
    float pos_right_m;
    float mean_m;

    pos_left_m = ((float)motor_3508[0].total_ecd - chassis_ang_init[0]) *
                 EXP3_METERS_PER_MOTOR_COUNT;
    pos_right_m = -((float)motor_3508[1].total_ecd - chassis_ang_init[1]) *
                  EXP3_METERS_PER_MOTOR_COUNT;
    mean_m = 0.5f * (pos_left_m + pos_right_m);

    g_exp3.tracking_in_band =
        ((fabsf(reference_m - mean_m) <= EXP3_TRACK_BAND_M) &&
         (fabsf(pos_left_m - pos_right_m) <= EXP3_SYNC_BAND_M)) ? 1U : 0U;
}

static void command_zero(void)
{
    g_exp3.request_left = 0.0f;
    g_exp3.request_right = 0.0f;
    g_exp3.allocated_left = 0.0f;
    g_exp3.allocated_right = 0.0f;
    g_exp3.applied_left = 0;
    g_exp3.applied_right = 0;
    g_exp3.sat_left = 0U;
    g_exp3.sat_right = 0U;
    g_exp3.sigma_y = 0.0f;
    g_exp3.sigma_alpha = 0.0f;
    g_exp3.z_aw_left = 0.0f;
    g_exp3.z_aw_right = 0.0f;
    g_exp3.delta_alloc_left = 0.0f;
    g_exp3.delta_alloc_right = 0.0f;
    g_exp3.delta_total_left = 0.0f;
    g_exp3.delta_total_right = 0.0f;
    CAN_cmd_3508(0, 0);
}

static void enter_fault_stop(void)
{
    g_exp3.running = 0U;
    g_exp3.finished = 1U;
    g_exp3.phase = EXP3_PHASE_FAULT;
    motor_stop_and_clear();
    command_zero();
}

void motor_task(void const *argument)
{
    uint32_t arm_start_ms;
    uint32_t feedback_wait_start_ms;
    uint32_t experiment_start_ms;
    uint32_t now_ms;
    uint8_t phase;
    uint8_t fault;
    float reference_m;
    float reference_velocity_mps;
    float reference_acceleration_mps2;

    (void)argument;
    g_exp3.phase = EXP3_PHASE_ARMING;
    command_zero();
    osDelay(1500U);

    feedback_wait_start_ms = HAL_GetTick();
    while ((motor_3508[0].rx_count == 0U) || (motor_3508[1].rx_count == 0U))
    {
        if ((HAL_GetTick() - feedback_wait_start_ms) >= 3000U)
        {
            if (motor_3508[0].rx_count == 0U)
            {
                g_exp3.fault_flags |= EXP3_FAULT_CAN_LEFT;
            }
            if (motor_3508[1].rx_count == 0U)
            {
                g_exp3.fault_flags |= EXP3_FAULT_CAN_RIGHT;
            }
            enter_fault_stop();
            for (;;)
            {
                command_zero();
                osDelay(1U);
            }
        }
        command_zero();
        osDelay(1U);
    }

    motor_init();
    arm_start_ms = HAL_GetTick();

    while ((HAL_GetTick() - arm_start_ms) < EXP3_ARM_DELAY_MS)
    {
        now_ms = HAL_GetTick();
        fault = protection_check(now_ms);
        if (fault != 0U)
        {
            g_exp3.fault_flags |= fault;
            enter_fault_stop();
            for (;;)
            {
                command_zero();
                osDelay(1U);
            }
        }
        command_zero();
        osDelay(1U);
    }

#if EXP3_AUTORUN_ENABLE
    experiment_start_ms = HAL_GetTick();
    g_exp3.running = 1U;

    for (;;)
    {
        now_ms = HAL_GetTick();
        g_exp3.elapsed_ms = now_ms - experiment_start_ms;
        trajectory_reference(g_exp3.elapsed_ms,
                             &phase,
                             &reference_m,
                             &reference_velocity_mps,
                             &reference_acceleration_mps2);
        g_exp3.reference_m = reference_m;
        g_exp3.reference_velocity_mps = reference_velocity_mps;
        g_exp3.reference_acceleration_mps2 = reference_acceleration_mps2;
        g_exp3.phase = phase;

        fault = protection_check(now_ms);
        g_exp3.fault_flags |= fault;

        if (g_exp3.fault_flags != 0U)
        {
            enter_fault_stop();
        }
        else if (g_exp3.elapsed_ms >= EXP3_TOTAL_MS)
        {
            g_exp3.running = 0U;
            g_exp3.finished = 1U;
            g_exp3.phase = EXP3_PHASE_DONE;
            motor_stop_and_clear();
            command_zero();
            update_tracking_gate(0.0f);
        }
        else
        {
            c2b_controller(reference_m,
                           reference_velocity_mps,
                           reference_acceleration_mps2);
            if ((g_exp3.sat_left != 0U) || (g_exp3.sat_right != 0U))
            {
                g_exp3.saturation_latched = 1U;
            }
            update_tracking_gate(reference_m);
            CAN_cmd_3508(g_exp3.applied_left, g_exp3.applied_right);
        }

        osDelay(1U);
    }
#else
    for (;;)
    {
        command_zero();
        osDelay(1U);
    }
#endif
}
