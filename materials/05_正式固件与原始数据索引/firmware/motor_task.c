#include "motor_task.h"
#include <math.h>
#include <string.h>

extern motor_measure_t motor_3508[2];

volatile uint8_t g_exp5_abort = 0U;
experiment5_state_t g_exp5;
static bool s_summary_sent = false;

static int16_t current_command(float request, float limit);

static inline bool is_finite_val(float x)
{
    return !isnan(x) && !isinf(x);
}

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

    if (elapsed_ms < EXP5_PRE_HOLD_MS)
    {
        *phase = EXP5_PHASE_PRE_HOLD;
        return;
    }

    forward_end_ms = EXP5_PRE_HOLD_MS + EXP5_FORWARD_MS;
    hold_end_ms = forward_end_ms + EXP5_HOLD_END_MS;
    return_end_ms = hold_end_ms + EXP5_RETURN_MS;

    if (elapsed_ms < forward_end_ms)
    {
        *phase = EXP5_PHASE_FORWARD;
        u = (float)(elapsed_ms - EXP5_PRE_HOLD_MS) / (float)EXP5_FORWARD_MS;
        duration_s = (float)EXP5_FORWARD_MS * 0.001f;
        s = smoothstep5(u);
        ds_du = 30.0f * u * u * (1.0f - u) * (1.0f - u);
        d2s_du2 = 60.0f * u * (1.0f - u) * (1.0f - 2.0f * u);
        *position_m = EXP5_STROKE_M * s;
        *velocity_mps = EXP5_STROKE_M * ds_du / duration_s;
        *acceleration_mps2 = EXP5_STROKE_M * d2s_du2 / (duration_s * duration_s);
        return;
    }

    if (elapsed_ms < hold_end_ms)
    {
        *phase = EXP5_PHASE_HOLD_END;
        *position_m = EXP5_STROKE_M;
        return;
    }

    if (elapsed_ms < return_end_ms)
    {
        *phase = EXP5_PHASE_RETURN;
        u = (float)(elapsed_ms - hold_end_ms) / (float)EXP5_RETURN_MS;
        duration_s = (float)EXP5_RETURN_MS * 0.001f;
        s = smoothstep5(u);
        ds_du = 30.0f * u * u * (1.0f - u) * (1.0f - u);
        d2s_du2 = 60.0f * u * (1.0f - u) * (1.0f - 2.0f * u);
        *position_m = EXP5_STROKE_M * (1.0f - s);
        *velocity_mps = -EXP5_STROKE_M * ds_du / duration_s;
        *acceleration_mps2 = -EXP5_STROKE_M * d2s_du2 / (duration_s * duration_s);
        return;
    }

    if (elapsed_ms < EXP5_TOTAL_MS)
    {
        *phase = EXP5_PHASE_RECOVERY_HOLD;
        return;
    }

    *phase = EXP5_PHASE_DONE;
}

static int16_t current_command(float request, float limit)
{
    /* Robust NaN and Inf Interception */
    if (!is_finite_val(request))
    {
        g_exp5.numerical_fault_latched = 1U;
        g_exp5.fault_flags |= EXP5_FAULT_NUMERICAL_NAN;
        return 0;
    }
    if (request > limit)
    {
        return (int16_t)limit;
    }
    if (request < -limit)
    {
        return (int16_t)(-limit);
    }
    return (int16_t)request;
}

#if (EXP5_WORK_MODE == EXP5_MODE_LIVE_CAN)
static uint32_t can_age_ms(uint8_t index, uint32_t now_ms)
{
    if (motor_3508[index].rx_count == 0U)
    {
        return 0xFFFFFFFFU;
    }
    return now_ms - motor_3508[index].last_rx_ms;
}
#endif

static uint8_t protection_check(uint32_t now_ms)
{
    uint8_t fault = 0U;

    /* Numerical Sanity Check: only latched or core-detected numerical faults */
    if (g_exp5.info.numerical_fault || g_exp5.numerical_fault_latched)
    {
        fault |= EXP5_FAULT_NUMERICAL_NAN;
    }

#if (EXP5_WORK_MODE == EXP5_MODE_LIVE_CAN)
    uint32_t age_left = can_age_ms(0U, now_ms) + g_exp5.inject_can_age_ms;
    uint32_t age_right = can_age_ms(1U, now_ms);
    if (age_left > EXP5_CAN_TIMEOUT_MS)
    {
        fault |= EXP5_FAULT_CAN_LEFT;
    }
    if (age_right > EXP5_CAN_TIMEOUT_MS)
    {
        fault |= EXP5_FAULT_CAN_RIGHT;
    }
    if ((motor_3508[0].temperate >= EXP5_TEMPERATURE_LIMIT_C) ||
        (motor_3508[1].temperate >= EXP5_TEMPERATURE_LIMIT_C))
    {
        fault |= EXP5_FAULT_TEMPERATURE;
    }

    float pos_left_m = ((float)motor_3508[0].total_ecd - chassis_ang_init[0]) *
                       EXP5_METERS_PER_MOTOR_COUNT;
    float pos_right_m = -((float)motor_3508[1].total_ecd - chassis_ang_init[1]) *
                        EXP5_METERS_PER_MOTOR_COUNT;

    float sync_err_m = fabsf(pos_right_m - pos_left_m) + g_exp5.inject_sync_err_m;
    float travel_m = fabsf(0.5f * (pos_left_m + pos_right_m));

    if (sync_err_m > EXP5_SYNC_FAULT_M)
    {
        fault |= EXP5_FAULT_SYNCHRONISM;
    }
    if (travel_m > EXP5_TRAVEL_FAULT_M)
    {
        fault |= EXP5_FAULT_TRAVEL;
    }

    if ((fabsf((float)motor_3508[0].speed_rpm) > EXP5_SPEED_LIMIT_RPM) ||
        (fabsf((float)motor_3508[1].speed_rpm) > EXP5_SPEED_LIMIT_RPM))
    {
        fault |= EXP5_FAULT_SPEED;
    }
#endif

    if (g_exp5_abort != 0U)
    {
        fault |= EXP5_FAULT_USER_ABORT;
    }

    return fault;
}

static void command_zero(void)
{
    g_exp5.calc_cmd_left = 0.0f;
    g_exp5.calc_cmd_right = 0.0f;
    g_exp5.applied_cmd_left = 0;
    g_exp5.applied_cmd_right = 0;
    g_exp5.sat_thrust = 0U;
    g_exp5.sat_torque = 0U;
    g_exp5.applied_output_enable = 0U;
    CAN_cmd_3508(0, 0);
}

static void enter_fault_stop(void)
{
    if (g_exp5.first_fault_flags == 0U && g_exp5.fault_flags != 0U)
    {
        g_exp5.first_fault_ms = g_exp5.elapsed_ms;
        g_exp5.first_fault_flags = g_exp5.fault_flags;
        g_exp5.first_fault_phase = g_exp5.phase;
    }
    g_exp5.running = 0U;
    g_exp5.finished = 1U;
    g_exp5.phase = EXP5_PHASE_FAULT;
    motor_stop_and_clear();
    command_zero();
}

static void update_tracking_gate(float reference_m, float yG, float e_sync)
{
    g_exp5.yG_m = yG;
    g_exp5.e_sync_m = e_sync;
    g_exp5.tracking_in_band =
        ((fabsf(reference_m - yG) <= EXP5_TRACK_BAND_M) &&
         (fabsf(e_sync) <= EXP5_SYNC_BAND_M)) ? 1U : 0U;
}

void motor_task(void const *argument)
{
    uint32_t arm_start_ms;
    uint32_t experiment_start_ms;
    uint32_t now_ms;
    uint8_t phase;
    uint8_t fault;
    float reference_m;
    float reference_velocity_mps;
    float reference_acceleration_mps2;
    
    /* Past states for 1st-order acceleration filtering */
    float y_vel_prev = 0.0f;
    float alpha_vel_prev = 0.0f;
    float qddot_filt[2] = {0.0f, 0.0f};

    (void)argument;
    
    /* Enable DWT cycle counter on Cortex-M4 for exact hardware profiling */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    /* Initialize E5 Controller configuration and reset state */
    e5_controller_init_config(&g_exp5.cfg);
    e5_controller_reset_state(&g_exp5.state);
    
    /* Apply overrides from experiment5_config.h */
    g_exp5.cfg.Ts = EXP5_CONTROL_TS_S;
    g_exp5.cfg.Le = EXP5_LE_M;
    g_exp5.cfg.Kf_L = EXP5_KF_LEFT_N_PER_COUNT;
    g_exp5.cfg.Kf_R = EXP5_KF_RIGHT_N_PER_COUNT;
    g_exp5.cfg.Imax_L = EXP5_CURRENT_LIMIT_LEFT;
    g_exp5.cfg.Imax_R = EXP5_CURRENT_LIMIT_RIGHT;
    g_exp5.cfg.slew_rate_max = EXP5_SLEW_RATE_MAX;

    g_exp5.cfg.M_hat[0] = EXP5_M_Y;
    g_exp5.cfg.M_hat[1] = EXP5_M_ALPHA;
    g_exp5.cfg.B_hat[0] = EXP5_B_Y;
    g_exp5.cfg.B_hat[1] = EXP5_B_ALPHA;
    g_exp5.cfg.K_hat[0] = 0.0f;
    g_exp5.cfg.K_hat[1] = EXP5_K_ALPHA;
    g_exp5.cfg.A_hat[0] = EXP5_A_Y;
    g_exp5.cfg.A_hat[1] = 0.0f;

    g_exp5.cfg.Gamma[0] = EXP5_GAMMA_Y;
    g_exp5.cfg.Gamma[1] = EXP5_GAMMA_ALPHA;
    g_exp5.cfg.K1[0] = EXP5_K1_Y;
    g_exp5.cfg.K1[1] = EXP5_K1_ALPHA;
    g_exp5.cfg.K2[0] = EXP5_K2_Y;
    g_exp5.cfg.K2[1] = EXP5_K2_ALPHA;
    g_exp5.cfg.phi[0] = EXP5_PHI_Y;
    g_exp5.cfg.phi[1] = EXP5_PHI_ALPHA;
    g_exp5.cfg.eps_v = EXP5_EPS_V;
    g_exp5.cfg.eps_alpha = EXP5_EPS_ALPHA;

    g_exp5.cfg.K_aw = EXP5_K_AW;
    g_exp5.cfg.lambda_aw = EXP5_LAMBDA_AW;

    g_exp5.cfg.enable_blnn = (EXP5_ENABLE_BLNN != 0U);
    g_exp5.cfg.enable_lc = (EXP5_ENABLE_LC != 0U);
    g_exp5.cfg.gamma_lc = EXP5_GAMMA_LC;
    g_exp5.cfg.nn_eps[0] = EXP5_NN_EPS_Y;
    g_exp5.cfg.nn_eps[1] = EXP5_NN_EPS_ALPHA;
    g_exp5.cfg.nn_eta[0] = EXP5_NN_ETA_Y;
    g_exp5.cfg.nn_eta[1] = EXP5_NN_ETA_ALPHA;

    g_exp5.phase = EXP5_PHASE_ARMING;
    g_exp5.inject_sync_err_m = 0.0f;
    g_exp5.inject_can_age_ms = 0U;
    g_exp5.inject_nan_trigger = 0U;
    g_exp5.inject_active = 0U;
    g_exp5.ctrl_call_count = 0U;
    g_exp5.first_fault_ms = 0U;
    g_exp5.first_fault_flags = 0U;
    g_exp5.first_fault_phase = 0U;
    exp5_streamer_init();
    command_zero();
    osDelay(1500U);

#if (EXP5_WORK_MODE == EXP5_MODE_LIVE_CAN)
    /* Hardware mode: Wait up to 3000 ms for active C620 ESC CAN feedback */
    uint32_t feedback_wait_start_ms = HAL_GetTick();
    while ((motor_3508[0].rx_count == 0U) || (motor_3508[1].rx_count == 0U))
    {
        if ((HAL_GetTick() - feedback_wait_start_ms) >= 3000U)
        {
            if (motor_3508[0].rx_count == 0U) g_exp5.fault_flags |= EXP5_FAULT_CAN_LEFT;
            if (motor_3508[1].rx_count == 0U) g_exp5.fault_flags |= EXP5_FAULT_CAN_RIGHT;
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
#else
    /* Independent Benchmark Mode: Bypasses CAN feedback waiting, runs instantly */
#endif

    motor_init();
    arm_start_ms = HAL_GetTick();

    while ((HAL_GetTick() - arm_start_ms) < EXP5_ARM_DELAY_MS)
    {
        now_ms = HAL_GetTick();
        fault = protection_check(now_ms);
        if (fault != 0U)
        {
            g_exp5.fault_flags |= fault;
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

#if EXP5_AUTORUN_ENABLE
    experiment_start_ms = HAL_GetTick();
    s_summary_sent = false;
    g_exp5.running = 1U;
    g_exp5.control_step_seq = 0U;
    g_exp5.active_control_steps = 0U;
    g_exp5.period_overrun_count = 0U;
    g_exp5.exec_overrun_count = 0U;
    g_exp5.overrun_count = 0U;
    g_exp5.task_exec_max_us = 0.0f;
    g_exp5.ctrl_exec_max_us = 0.0f;
    g_exp5.nan_injected_to_core = 0U;
    g_exp5.drain_status = 0U;
    g_exp5.summary_send_status = 0U;
    g_exp5.task_period_min_us = 9999.0f;
    g_exp5.task_period_max_us = 0.0f;
    g_exp5.max_sync_err_m = 0.0f;
    g_exp5.max_track_err_m = 0.0f;

    uint32_t last_loop_start = DWT->CYCCNT;
    TickType_t xLastWakeTime = xTaskGetTickCount();

    for (;;)
    {
        uint32_t loop_start = DWT->CYCCNT;
        
        /* 1. Track task loop wakeup interval (Jitter & Period Monitoring) */
        uint32_t period_cyc = loop_start - last_loop_start;
        last_loop_start = loop_start;
        g_exp5.task_period_cycles = period_cyc;
        g_exp5.task_period_us = (float)period_cyc / 168.0f;

        now_ms = HAL_GetTick();
        g_exp5.elapsed_ms = now_ms - experiment_start_ms;

        /* Feedback calculation (always updated every cycle for telemetry & safety) */
#if (EXP5_WORK_MODE == EXP5_MODE_LIVE_CAN)
        float pos_left_m = ((float)motor_3508[0].total_ecd - chassis_ang_init[0]) *
                           EXP5_METERS_PER_MOTOR_COUNT;
        float pos_right_m = -((float)motor_3508[1].total_ecd - chassis_ang_init[1]) *
                            EXP5_METERS_PER_MOTOR_COUNT;
        float vel_left_mps = ((float)motor_3508[0].speed_rpm / 60.0f) *
                             (2.0f * EXP5_PI * EXP5_DRIVE_RADIUS_M / EXP5_GEAR_RATIO);
        float vel_right_mps = -((float)motor_3508[1].speed_rpm / 60.0f) *
                              (2.0f * EXP5_PI * EXP5_DRIVE_RADIUS_M / EXP5_GEAR_RATIO);
#else
        float t_s = (float)g_exp5.elapsed_ms * 0.001f;
        float pos_left_m = reference_m * 0.98f;
        float pos_right_m = reference_m * 0.98f;
        float vel_left_mps = reference_velocity_mps * 0.98f;
        float vel_right_mps = reference_velocity_mps * 0.98f;
#endif
        g_exp5.yG_m = 0.5f * (pos_left_m + pos_right_m);
        g_exp5.e_sync_m = (pos_right_m - pos_left_m);

        /* Software Fault Injection & Autonomous Removal Logic:
         * Always executed unconditionally on every 1 ms tick, even in faulted state.
         * Stimulus is injected strictly during [2000, 3000) ms, and strictly cleared outside.
         */
        g_exp5.inject_active = 0U;
#if (EXP5_FAULT_INJECT_MODE == EXP5_FAULT_INJECT_SYNC)
        if (g_exp5.elapsed_ms >= 2000U && g_exp5.elapsed_ms < 3000U)
        {
            g_exp5.inject_sync_err_m = 0.008f; /* 8.0 mm > 5.0 mm threshold */
            g_exp5.inject_active = 1U;
        }
        else
        {
            g_exp5.inject_sync_err_m = 0.0f;   /* Stimulus strictly removed */
        }
#elif (EXP5_FAULT_INJECT_MODE == EXP5_FAULT_INJECT_CAN_TIMEOUT)
        if (g_exp5.elapsed_ms >= 2000U && g_exp5.elapsed_ms < 3000U)
        {
            g_exp5.inject_can_age_ms = 60U;    /* 60 ms > 50 ms threshold */
            g_exp5.inject_active = 1U;
        }
        else
        {
            g_exp5.inject_can_age_ms = 0U;     /* Stimulus strictly removed */
        }
#elif (EXP5_FAULT_INJECT_MODE == EXP5_FAULT_INJECT_NAN)
        if (g_exp5.elapsed_ms >= 2000U && g_exp5.elapsed_ms < 3000U)
        {
            g_exp5.inject_nan_trigger = 1U;    /* NaN numerical fault triggered */
            g_exp5.inject_active = 1U;
        }
        else
        {
            g_exp5.inject_nan_trigger = 0U;    /* Stimulus strictly removed */
        }
#endif

        if (g_exp5.fault_flags == 0U && !g_exp5.finished)
        {
            trajectory_reference(g_exp5.elapsed_ms,
                                 &phase,
                                 &reference_m,
                                 &reference_velocity_mps,
                                 &reference_acceleration_mps2);
            g_exp5.reference_m = reference_m;
            g_exp5.reference_velocity_mps = reference_velocity_mps;
            g_exp5.reference_acceleration_mps2 = reference_acceleration_mps2;
            g_exp5.phase = phase;
        }

        /* Statistics scoped strictly to active trajectory execution (Phases 1 to 5) */
        bool is_active_control = (g_exp5.running != 0U &&
                                  g_exp5.phase >= EXP5_PHASE_PRE_HOLD &&
                                  g_exp5.phase <= EXP5_PHASE_RECOVERY_HOLD);
        if (is_active_control)
        {
            g_exp5.active_control_steps++;
            if (g_exp5.active_control_steps > 10U)
            {
                if (g_exp5.task_period_us > g_exp5.task_period_max_us)
                {
                    g_exp5.task_period_max_us = g_exp5.task_period_us;
                }
                if (g_exp5.task_period_us < g_exp5.task_period_min_us)
                {
                    g_exp5.task_period_min_us = g_exp5.task_period_us;
                }
                if (g_exp5.task_period_us > 1050.0f)
                {
                    g_exp5.period_overrun_count++;
                    g_exp5.overrun_count++;
                }
            }
        }

        fault = protection_check(now_ms);
        g_exp5.fault_flags |= fault;

        if (g_exp5.fault_flags != 0U)
        {
            enter_fault_stop();
        }
        else if (g_exp5.elapsed_ms >= EXP5_TOTAL_MS)
        {
            g_exp5.running = 0U;
            g_exp5.finished = 1U;
            g_exp5.phase = EXP5_PHASE_DONE;
            motor_stop_and_clear();
            command_zero();
            update_tracking_gate(0.0f, 0.0f, 0.0f);
        }
        else
        {
            /* 2. Position & Velocity Feedback */
            float q_meas[2], qdot_meas[2], qd[2], qdot_d[2], qddot_d[2];

#if (EXP5_WORK_MODE == EXP5_MODE_LIVE_CAN)
            q_meas[0] = 0.5f * (pos_left_m + pos_right_m);
            q_meas[1] = (pos_right_m - pos_left_m) / EXP5_LE_M;
            qdot_meas[0] = 0.5f * (vel_left_mps + vel_right_mps);
            qdot_meas[1] = (vel_right_mps - vel_left_mps) / EXP5_LE_M;
#else
            q_meas[0] = reference_m * 0.98f;
            q_meas[1] = 0.0001f * sinf(5.0f * t_s);
            qdot_meas[0] = reference_velocity_mps * 0.98f;
            qdot_meas[1] = 0.0005f * cosf(5.0f * t_s);
#endif

#if (EXP5_FAULT_INJECT_MODE == EXP5_FAULT_INJECT_NAN)
            if (g_exp5.inject_nan_trigger != 0U)
            {
                q_meas[0] = (float)NAN; /* True IEEE 754 NaN input into control core */
                g_exp5.nan_injected_to_core++;
            }
#endif

            qd[0] = reference_m;
            qd[1] = 0.0f;
            qdot_d[0] = reference_velocity_mps;
            qdot_d[1] = 0.0f;
            qddot_d[0] = reference_acceleration_mps2;
            qddot_d[1] = 0.0f;

            /* Filtered modal acceleration estimate (50 Hz cutoff filter) */
            float raw_accel_y = (qdot_meas[0] - y_vel_prev) / EXP5_CONTROL_TS_S;
            float raw_accel_a = (qdot_meas[1] - alpha_vel_prev) / EXP5_CONTROL_TS_S;
            y_vel_prev = qdot_meas[0];
            alpha_vel_prev = qdot_meas[1];
            qddot_filt[0] = 0.76f * qddot_filt[0] + 0.24f * raw_accel_y;
            qddot_filt[1] = 0.76f * qddot_filt[1] + 0.24f * raw_accel_a;

            /* 3. Execute E5 Coordinated BLNN Step with DWT Cycle Profiling */
            g_exp5.ctrl_call_count++;

            uint32_t t_ctrl_start = DWT->CYCCNT;
            e5_controller_step(
                q_meas, qdot_meas, qddot_filt,
                qd, qdot_d, qddot_d,
                &g_exp5.cfg, &g_exp5.state, &g_exp5.info);
            uint32_t t_ctrl_end = DWT->CYCCNT;
            g_exp5.ctrl_cycles = t_ctrl_end - t_ctrl_start;
            g_exp5.ctrl_exec_us = (float)g_exp5.ctrl_cycles / 168.0f;
            if (is_active_control && g_exp5.active_control_steps > 10U)
            {
                if (g_exp5.ctrl_exec_us > g_exp5.ctrl_exec_max_us)
                {
                    g_exp5.ctrl_exec_max_us = g_exp5.ctrl_exec_us;
                }
            }

            /* Same-cycle exception handling: check controller numerical fault immediately */
            if (g_exp5.info.numerical_fault)
            {
                g_exp5.numerical_fault_latched = 1U;
                g_exp5.fault_flags |= EXP5_FAULT_NUMERICAL_NAN;
                enter_fault_stop();
            }

            /* 4. Disambiguated Command Separation & Saturation Status */
            g_exp5.calc_cmd_left = g_exp5.info.iL_cmd;
            g_exp5.calc_cmd_right = g_exp5.info.iR_cmd;

            g_exp5.sat_thrust = g_exp5.info.is_sat_thrust ? 1U : 0U;
            g_exp5.sat_torque = g_exp5.info.is_sat_torque ? 1U : 0U;
            if (g_exp5.info.sat_alloc)
            {
                g_exp5.sat_latched = 1U;
            }

            update_tracking_gate(reference_m, q_meas[0], EXP5_LE_M * q_meas[1]);
            /* Board-side 1 kHz sampled peak error accumulation: strictly scoped to Phase 2 to 5 */
            bool is_eval_window = (g_exp5.running != 0U &&
                                   g_exp5.phase >= EXP5_PHASE_FORWARD &&
                                   g_exp5.phase <= EXP5_PHASE_RECOVERY_HOLD);
            if (is_eval_window)
            {
                float abs_sync = fabsf(g_exp5.e_sync_m);
                if (abs_sync > g_exp5.max_sync_err_m) g_exp5.max_sync_err_m = abs_sync;
                float abs_track = fabsf(reference_m - g_exp5.yG_m);
                if (abs_track > g_exp5.max_track_err_m) g_exp5.max_track_err_m = abs_track;
            }

            /* 5. Safe Motor Output Routing: Strict separation between calculated and applied */
            int16_t clamped_left = current_command(g_exp5.calc_cmd_left, EXP5_CURRENT_LIMIT_LEFT);
            int16_t clamped_right = current_command(g_exp5.calc_cmd_right, EXP5_CURRENT_LIMIT_RIGHT);

#if EXP5_ENABLE_MOTOR_OUTPUT
            g_exp5.applied_output_enable = 1U;
            g_exp5.applied_cmd_left = clamped_left;
            g_exp5.applied_cmd_right = clamped_right;
            CAN_cmd_3508(g_exp5.applied_cmd_left, g_exp5.applied_cmd_right);
#else
            g_exp5.applied_output_enable = 0U;
            g_exp5.applied_cmd_left = 0;
            g_exp5.applied_cmd_right = 0;
            (void)clamped_left;
            (void)clamped_right;
            CAN_cmd_3508(0, 0); /* Zero torque command transmitted to C620 ESC */
#endif
        }

        /* 6. Runtime Stack Watermark Monitoring */
        g_exp5.stack_high_watermark_words = (uint32_t)uxTaskGetStackHighWaterMark(NULL);
        g_exp5.stack_min_free_bytes = g_exp5.stack_high_watermark_words * sizeof(StackType_t);

        /* 6. Runtime Stack Watermark Monitoring */
        g_exp5.stack_high_watermark_words = (uint32_t)uxTaskGetStackHighWaterMark(NULL);
        g_exp5.stack_min_free_bytes = g_exp5.stack_high_watermark_words * sizeof(StackType_t);

        /* 7. 1 kHz High-Speed Streamer Frame Construction (62 Bytes, 921600 bps) */
        exp5_highspeed_frame_t frame;
        memset(&frame, 0, sizeof(frame));
        frame.header[0] = EXP5_STREAMER_FRAME_HEADER_0;
        frame.header[1] = EXP5_STREAMER_FRAME_HEADER_DATA;
        frame.seq = g_exp5.control_step_seq++;
        frame.timestamp_ms = g_exp5.elapsed_ms;
        frame.phase = g_exp5.phase;
        frame.fault_flags = g_exp5.fault_flags;
        frame.dma_overflow_count = exp5_streamer_get_overflow_sat();

        uint8_t status = 0U;
        if (g_exp5.sat_latched) status |= EXP5_STATUS_SAT_LATCHED;
        if (g_exp5.sat_thrust)  status |= EXP5_STATUS_SAT_THRUST;
        if (g_exp5.sat_torque)  status |= EXP5_STATUS_SAT_TORQUE;
        if (g_exp5.tracking_in_band) status |= EXP5_STATUS_TRACKING_IN_BAND;
        if (exp5_streamer_get_overflow_latched()) status |= EXP5_STATUS_DMA_OVERFLOW_LATCHED;
        if (g_exp5.numerical_fault_latched || g_exp5.info.numerical_fault) status |= EXP5_STATUS_NUMERICAL_FAULT;
        if (g_exp5.applied_output_enable)   status |= EXP5_STATUS_OUTPUT_ENABLED;
        if (g_exp5.inject_active)           status |= EXP5_STATUS_INJECT_ACTIVE;
        frame.status_flags = status;

        uint32_t c_us_x10 = (uint32_t)(g_exp5.ctrl_exec_us * 10.0f + 0.5f);
        frame.ctrl_exec_us_x10 = (c_us_x10 > 65535U) ? 65535U : (uint16_t)c_us_x10;

        /* 1. Trajectory & Kinematics */
        frame.ref_pos_m = g_exp5.reference_m;
        frame.ref_vel_mps = g_exp5.reference_velocity_mps;
        frame.pos_L_m = pos_left_m;
        frame.pos_R_m = pos_right_m;

        float v_L = vel_left_mps * 1000.0f;
        if (v_L > 32767.0f) v_L = 32767.0f;
        else if (v_L < -32768.0f) v_L = -32768.0f;
        frame.vel_L_mmps = (int16_t)v_L;

        float v_R = vel_right_mps * 1000.0f;
        if (v_R > 32767.0f) v_R = 32767.0f;
        else if (v_R < -32768.0f) v_R = -32768.0f;
        frame.vel_R_mmps = (int16_t)v_R;

        frame.rpm_L = motor_3508[0].speed_rpm;
        frame.rpm_R = motor_3508[1].speed_rpm;

        /* 2. Sliding Mode Variables */
        frame.sigma_y = g_exp5.info.sigma[0];
        frame.sigma_alpha = g_exp5.info.sigma[1];

        /* 3. Pre-Allocation Requests */
        frame.F_cmd = g_exp5.info.F_cmd;
        frame.T_cmd = g_exp5.info.T_cmd;

        /* 4. Neural Network Modal Compensation */
        frame.F_nn = g_exp5.info.v_nn_raw_modal[0];
        frame.T_nn = g_exp5.info.v_nn_raw_modal[1];

        /* 5. Anti-Windup States */
        frame.z_aw_L = g_exp5.state.z_aw[0];
        frame.z_aw_R = g_exp5.state.z_aw[1];

        /* 6. Current Allocation & Commands */
        frame.calc_cmd_L = g_exp5.calc_cmd_left;
        frame.calc_cmd_R = g_exp5.calc_cmd_right;
        frame.applied_cmd_L = g_exp5.applied_cmd_left;
        frame.applied_cmd_R = g_exp5.applied_cmd_right;

        /* 7. Coordination & Adaptation Metrics */
        float rho_y = g_exp5.info.rho_lc[0];
        if (rho_y < 0.0f) rho_y = 0.0f;
        if (rho_y > 1.0f) rho_y = 1.0f;
        frame.rho_lc_y_x1000 = (uint16_t)(rho_y * 1000.0f + 0.5f);

        float rho_alpha = g_exp5.info.rho_lc[1];
        if (rho_alpha < 0.0f) rho_alpha = 0.0f;
        if (rho_alpha > 1.0f) rho_alpha = 1.0f;
        frame.rho_lc_alpha_x1000 = (uint16_t)(rho_alpha * 1000.0f + 0.5f);

        float wy = g_exp5.info.norm_W_y * 1000.0f;
        if (wy < 0.0f) wy = 0.0f;
        if (wy > 65535.0f) wy = 65535.0f;
        frame.norm_W_y_x1000 = (uint16_t)(wy + 0.5f);

        float wa = g_exp5.info.norm_W_alpha * 1000.0f;
        if (wa < 0.0f) wa = 0.0f;
        if (wa > 65535.0f) wa = 65535.0f;
        frame.norm_W_alpha_x1000 = (uint16_t)(wa + 0.5f);

        /* Automatic silence window & SUMMARY frame management */
        bool is_terminal_window = false;

        #define EXP5_FAULT_COAST_MS 1500U

        if (g_exp5.first_fault_flags != 0U && g_exp5.elapsed_ms >= (g_exp5.first_fault_ms + EXP5_FAULT_COAST_MS))
        {
            is_terminal_window = true;
        }
        else if (g_exp5.finished && g_exp5.elapsed_ms >= (EXP5_TOTAL_MS + 2000U))
        {
            is_terminal_window = true;
        }

        if (is_terminal_window)
        {
            if (!s_summary_sent)
            {
                /* 1. Drain queue and verify completion */
                bool drain_ok = exp5_streamer_drain(100U);
                g_exp5.drain_status = drain_ok ? 1U : 0U;

                /* 2. Format and send SUMMARY frame */
                exp5_summary_frame_t summary;
                memset(&summary, 0, sizeof(summary));
                exp5_streamer_diag_t diag;
                exp5_streamer_get_diag(&diag);

                summary.data_generated = diag.data_generated;
                summary.enqueue_ok = diag.enqueue_ok;
                summary.enqueue_drop = diag.enqueue_drop;
                summary.dma_start_ok = diag.dma_start_ok;
                summary.dma_start_busy = (diag.dma_start_busy > 65535U) ? 65535U : (uint16_t)diag.dma_start_busy;
                summary.dma_start_error = (diag.dma_start_error > 65535U) ? 65535U : (uint16_t)diag.dma_start_error;

                uint32_t sync_um = (uint32_t)(g_exp5.max_sync_err_m * 1000000.0f + 0.5f);
                summary.max_sync_err_um = (sync_um > 65535U) ? 65535U : (uint16_t)sync_um;

                uint32_t track_um = (uint32_t)(g_exp5.max_track_err_m * 1000000.0f + 0.5f);
                summary.max_track_err_um = (track_um > 65535U) ? 65535U : (uint16_t)track_um;

                summary.tx_complete = diag.tx_complete;
                summary.uart_error = diag.uart_error;
                summary.tx_aborted = diag.tx_aborted;
                summary.queue_high_watermark = diag.queue_high_watermark;

                summary.ctrl_call_count = g_exp5.ctrl_call_count;
                summary.first_fault_ms = g_exp5.first_fault_ms;
                summary.first_fault_flags = g_exp5.first_fault_flags;
                summary.first_fault_phase = g_exp5.first_fault_phase;
                summary.final_phase = g_exp5.phase;
                summary.final_fault_flags = g_exp5.fault_flags;

                uint32_t t_max_x10 = (uint32_t)(g_exp5.task_exec_max_us * 10.0f + 0.5f);
                summary.task_exec_max_us_x10 = (t_max_x10 > 65535U) ? 65535U : (uint16_t)t_max_x10;

                uint32_t c_max_x10 = (uint32_t)(g_exp5.ctrl_exec_max_us * 10.0f + 0.5f);
                summary.ctrl_exec_max_us_x10 = (c_max_x10 > 65535U) ? 65535U : (uint16_t)c_max_x10;

                summary.nan_injected_count = (g_exp5.nan_injected_to_core > 255U) ? 255U : (uint8_t)g_exp5.nan_injected_to_core;
                summary.drain_status = g_exp5.drain_status;

                bool send_ok = exp5_streamer_send_summary(&summary, 50U);
                g_exp5.summary_send_status = send_ok ? 1U : 0U;
                s_summary_sent = true;
            }
        }
        else
        {
#if (EXP5_STREAMER_DECIMATION > 1U)
            if ((frame.seq % EXP5_STREAMER_DECIMATION) == 0U)
            {
                exp5_streamer_post_frame(&frame);
            }
#else
            exp5_streamer_post_frame(&frame);
#endif
        }

        /* 8. Full Cycle Work Execution Time: strictly encompasses feedback, protection,
         * control core, command routing, frame formatting, CRC16, and DMA enqueue.
         */
        uint32_t loop_end = DWT->CYCCNT;
        g_exp5.task_cycles = loop_end - loop_start;
        g_exp5.task_exec_us = (float)g_exp5.task_cycles / 168.0f;
        if (is_active_control && g_exp5.active_control_steps > 10U)
        {
            if (g_exp5.task_exec_us > g_exp5.task_exec_max_us)
            {
                g_exp5.task_exec_max_us = g_exp5.task_exec_us;
            }
            if (g_exp5.task_exec_us > 1000.0f)
            {
                g_exp5.exec_overrun_count++;
                g_exp5.overrun_count++;
            }
        }

        vTaskDelayUntil(&xLastWakeTime, pdMS_TO_TICKS(1U));
    }
#else
    for (;;)
    {
        command_zero();
        osDelay(1U);
    }
#endif
}
