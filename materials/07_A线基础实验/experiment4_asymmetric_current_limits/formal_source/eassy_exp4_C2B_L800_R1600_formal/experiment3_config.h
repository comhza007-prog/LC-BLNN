#ifndef EXPERIMENT3_CONFIG_H
#define EXPERIMENT3_CONFIG_H

/*
 * Experiment 4 C2b asymmetric-actuator firmware (fixed 2 kg load):
 *   - controller: C2b dynamic anti-windup controller (controller id 3)
 *   - allocation: independent clipping; external-saturation residual feedback
 *   - goal: quantify robustness when the left and right current capabilities
 *     are deliberately different on the frozen 10 mm low-speed trajectory
 */
#define EXP3_AUTORUN_ENABLE              1U
#define EXP3_ARM_DELAY_MS                3000U

/* Timeline after arming: 4 s hold, 4 s forward, 2 s end hold,
 * 4 s return, and 6 s recovery observation. Total = 20 s.
 */
#define EXP3_PRE_HOLD_MS                 4000U
#define EXP3_FORWARD_MS                  4000U
#define EXP3_HOLD_END_MS                 2000U
#define EXP3_RETURN_MS                   4000U
#define EXP3_RECOVERY_HOLD_MS            6000U
#define EXP3_TOTAL_MS                    20000U

#define EXP3_STROKE_M                    0.010f
#define EXP3_DRIVE_RADIUS_M              0.030f
#define EXP3_GEAR_RATIO                  19.0f
#define EXP3_ENCODER_CPR                 8192.0f
#define EXP3_SPEED_LIMIT_RPM             300.0f

#define EXP3_CURRENT_TIER_HIGH           1U
#define EXP3_CURRENT_TIER                EXP3_CURRENT_TIER_HIGH

/* Experiment 4 varies these two limits while keeping their sum constant.
 * The generated formal projects use 1200/1200, 800/1600, or 1600/800.
 */
#define EXP4_CURRENT_LIMIT_LEFT          800.0f
#define EXP4_CURRENT_LIMIT_RIGHT         1600.0f

/* Keep the internal PID request range larger than the firmware current clamp
 * so the unsaturated request and external saturation residual are observable.
 */
#define EXP3_REQUEST_LIMIT               4000.0f
#define EXP3_CURRENT_I_LIMIT             500.0f
#define EXP3_ADDED_LOAD_KG               2.000f
#define EXP3_CONTROLLER_ID               3U

/* C2a nominal model and gains, frozen from the simulation controller.
 * Kf remains a model assumption until the separate H3 force calibration is
 * completed; therefore this short-stroke build is qualification-only.
 */
#define EXP3_C2_LE_M                     0.600f
#define EXP3_C2_KF_LEFT_N_PER_COUNT      0.0061979167f
#define EXP3_C2_KF_RIGHT_N_PER_COUNT     0.0061979167f
#define EXP3_C2_M_Y                      12.445f
#define EXP3_C2_M_ALPHA                  0.373350f
#define EXP3_C2_B_Y                      63.0f
#define EXP3_C2_B_ALPHA                  22.5f
#define EXP3_C2_K_ALPHA                  2000.0f
#define EXP3_C2_A_Y                      14.4f
#define EXP3_C2_GAMMA_Y                  30.0f
#define EXP3_C2_GAMMA_ALPHA              20.0f
#define EXP3_C2_K1_Y                     45.0f
#define EXP3_C2_K1_ALPHA                 30.0f
#define EXP3_C2_K2_Y                     12.0f
#define EXP3_C2_K2_ALPHA                 6.0f
#define EXP3_C2_PHI_Y                    0.03f
#define EXP3_C2_PHI_ALPHA                0.03f
#define EXP3_C2_EPS_V                    0.01f
#define EXP3_C2_EPS_ALPHA                0.01f

/* C2b dynamic anti-windup state:
 *   z(k+1) = z(k) + Ts * (Kaw * Delta_i_external - lambda * z(k))
 * The 1 ms task period and gains are frozen from controller_c2b_aw.m.
 */
#define EXP3_C2_K_AW                     20.0f
#define EXP3_C2_LAMBDA_AW                20.0f
#define EXP3_CONTROL_TS_S                0.001f

/* Recovery is evaluated offline after the final saturation sample. */
#define EXP3_TRACK_BAND_M                0.001f
#define EXP3_SYNC_BAND_M                 0.001f

/* Software protection thresholds. A physical power-cut E-stop is required. */
#define EXP3_CAN_TIMEOUT_MS              50U
#define EXP3_TEMPERATURE_LIMIT_C         80U
#define EXP3_SYNC_FAULT_M                0.005f
#define EXP3_TRAVEL_FAULT_M              0.025f

#define EXP3_PI                          3.14159265358979323846f
#define EXP3_METERS_PER_MOTOR_COUNT      \
    ((2.0f * EXP3_PI * EXP3_DRIVE_RADIUS_M) / (EXP3_GEAR_RATIO * EXP3_ENCODER_CPR))

#endif
