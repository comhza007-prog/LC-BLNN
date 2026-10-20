#ifndef EXPERIMENT3_CONFIG_H
#define EXPERIMENT3_CONFIG_H

/*
 * Experiment 3 C2a hardware-admission firmware (fixed 2 kg load):
 *   - controller: C2a robust error-surface controller (controller id 1)
 *   - allocation: independent clipping; no dynamic anti-windup
 *   - goal: verify sign and stability on a 10 mm low-speed trajectory
 *
 * This is an H4 qualification run, not a formal 2 x 2 trial.
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

#define EXP3_CURRENT_LIMIT               2400.0f

/* Keep the internal PID request range larger than the firmware current clamp
 * so the unsaturated request and external saturation residual are observable.
 */
#define EXP3_REQUEST_LIMIT               4000.0f
#define EXP3_CURRENT_I_LIMIT             500.0f
#define EXP3_ADDED_LOAD_KG               2.000f
#define EXP3_CONTROLLER_ID               1U

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
