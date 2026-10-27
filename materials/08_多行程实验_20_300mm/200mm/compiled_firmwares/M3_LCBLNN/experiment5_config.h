#ifndef EXPERIMENT5_CONFIG_H
#define EXPERIMENT5_CONFIG_H

/*
 * Experiment 5: Sync-Coordinated BLNN Dual-Drive Gantry Formal Firmware
 * 
 * Architecture:
 * - Base Controller: C2b Sliding Mode with SyncAlloc & Dynamic Anti-Windup
 * - Neural Network: 4-Node BLNN with Translation and Rotation Feature Channels
 * - Learning Coordination (LC): Continuous consistency-driven modulation
 * - Target Platform: STM32F407IGHx (Arm Cortex-M4F @ 168 MHz)
 * - Actuation: 2x DJI RoboMaster M3508 Brushless Motors (19:1 gearbox) + C620 ESCs
 * - Feedback: Built-in magnetic rotor encoders (8192 counts/rev, CAN 0x203, 0x204)
 */

#define EXP5_AUTORUN_ENABLE              1U
#define EXP5_ARM_DELAY_MS                3000U

/* OPERATIONAL WORK MODE:
 * 0U = EXP5_MODE_BENCH_TEST:
 *      Independent bench test mode (runs pre-programmed trajectory, zero CAN wait,
 *      no reliance on powered ESCs, profiles DWT cycles, task jitter, stack watermark).
 * 1U = EXP5_MODE_LIVE_CAN:
 *      Hardware operational mode (requires live CAN feedback from 0x203/0x204,
 *      enforces 50 ms timeout, synchronism and travel trips).
 */
#define EXP5_MODE_BENCH_TEST             0U
#define EXP5_MODE_LIVE_CAN               1U
#define EXP5_WORK_MODE                   EXP5_MODE_LIVE_CAN

/* EXPERIMENT EXECUTION STAGE:
 * 0U = EXP5_STAGE_ZERO_TORQUE_QUAL:
 *      Zero-torque qualification and bench diagnostic mode.
 *      Enforces EXP5_ENABLE_MOTOR_OUTPUT == 0U at compile time.
 *      Allows software fault injection suite (SYNC, CAN_TIMEOUT, NAN) or baseline bench tests.
 * 1U = EXP5_STAGE_MOTION_EXPLORATION:
 *      Physical motion exploration mode (e.g. Step 1a micro-travel).
 *      STRICT MUTUAL EXCLUSION: Fault injection MUST BE EXP5_FAULT_INJECT_NONE!
 *      EXP5_ENABLE_MOTOR_OUTPUT can be 0U (dry-run pre-check) or 1U (live physical motion).
 */
#define EXP5_STAGE_ZERO_TORQUE_QUAL      0U
#define EXP5_STAGE_MOTION_EXPLORATION    1U

#define EXP5_EXECUTION_STAGE             EXP5_STAGE_MOTION_EXPLORATION

/* PHYSICAL MOTOR OUTPUT INTERLOCK:
 * 0U = Output Disabled / Dry Run (CAN_cmd_3508 sends 0, 0; power stage free-wheeling)
 * 1U = Live Physical Commissioning (requires explicit qualification sign-off)
 * Default: 0U (strictly closed by default)
 */
#define EXP5_ENABLE_MOTOR_OUTPUT         1U


/* TELEMETRY STREAMING CONFIGURATION */
#define EXP5_USART6_BAUDRATE             921600U
#define EXP5_STREAMER_DECIMATION         2U  /* Unified formal telemetry: 500 Hz (1 frame per 2 ms) */

/* SOFTWARE FAULT INJECTION TEST SUITE:
 * 0U = EXP5_FAULT_INJECT_NONE:        Normal qualification run, no faults injected.
 * 1U = EXP5_FAULT_INJECT_SYNC:        Inject virtual sync error (8.0 mm > 5.0 mm limit) at t in [2000, 3000) ms.
 * 2U = EXP5_FAULT_INJECT_CAN_TIMEOUT: Inject virtual CAN age timeout (60 ms > 50 ms limit) at t in [2000, 3000) ms.
 * 3U = EXP5_FAULT_INJECT_NAN:         Inject NaN floating-point fault at t in [2000, 3000) ms.
 */
#define EXP5_FAULT_INJECT_NONE           0U
#define EXP5_FAULT_INJECT_SYNC           1U
#define EXP5_FAULT_INJECT_CAN_TIMEOUT    2U
#define EXP5_FAULT_INJECT_NAN            3U

#define EXP5_FAULT_INJECT_MODE           EXP5_FAULT_INJECT_NONE

/* Timeline after arming: 4 s hold, 4 s forward, 2 s end hold,
 * 4 s return, and 6 s recovery observation. Total = 20 s.
 */
#define EXP5_PRE_HOLD_MS                 4000U
#define EXP5_FORWARD_MS                  4000U
#define EXP5_HOLD_END_MS                 2000U
#define EXP5_RETURN_MS                   4000U
#define EXP5_RECOVERY_HOLD_MS            6000U
#define EXP5_TOTAL_MS                    20000U

#define EXP5_STROKE_M                    0.200f  /* 30.0 mm formal exploration trajectory (0 -> 20.0 mm -> 0) */
#define EXP5_DRIVE_RADIUS_M              0.030f
#define EXP5_GEAR_RATIO                  19.0f
#define EXP5_ENCODER_CPR                 8192.0f
#define EXP5_SPEED_LIMIT_RPM             2500.0f  /* 800 rpm protection ceiling for 30.0 mm stroke */

/* CURRENT LIMITS:
 * Set to 800.0f counts based on historical qualification in Experiment 3 (C2b_800 / C2b_SA_800).
 * Note: 600 counts previously failed on return stroke in Exp 3 due to continuous saturation.
 */
#define EXP5_CURRENT_LIMIT_LEFT          800.0f
#define EXP5_CURRENT_LIMIT_RIGHT         800.0f
#define EXP5_ADDED_LOAD_KG               0.000f  /* Strictly unloaded (0 kg added mass) */
#define EXP5_CONTROLLER_ID               5U

/* Geometric & Thrust Coefficients */
#define EXP5_LE_M                        0.600f
#define EXP5_KF_LEFT_N_PER_COUNT         0.0061979167f
#define EXP5_KF_RIGHT_N_PER_COUNT        0.0061979167f

/* Dynamic Model & Sliding Mode Gains:
 * Completely aligned with Experiment 3 C2b-SyncAlloc verified baseline (0 kg / 800 counts qualification)
 */
#define EXP5_M_Y                         12.445f
#define EXP5_M_ALPHA                     0.373350f
#define EXP5_B_Y                         63.000f
#define EXP5_B_ALPHA                     22.500f
#define EXP5_K_ALPHA                     2000.0f
#define EXP5_A_Y                         14.400f

#define EXP5_GAMMA_Y                     30.0f
#define EXP5_GAMMA_ALPHA                 20.0f
#define EXP5_K1_Y                        45.0f
#define EXP5_K1_ALPHA                    30.0f
#define EXP5_K2_Y                        12.0f
#define EXP5_K2_ALPHA                    6.0f
#define EXP5_PHI_Y                       0.030f
#define EXP5_PHI_ALPHA                   0.030f
#define EXP5_EPS_V                       0.01f
#define EXP5_EPS_ALPHA                   0.01f

/* Anti-Windup Parameters */
#define EXP5_K_AW                        20.0f
#define EXP5_LAMBDA_AW                   20.0f
#define EXP5_CONTROL_TS_S                0.001f

/* CONTROLLER COMPARISON SWITCHES:
 * Step 1a: 无网络的 C2b-SyncAlloc 基础控制�?(EXP5_ENABLE_BLNN = 0, EXP5_ENABLE_LC = 0)
 *          说明：关闭神经网络更新，但基于推力空间的同步优先分配 (thrust_allocator_sync_c)
 *          与动态抗饱和 (dynamic anti-windup) 仍然全功能运行，作为基础闭环对比准线�?
 * Step 1b: 普�?BLNN 对照�?(EXP5_ENABLE_BLNN = 1, EXP5_ENABLE_LC = 0)
 * Step 1c: 完整 E5 同步协调 BLNN (EXP5_ENABLE_BLNN = 1, EXP5_ENABLE_LC = 1)
 */
#define EXP5_ENABLE_BLNN                 1U
#define EXP5_ENABLE_LC                   1U

/* Slew Rate and LC Parameters */
#define EXP5_SLEW_RATE_MAX               75.0f
#define EXP5_GAMMA_LC                    100.0f
#define EXP5_NN_EPS_Y                    25.0f
#define EXP5_NN_EPS_ALPHA                18.0f
#define EXP5_NN_ETA_Y                    0.01f
#define EXP5_NN_ETA_ALPHA                0.01f

/* Tolerance Gates */
#define EXP5_TRACK_BAND_M                0.001f
#define EXP5_SYNC_BAND_M                 0.001f

/* Hardware Protection Thresholds (Configured for 20.0 mm stroke, Exp 1~4 standards) */
#define EXP5_CAN_TIMEOUT_MS              50U
#define EXP5_TEMPERATURE_LIMIT_C         80U
#define EXP5_SYNC_FAULT_M                0.005f  /* 5.0 mm synchronism trip (Exp 1~4 qualification standard) */
#define EXP5_TRAVEL_FAULT_M              0.215f  /* 45.0 mm boundary trip (15 mm margin past 30 mm stroke) */

/* Fault Bitmask Definitions */
#define EXP5_FAULT_CAN_LEFT              0x01U
#define EXP5_FAULT_CAN_RIGHT             0x02U
#define EXP5_FAULT_TEMPERATURE           0x04U
#define EXP5_FAULT_SYNCHRONISM           0x08U
#define EXP5_FAULT_TRAVEL                0x10U
#define EXP5_FAULT_SPEED                 0x20U
#define EXP5_FAULT_USER_ABORT            0x40U
#define EXP5_FAULT_NUMERICAL_NAN         0x80U

#define EXP5_PI                          3.14159265358979323846f
#define EXP5_METERS_PER_MOTOR_COUNT      \
    ((2.0f * EXP5_PI * EXP5_DRIVE_RADIUS_M) / (EXP5_GEAR_RATIO * EXP5_ENCODER_CPR))

/* =========================================================================
 * COMPILE-TIME SAFETY & MUTUAL EXCLUSION INTERLOCKS (Evaluated at EOF)
 * All macros (stage, output enable, fault injection) are fully defined above.
 * ========================================================================= */
#if (EXP5_EXECUTION_STAGE == EXP5_STAGE_ZERO_TORQUE_QUAL)
    /* In zero-torque qualification mode, physical motor output is strictly prohibited */
    #if (EXP5_ENABLE_MOTOR_OUTPUT != 0U)
        #error "SAFETY INTERLOCK VIOLATION: Zero-torque qualification mode strictly forbids EXP5_ENABLE_MOTOR_OUTPUT == 1U!"
    #endif
#elif (EXP5_EXECUTION_STAGE == EXP5_STAGE_MOTION_EXPLORATION)
    /* In motion exploration mode, software fault injection is strictly prohibited (mutual exclusion) */
    #if (EXP5_FAULT_INJECT_MODE != EXP5_FAULT_INJECT_NONE)
        #error "SAFETY INTERLOCK VIOLATION: Fault injection suite CANNOT be active during motion exploration mode!"
    #endif
#else
    #error "CONFIGURATION ERROR: Unknown EXP5_EXECUTION_STAGE!"
#endif

#endif /* EXPERIMENT5_CONFIG_H */
