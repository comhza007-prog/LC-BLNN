#ifndef MOTOR_TASK_H
#define MOTOR_TASK_H

#include "FreeRTOS.h"
#include "task.h"
#include "main.h"
#include "cmsis_os.h"
#include "CAN_receive.h"
#include "motor.h"
#include "experiment5_config.h"
#include "controller_e5_blnn_lc.h"
#include "exp5_streamer.h"

typedef enum
{
    EXP5_PHASE_ARMING = 0,
    EXP5_PHASE_PRE_HOLD = 1,
    EXP5_PHASE_FORWARD = 2,
    EXP5_PHASE_HOLD_END = 3,
    EXP5_PHASE_RETURN = 4,
    EXP5_PHASE_RECOVERY_HOLD = 5,
    EXP5_PHASE_DONE = 6,
    EXP5_PHASE_FAULT = 9
} exp5_phase_t;

typedef struct
{
    /* High-resolution Timing & Jitter Profile (DWT @ 168 MHz) */
    volatile uint32_t ctrl_cycles;            /* Core control algorithm execution cycles */
    volatile float ctrl_exec_us;              /* Core control algorithm execution time in us */
    volatile float ctrl_exec_max_us;          /* Peak core control algorithm execution time in us */
    volatile uint32_t task_cycles;            /* Entire motor_task loop execution cycles */
    volatile float task_exec_us;              /* Entire motor_task loop execution time in us */
    volatile float task_exec_max_us;          /* Peak task loop execution time in us */
    
    volatile uint32_t task_period_cycles;     /* Elapsed cycles between consecutive task wakeups */
    volatile float task_period_us;            /* Actual interval between task wakeups in us */
    volatile float task_period_min_us;        /* Minimum loop period in us */
    volatile float task_period_max_us;        /* Maximum loop period in us (upper jitter bound) */
    volatile uint32_t period_overrun_count;   /* Wakeup interval exceedance count (> 1050 us) */
    volatile uint32_t exec_overrun_count;     /* Task execution deadline exceedance count (> 1000 us) */
    volatile uint32_t overrun_count;          /* Total overrun events = period_overrun + exec_overrun */
    
    /* Runtime Stack Watermark */
    volatile uint32_t stack_high_watermark_words; /* FreeRTOS minimum unused stack in words */
    volatile uint32_t stack_min_free_bytes;       /* Minimum unused stack in bytes */

    /* Disambiguated Output Commands */
    volatile float calc_cmd_left;             /* Algorithm calculated current command left (counts) */
    volatile float calc_cmd_right;            /* Algorithm calculated current command right (counts) */
    volatile int16_t applied_cmd_left;        /* Actually transmitted CAN current command left */
    volatile int16_t applied_cmd_right;       /* Actually transmitted CAN current command right */
    volatile uint8_t applied_output_enable;   /* Output enable state (0 = zero output safe mode, 1 = live) */

    /* Allocation & Saturation Flags */
    volatile uint8_t sat_thrust;              /* Thrust allocator limited total thrust FG */
    volatile uint8_t sat_torque;              /* Thrust allocator limited sync torque Talpha */
    volatile uint8_t sat_latched;             /* Saturation occurred flag */
    volatile uint8_t numerical_fault_latched; /* NaN / Inf fault latched */
    volatile uint8_t fault_flags;             /* Fault code bitmask */

    /* Software Fault Injection Stimulus Fields */
    volatile float inject_sync_err_m;         /* Injected virtual synchronism offset (m) */
    volatile uint32_t inject_can_age_ms;      /* Injected virtual CAN age offset (ms) */
    volatile uint8_t inject_nan_trigger;      /* Injected virtual NaN trigger flag */
    volatile uint8_t inject_active;           /* True when injection stimulus is applied [2000, 3000) ms */
    volatile uint32_t nan_injected_to_core;   /* Count of actual NaN fed directly into e5_controller_step */

    /* Independent Control Core Call Counter (stops when faulted) */
    volatile uint32_t ctrl_call_count;        /* Actual number of e5_controller_step() executions */
    volatile uint8_t  drain_status;           /* 1 if queue drain succeeded, 0 if timeout */
    volatile uint8_t  summary_send_status;    /* 1 if summary UART transmit succeeded, 0 if failed */
    volatile uint32_t first_fault_ms;         /* Timestamp ms when first fault tripped */
    volatile uint8_t  first_fault_flags;      /* Fault flags captured at first trip */
    volatile uint8_t  first_fault_phase;      /* Phase captured at first trip */

    /* Control step sequence counter (for 1 kHz data integrity verification) */
    volatile uint32_t control_step_seq;
    volatile uint32_t active_control_steps;   /* Steps executed strictly during active trajectory (phases 1-5) */

    /* Trajectory and State Variables */
    volatile uint32_t elapsed_ms;
    volatile uint8_t phase;
    volatile uint8_t running;
    volatile uint8_t finished;
    volatile float reference_m;
    volatile float reference_velocity_mps;
    volatile float reference_acceleration_mps2;
    volatile float yG_m;
    volatile float e_sync_m;
    volatile uint8_t tracking_in_band;

    /* Board-side 1 kHz True Extreme Value Tracking (Active Trajectory) */
    volatile float max_sync_err_m;            /* Peak absolute synchronism error evaluated at 1 kHz */
    volatile float max_track_err_m;           /* Peak absolute tracking error evaluated at 1 kHz */

    /* Core Controller Structures */
    e5_controller_cfg_t cfg;
    e5_controller_state_t state;
    e5_controller_info_t info;
} experiment5_state_t;

extern volatile uint8_t g_exp5_abort;
extern experiment5_state_t g_exp5;

void motor_task(void const *argument);

#endif /* MOTOR_TASK_H */
