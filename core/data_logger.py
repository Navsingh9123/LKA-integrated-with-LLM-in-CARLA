"""
Timestamped CSV logger for LKA controller run data.
"""

import csv
import os
import datetime

CSV_HEADERS = [
    # Core state
    "timestamp_s",
    "x_m",
    "y_m",
    "yaw_deg",
    "speed_ms",
    "speed_kmh",
    "throttle",
    "steer_cmd",
    "lateral_error_m",
    "heading_error_rad",
    #  Controller debug
    "target_x",
    "target_y",
    "lookahead_m",           # pure_pursuit / ai_pure_pursuit
    "alpha_rad",             # pure_pursuit / ai_pure_pursuit
    "predicted_cost",        # mpc / ai_mpc
    "selected_steer_rad",    # mpc / ai_mpc
    # AI/LLM
    "latest_user_request",
    "llm_status",
    "llm_reason",
    "active_target_speed_kmh",
    "active_lookahead_base",     # ai_pure_pursuit
    "active_lookahead_gain",     # ai_pure_pursuit
    "active_stanley_k",          # ai_stanley
    "active_softening_speed",    # ai_stanley
    "active_Kp_cte",             # ai_pid
    "active_Kd_cte",             # ai_pid
    "active_K_heading",          # ai_pid
    "active_q_lateral",          # ai_mpc
    "active_q_heading",          # ai_mpc
    "active_r_steer",            # ai_mpc
    "active_r_steer_change",     # ai_mpc
]

# Directory relative to the project root where CSV files are saved
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "logs")


class DataLogger(object):
    """
    Writes one CSV row per control loop iteration.

    """

    def __init__(self, controller_name="controller"):

        os.makedirs(LOG_DIR, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "{}_{}.csv".format(controller_name, timestamp)
        self.filepath = os.path.join(LOG_DIR, filename)

        self._file = open(self.filepath, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_HEADERS,
                                      extrasaction="ignore", restval="")
        self._writer.writeheader()
        self._row_count = 0

    def log(self, row):

        self._writer.writerow(row)
        self._row_count += 1

    def close(self)
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    @property
    def row_count(self):
        """Number of rows written so far."""
        return self._row_count
