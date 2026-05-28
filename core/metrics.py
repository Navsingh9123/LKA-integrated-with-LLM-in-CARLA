"""
metrics.py

Lateral and heading error computation, plus end-of-run summary statistics
"""

import math



# Per-timestep helpers


def normalise_angle(angle_rad):

    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def nearest_route_point(state, route):

    best_idx = 0
    best_dist_sq = float("inf")

    for i, wp in enumerate(route):
        dx = wp.x - state.x
        dy = wp.y - state.y
        dist_sq = dx * dx + dy * dy
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_idx = i

    return best_idx, route[best_idx]


def lateral_error(state, route):

    if not route:
        return 0.0
    if len(route) == 1:
        dx = route[0].x - state.x
        dy = route[0].y - state.y
        return math.sqrt(dx * dx + dy * dy)

    min_dist_sq = float("inf")

    for i in range(len(route) - 1):
        ax, ay = route[i].x,     route[i].y
        bx, by = route[i + 1].x, route[i + 1].y

        abx = bx - ax
        aby = by - ay
        seg_len_sq = abx * abx + aby * aby

        if seg_len_sq < 1e-12:
            dx = ax - state.x
            dy = ay - state.y
        else:
            # Parameter of the closest point on the segment, clamped to [0, 1].
            t = ((state.x - ax) * abx + (state.y - ay) * aby) / seg_len_sq
            t = max(0.0, min(1.0, t))
            px = ax + t * abx
            py = ay + t * aby
            dx = px - state.x
            dy = py - state.y

        dist_sq = dx * dx + dy * dy
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq

    return math.sqrt(min_dist_sq)


def heading_error(state, route):

    _, nearest = nearest_route_point(state, route)
    error_rad = nearest.yaw_rad - state.yaw_rad
    return normalise_angle(error_rad)


# End-of-run summary

def calculate_summary_metrics(log_rows):

    if not log_rows:
        return {}

    n = len(log_rows)

    lat_errors  = [r["lateral_error_m"]   for r in log_rows]
    head_errors = [r["heading_error_rad"]  for r in log_rows]
    steers      = [r["steer_cmd"]          for r in log_rows]
    speeds      = [r["speed_kmh"]          for r in log_rows]
    times       = [r["timestamp_s"]        for r in log_rows]

    # Lateral error statistics
    rms_lat   = math.sqrt(sum(e ** 2 for e in lat_errors) / n)
    mean_lat  = sum(abs(e) for e in lat_errors) / n
    max_lat   = max(abs(e) for e in lat_errors)

    # Heading error statistics (convert rad -> deg for readability)
    head_deg  = [math.degrees(e) for e in head_errors]
    rms_head  = math.sqrt(sum(e ** 2 for e in head_deg) / n)
    mean_head = sum(abs(e) for e in head_deg) / n

    # Steering command statistics
    mean_steer = sum(abs(s) for s in steers) / n
    max_steer  = max(abs(s) for s in steers)

    # Speed statistics
    mean_speed = sum(speeds) / n
    max_speed  = max(speeds)

    # Duration
    duration = times[-1] - times[0] if len(times) > 1 else 0.0

    return {
        "rms_lateral_error_m":       rms_lat,
        "mean_abs_lateral_error_m":  mean_lat,
        "max_abs_lateral_error_m":   max_lat,
        "rms_heading_error_deg":     rms_head,
        "mean_abs_heading_error_deg": mean_head,
        "mean_abs_steer_cmd":        mean_steer,
        "max_abs_steer_cmd":         max_steer,
        "mean_speed_kmh":            mean_speed,
        "max_speed_kmh":             max_speed,
        "total_duration_s":          duration,
    }


def print_summary_table(metrics, controller_name="Controller"):
  
    if not metrics:
        print("[METRICS] No data to summarise.")
        return

    sep = "-" * 46
    print("\n" + sep)
    print("  Performance Summary — {}".format(controller_name))
    print(sep)
    print("  {:<36} {:>7}".format("RMS lateral error",
          "{:.4f} m".format(metrics["rms_lateral_error_m"])))
    print("  {:<36} {:>7}".format("Mean abs lateral error",
          "{:.4f} m".format(metrics["mean_abs_lateral_error_m"])))
    print("  {:<36} {:>7}".format("Max abs lateral error",
          "{:.4f} m".format(metrics["max_abs_lateral_error_m"])))
    print("  {:<36} {:>7}".format("RMS heading error",
          "{:.4f} deg".format(metrics["rms_heading_error_deg"])))
    print("  {:<36} {:>7}".format("Mean abs heading error",
          "{:.4f} deg".format(metrics["mean_abs_heading_error_deg"])))
    print("  {:<36} {:>7}".format("Mean abs steer command",
          "{:.4f}".format(metrics["mean_abs_steer_cmd"])))
    print("  {:<36} {:>7}".format("Max abs steer command",
          "{:.4f}".format(metrics["max_abs_steer_cmd"])))
    print("  {:<36} {:>7}".format("Mean speed",
          "{:.2f} km/h".format(metrics["mean_speed_kmh"])))
    print("  {:<36} {:>7}".format("Max speed",
          "{:.2f} km/h".format(metrics["max_speed_kmh"])))
    print("  {:<36} {:>7}".format("Total duration",
          "{:.1f} s".format(metrics["total_duration_s"])))
    print(sep + "\n")
