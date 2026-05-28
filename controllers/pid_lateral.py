
import math


class PIDLateralResult(object):
    """Debug output returned alongside the steering command."""

    def __init__(self, steer_cmd, nearest_x, nearest_y,
                 cross_track_error, heading_error_rad, steer_rad):
        self.steer_cmd         = steer_cmd          # CARLA steering command [-1, 1]
        self.nearest_x         = nearest_x          # nearest route point world x (m)
        self.nearest_y         = nearest_y          # nearest route point world y (m)
        self.cross_track_error = cross_track_error  # signed CTE (m)
        self.heading_error_rad = heading_error_rad  # heading error (rad)
        self.steer_rad         = steer_rad          # physical steering angle (rad)

    def __repr__(self):
        return (
            "PIDLateral(steer={:.3f}, nearest=({:.1f},{:.1f}), "
            "cte={:.3f}m, he={:.3f}rad)"
        ).format(
            self.steer_cmd,
            self.nearest_x, self.nearest_y,
            self.cross_track_error,
            self.heading_error_rad,
        )


class PIDLateralController(object):
    """
    Lateral PID controller that blends cross-track error (with derivative
    and integral terms) and heading error into a single steering command.

    Signed CTE convention (matches Stanley):
        cte > 0  → path is to the left  of the vehicle → steer left
        cte < 0  → path is to the right of the vehicle → steer right

    Args:
        Kp_cte:       Proportional gain on cross-track error.
        Ki_cte:       Integral gain on cross-track error.
        Kd_cte:       Derivative gain on cross-track error.
        K_heading:    Gain on heading error.
        max_steer_rad: Physical steering limit for normalisation to [-1, 1].
        integral_limit: Maximum absolute value of the integral accumulator
                        to prevent wind-up (metres·seconds).
    """

    def __init__(self,
                 Kp_cte=0.25,
                 Ki_cte=0.00,
                 Kd_cte=0.10,
                 K_heading=0.80,
                 max_steer_rad=0.7,
                 integral_limit=5.0):
        self.Kp_cte        = Kp_cte
        self.Ki_cte        = Ki_cte
        self.Kd_cte        = Kd_cte
        self.K_heading     = K_heading
        self.max_steer_rad = max_steer_rad
        self.integral_limit = integral_limit

        self._prev_cte    = 0.0   # CTE from the previous timestep
        self._integral    = 0.0   # accumulated integral of CTE
        self._first_step  = True  # skip derivative on the very first call

    def compute(self, state, route, dt=0.05):
        """
        Compute the PID steering command for the current vehicle state.

        Args:
            state:  VehicleState object (from core/vehicle_state.py).
            route:  List of RoutePoint objects (from core/route_planner.py).
            dt:     Elapsed time since the last call (seconds). Used for the
                    derivative and integral terms. Defaults to the nominal
                    loop period (LOOP_DT = 0.05 s).

        Returns:
            PIDLateralResult with steer_cmd and debug fields.
            Returns a zeroed result if route is empty.
        """
        if not route:
            return self._zero_result()

        # Find the nearest route waypoint to the vehicle centre.
        nearest_idx, nearest = self._nearest_waypoint(state, route)

        # ── 1. Signed cross-track error
        # Vector from the vehicle to the nearest waypoint, projected onto
        # the route's left-perpendicular unit vector.
        #
        # Left perpendicular of route tangent: (-sin(yaw), cos(yaw))
        #
        # cte > 0 → path lies to the left  of the vehicle
        # cte < 0 → path lies to the right of the vehicle
        dx = nearest.x - state.x
        dy = nearest.y - state.y
        cte = -math.sin(nearest.yaw_rad) * dx + math.cos(nearest.yaw_rad) * dy

        # ── 2. Heading error ─────────────────────────────────────────────────
        he = nearest.yaw_rad - state.yaw_rad
        he = _normalise(he)

        # ── 3. PID terms ─────────────────────────────────────────────────────
        # Derivative: rate of change of CTE. Zero on the first call to avoid
        # a large spike from an uninitialised _prev_cte.
        if self._first_step:
            dcte = 0.0
            self._first_step = False
        else:
            dcte = (cte - self._prev_cte) / dt if dt > 0.0 else 0.0

        # Integral with anti-wind-up clamp.
        self._integral += cte * dt
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit, self._integral))

        self._prev_cte = cte

        # ── 4. Steering law ──────────────────────────────────────────────────
        steer_rad = (self.Kp_cte    * cte
                     + self.Kd_cte  * dcte
                     + self.Ki_cte  * self._integral
                     + self.K_heading * he)

        steer_rad = max(-self.max_steer_rad, min(self.max_steer_rad, steer_rad))

        # Convert to CARLA normalised command and clip.
        steer_cmd = steer_rad / self.max_steer_rad
        steer_cmd = max(-1.0, min(1.0, steer_cmd))

        return PIDLateralResult(
            steer_cmd=steer_cmd,
            nearest_x=nearest.x,
            nearest_y=nearest.y,
            cross_track_error=cte,
            heading_error_rad=he,
            steer_rad=steer_rad,
        )

    def reset(self):
        """Clear integral accumulator and derivative memory."""
        self._prev_cte   = 0.0
        self._integral   = 0.0
        self._first_step = True

   
    def _nearest_waypoint(self, state, route):
        """Return (index, RoutePoint) of the waypoint closest to the vehicle."""
        best_idx  = 0
        best_dist = float("inf")
        for i, wp in enumerate(route):
            dx = wp.x - state.x
            dy = wp.y - state.y
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_idx  = i
        return best_idx, route[best_idx]

    def _zero_result(self):
        return PIDLateralResult(
            steer_cmd=0.0,
            nearest_x=0.0,
            nearest_y=0.0,
            cross_track_error=0.0,
            heading_error_rad=0.0,
            steer_rad=0.0,
        )


def _normalise(angle_rad):
    """Wrap angle to (-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
