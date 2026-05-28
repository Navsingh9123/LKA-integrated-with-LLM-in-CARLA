import math


# Candidate steering angles (radians) evaluated at every control step.
# Symmetric set covering roughly ±0.35 rad of the ±0.7 rad physical limit.

_CANDIDATES = [-0.35, -0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25, 0.35]


def _normalise(angle_rad):
    """Wrap angle to (-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


# Result container


class MPCResult(object):
    """Debug output returned alongside the steering command."""

    def __init__(self, steer_cmd, selected_steer_rad, predicted_cost,
                 nearest_x, nearest_y, heading_error_rad, lateral_error_m):
        self.steer_cmd          = steer_cmd           # CARLA steering command [-1, 1]
        self.selected_steer_rad = selected_steer_rad  # winning candidate (rad)
        self.predicted_cost     = predicted_cost      # total accumulated cost of winner
        self.nearest_x          = nearest_x           # nearest route point world x (m)
        self.nearest_y          = nearest_y           # nearest route point world y (m)
        self.heading_error_rad  = heading_error_rad   # heading error at current state
        self.lateral_error_m    = lateral_error_m     # lateral error at current state

    def __repr__(self):
        return (
            "MPC(steer={:.3f}, sel={:.3f}rad, cost={:.4f}, "
            "nearest=({:.1f},{:.1f}), he={:.3f}rad)"
        ).format(
            self.steer_cmd,
            self.selected_steer_rad,
            self.predicted_cost,
            self.nearest_x, self.nearest_y,
            self.heading_error_rad,
        )



# Controller


class MPCController(object):
    """
    Sampled kinematic MPC-style lateral controller.

    For every CARLA control step:
      1. Enumerate _CANDIDATES (9 constant steering angles).
      2. For each candidate, step the kinematic bicycle model forward
         for horizon_steps * dt seconds.
      3. Score with a quadratic cost on lateral error, heading error,
         steering magnitude and steering change.
      4. Apply the lowest-cost candidate.

    Args:
        wheelbase:      Effective vehicle wheelbase (m). Tesla Model 3 ≈ 2.875 m.
        horizon_steps:  Number of prediction steps per candidate evaluation.
        dt:             Prediction timestep (s).  Independent of CARLA LOOP_DT;
                        the horizon covers horizon_steps * dt seconds ahead.
        max_steer_rad:  Physical steering limit used to normalise to [-1, 1].
        q_lateral:      Quadratic weight on lateral error.
        q_heading:      Quadratic weight on heading error.
        r_steer:        Quadratic weight on steering magnitude (effort penalty).
        r_steer_change: Quadratic weight on steering change from previous step
                        (smoothness penalty).
    """

    def __init__(self,
                 wheelbase=2.875,
                 horizon_steps=10,
                 dt=0.10,
                 max_steer_rad=0.7,
                 q_lateral=1.0,
                 q_heading=0.5,
                 r_steer=0.1,
                 r_steer_change=0.2):
        self.wheelbase      = wheelbase
        self.horizon_steps  = horizon_steps
        self.dt             = dt
        self.max_steer_rad  = max_steer_rad
        self.q_lateral      = q_lateral
        self.q_heading      = q_heading
        self.r_steer        = r_steer
        self.r_steer_change = r_steer_change

        # Steering applied at the previous control step; used for the
        # change-penalty term to discourage abrupt reversals.
        self._prev_steer_rad = 0.0


    # Public interface


    def compute(self, state, route):
        """
        Evaluate all candidates and return the lowest-cost steering command.

        Args:
            state:  VehicleState object (from core/vehicle_state.py).
            route:  List of RoutePoint objects (from core/route_planner.py).

        Returns:
            MPCResult with steer_cmd and debug fields.
            Returns a zeroed result if route is empty.
        """
        if not route:
            return self._zero_result()

        best_steer = 0.0
        best_cost  = float("inf")

        for candidate in _CANDIDATES:
            cost = self._evaluate_candidate(state, route, candidate)
            if cost < best_cost:
                best_cost  = cost
                best_steer = candidate

        # Persist selected steering for the next step's change-penalty term.
        self._prev_steer_rad = best_steer

        steer_cmd = max(-1.0, min(1.0, best_steer / self.max_steer_rad))

        # Compute current-state metrics for logging (not used in optimisation).
        nearest = self._nearest_waypoint(state.x, state.y, route)
        lat_err = math.sqrt(
            (state.x - nearest.x) ** 2 + (state.y - nearest.y) ** 2
        )
        he = _normalise(nearest.yaw_rad - state.yaw_rad)

        return MPCResult(
            steer_cmd=steer_cmd,
            selected_steer_rad=best_steer,
            predicted_cost=best_cost,
            nearest_x=nearest.x,
            nearest_y=nearest.y,
            heading_error_rad=he,
            lateral_error_m=lat_err,
        )


    # Internal helpers


    def _evaluate_candidate(self, state, route, steer_rad):
        """
        Simulate `steer_rad` held constant over the prediction horizon and
        return the accumulated cost.

        The kinematic bicycle model is stepped forward horizon_steps times.
        At each step the nearest route point is found and errors accumulated.
        The control-effort and change-penalty terms are also added per step.
        """
        x   = state.x
        y   = state.y
        yaw = state.yaw_rad
        v   = state.speed_ms

        # Precompute tan(steer_rad) — constant over the horizon.
        tan_steer = math.tan(steer_rad)

        cost = 0.0
        for _ in range(self.horizon_steps):
            # Kinematic bicycle model step.
            x   += v * math.cos(yaw) * self.dt
            y   += v * math.sin(yaw) * self.dt
            yaw += (v / self.wheelbase) * tan_steer * self.dt

            # Error relative to the nearest route point at the predicted position.
            nearest  = self._nearest_waypoint(x, y, route)
            lat_err  = math.sqrt((x - nearest.x) ** 2 + (y - nearest.y) ** 2)
            head_err = _normalise(nearest.yaw_rad - yaw)

            cost += (self.q_lateral      * lat_err  ** 2
                   + self.q_heading      * head_err ** 2
                   + self.r_steer        * steer_rad ** 2
                   + self.r_steer_change * (steer_rad - self._prev_steer_rad) ** 2)

        return cost

    def _nearest_waypoint(self, x, y, route):
        """Return the RoutePoint in route closest to world position (x, y)."""
        best_wp = route[0]
        best_sq = float("inf")
        for wp in route:
            sq = (wp.x - x) ** 2 + (wp.y - y) ** 2
            if sq < best_sq:
                best_sq = sq
                best_wp = wp
        return best_wp

    def _zero_result(self):
        return MPCResult(
            steer_cmd=0.0,
            selected_steer_rad=0.0,
            predicted_cost=0.0,
            nearest_x=0.0,
            nearest_y=0.0,
            heading_error_rad=0.0,
            lateral_error_m=0.0,
        )
