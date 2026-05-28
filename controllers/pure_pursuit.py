import math


class PurePursuitResult(object):

    def __init__(self, steer_cmd, target_x, target_y,
                 lookahead_distance, alpha_rad, steer_rad):
        self.steer_cmd = steer_cmd              # CARLA steering command [-1, 1]
        self.target_x = target_x               # target waypoint world x (m)
        self.target_y = target_y               # target waypoint world y (m)
        self.lookahead_distance = lookahead_distance  # dynamic Ld (m)
        self.alpha_rad = alpha_rad             # heading error to target (rad)
        self.steer_rad = steer_rad             # geometric steering angle (rad)

    def __repr__(self):
        return (
            "PP(steer={:.3f}, target=({:.1f},{:.1f}), "
            "Ld={:.2f}m, alpha={:.3f}rad)"
        ).format(
            self.steer_cmd,
            self.target_x, self.target_y,
            self.lookahead_distance,
            self.alpha_rad,
        )


class PurePursuitController(object):
    """
    pure-pursuit lateral controller.
    Args:
        wheelbase:       Effective vehicle wheelbase (metres). Tesla Model 3 ≈ 2.875 m.
        lookahead_base:  Minimum lookahead distance at zero speed (metres).
        lookahead_gain:  Speed-proportional lookahead gain (s).  Ld increases with speed.
        max_steer_rad:   Physical steering limit used to normalise to [-1, 1] for CARLA.
    """

    def __init__(self,
                 wheelbase=2.875,
                 lookahead_base=6.0,
                 lookahead_gain=0.5,
                 max_steer_rad=0.7):
        self.wheelbase = wheelbase
        self.lookahead_base = lookahead_base
        self.lookahead_gain = lookahead_gain
        self.max_steer_rad = max_steer_rad

        # Index of the last target waypoint — avoids re-scanning from index 0
        # every step, which would cause backward jumps near the route start.
        self._last_target_idx = 0

    def compute(self, state, route):
        """
        Compute the steering command for the current vehicle state.
        Returns:
            PurePursuitResult with steer_cmd and debug fields.
            Returns a zeroed result if the route is exhausted.
        """
        if not route:
            return self._zero_result()

        # Dynamic lookahead distance grows with speed to remain stable at high speed.
        Ld = self.lookahead_base + self.lookahead_gain * state.speed_ms

        # Find the first waypoint ahead of the vehicle that is at least Ld away.
        target_idx = self._find_target_waypoint(state, route, Ld)
        target = route[target_idx]

        # Vector from vehicle to target point in world frame.
        dx = target.x - state.x
        dy = target.y - state.y

        # Transform target into vehicle body frame.
        # alpha = angle between vehicle heading and direction to target.
        angle_to_target = math.atan2(dy, dx)
        alpha = angle_to_target - state.yaw_rad

        # Normalise alpha to [-pi, pi] to avoid wrap-around discontinuities.
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        # Pure pursuit steering formula (bicycle model).
        steer_rad = math.atan2(2.0 * self.wheelbase * math.sin(alpha), Ld)

        # Convert to CARLA's normalised steering command and hard-clip.
        steer_cmd = steer_rad / self.max_steer_rad
        steer_cmd = max(-1.0, min(1.0, steer_cmd))

        return PurePursuitResult(
            steer_cmd=steer_cmd,
            target_x=target.x,
            target_y=target.y,
            lookahead_distance=Ld,
            alpha_rad=alpha,
            steer_rad=steer_rad,
        )


    # Internal helpers


    def _find_target_waypoint(self, state, route, Ld):
      
        start = max(0, self._last_target_idx)

        for i in range(start, len(route)):
            wp = route[i]
            dist = math.sqrt((wp.x - state.x) ** 2 + (wp.y - state.y) ** 2)
            if dist >= Ld:
                self._last_target_idx = i
                return i

        # Route exhausted — hold on the final waypoint.
        self._last_target_idx = len(route) - 1
        return self._last_target_idx

    def _zero_result(self):
        return PurePursuitResult(
            steer_cmd=0.0,
            target_x=0.0,
            target_y=0.0,
            lookahead_distance=self.lookahead_base,
            alpha_rad=0.0,
            steer_rad=0.0,
        )
