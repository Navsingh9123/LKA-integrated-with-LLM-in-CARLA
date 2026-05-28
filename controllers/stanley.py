import math


class StanleyResult(object):
    """Debug output returned alongside the steering command."""

    def __init__(self, steer_cmd, nearest_x, nearest_y,
                 heading_error_rad, cross_track_error, steer_rad):
        self.steer_cmd         = steer_cmd          # CARLA steering command [-1, 1]
        self.nearest_x         = nearest_x          # nearest route point world x (m)
        self.nearest_y         = nearest_y          # nearest route point world y (m)
        self.heading_error_rad = heading_error_rad  # heading error (rad)
        self.cross_track_error = cross_track_error  # signed CTE (m), + = left of path
        self.steer_rad         = steer_rad          # physical steering angle (rad)

    def __repr__(self):
        return (
            "Stanley(steer={:.3f}, nearest=({:.1f},{:.1f}), "
            "he={:.3f}rad, cte={:.3f}m)"
        ).format(
            self.steer_cmd,
            self.nearest_x, self.nearest_y,
            self.heading_error_rad,
            self.cross_track_error,
        )


class StanleyController(object):
    """
    Stanley lateral controller using front-axle error formulation.

    The steering command is:
        steer_rad = heading_error + atan2(k * cte, speed + softening_speed)

    where cte is the signed cross-track error at the front axle position.

    Args:
        wheelbase:       Distance between front and rear axles (m).
                         Tesla Model 3 ≈ 2.875 m.
        k:               Cross-track error gain. Higher values produce
                         more aggressive lateral correction.
        softening_speed: Added to speed in the denominator to prevent
                         division-by-zero and reduce gain at low speed (m/s).
        max_steer_rad:   Physical steering limit for normalisation to [-1, 1].
    """

    def __init__(self,
                 wheelbase=2.875,
                 k=1.0,
                 softening_speed=1.0,
                 max_steer_rad=0.7):
        self.wheelbase       = wheelbase
        self.k               = k
        self.softening_speed = softening_speed
        self.max_steer_rad   = max_steer_rad

    def compute(self, state, route):
        """
        Compute the Stanley steering command for the current vehicle state.

        Args:
            state:  VehicleState object (from core/vehicle_state.py).
            route:  List of RoutePoint objects (from core/route_planner.py).

        Returns:
            StanleyResult with steer_cmd and debug fields.
            Returns a zeroed result if route is empty.
        """
        if not route:
            return self._zero_result()

        # Project vehicle position to its front axle.
        front_x = state.x + self.wheelbase * math.cos(state.yaw_rad)
        front_y = state.y + self.wheelbase * math.sin(state.yaw_rad)

        # Find the route waypoint nearest to the front axle.
        nearest_idx, nearest = self._nearest_to_front(front_x, front_y, route)

        # ── 1. Heading error ────────────────────────────────────────────────
        # Difference between route heading and vehicle heading.
        # Positive means the route is turning left relative to the vehicle.
        he = nearest.yaw_rad - state.yaw_rad
        he = _normalise(he)

        # ── 2. Cross-track error (signed) ───────────────────────────────────
        # Project the waypoint-to-front-axle vector onto the route's
        # left-perpendicular unit vector to obtain a signed distance.
        #
        # Route tangent direction: (cos(route_yaw), sin(route_yaw))
        # Left perpendicular:      (-sin(route_yaw), cos(route_yaw))
        #
        # cte > 0  → reference path lies to the LEFT  of the front axle → steer left
        # cte < 0  → reference path lies to the RIGHT of the front axle → steer right
        dx  = nearest.x - front_x
        dy  = nearest.y - front_y
        cte = -math.sin(nearest.yaw_rad) * dx + math.cos(nearest.yaw_rad) * dy

        # ── 3. Stanley steering law ─────────────────────────────────────────
        effective_speed = state.speed_ms + self.softening_speed
        steer_rad = he + math.atan2(self.k * cte, effective_speed)
        steer_rad = max(-self.max_steer_rad, min(self.max_steer_rad, steer_rad))

        # Convert to CARLA normalised command and clip.
        steer_cmd = steer_rad / self.max_steer_rad
        steer_cmd = max(-1.0, min(1.0, steer_cmd))

        return StanleyResult(
            steer_cmd=steer_cmd,
            nearest_x=nearest.x,
            nearest_y=nearest.y,
            heading_error_rad=he,
            cross_track_error=cte,
            steer_rad=steer_rad,
        )


    # Internal helpers

    def _nearest_to_front(self, front_x, front_y, route):
        """Return (index, RoutePoint) of the waypoint closest to the front axle."""
        best_idx  = 0
        best_dist = float("inf")
        for i, wp in enumerate(route):
            dx = wp.x - front_x
            dy = wp.y - front_y
            dist = dx * dx + dy * dy   # squared distance — no need for sqrt
            if dist < best_dist:
                best_dist = dist
                best_idx  = i
        return best_idx, route[best_idx]

    def _zero_result(self):
        return StanleyResult(
            steer_cmd=0.0,
            nearest_x=0.0,
            nearest_y=0.0,
            heading_error_rad=0.0,
            cross_track_error=0.0,
            steer_rad=0.0,
        )


def _normalise(angle_rad):
    """Wrap angle to (-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
