"""
Generates a lane-centre waypoint route from a CARLA map.
"""

import math


class RoutePoint(object):
    """A single point along the planned route."""

    def __init__(self, x, y, yaw_deg, yaw_rad, location):
        self.x = x                  # world x (metres)
        self.y = y                  # world y (metres)
        self.yaw_deg = yaw_deg      # heading in degrees (raw from CARLA)
        self.yaw_rad = yaw_rad      # heading in radians, normalised to (-pi, pi]
        self.location = location    # original carla.Location, kept for debugging

    def __repr__(self):
        return "RoutePoint(x={:.2f}, y={:.2f}, yaw={:.2f}deg)".format(
            self.x, self.y, self.yaw_deg
        )


def generate_route(world, start_location, distance=80.0, spacing=2.0):
    
    carla_map = world.get_map()

    # Snap start_location to the nearest drivable road waypoint.
    # project_to_road=True ensures we land on the lane centre line.
    current_wp = carla_map.get_waypoint(
        start_location,
        project_to_road=True,
        lane_type=_get_driving_lane_type(),
    )

    if current_wp is None:
        print("[WARN] route_planner: no road waypoint found near start location.")
        return []

    route = []
    accumulated = 0.0

    while accumulated < distance:
        loc = current_wp.transform.location
        yaw_deg = current_wp.transform.rotation.yaw
        yaw_rad = math.atan2(math.sin(math.radians(yaw_deg)),
                             math.cos(math.radians(yaw_deg)))

        route.append(RoutePoint(
            x=loc.x,
            y=loc.y,
            yaw_deg=yaw_deg,
            yaw_rad=yaw_rad,
            location=loc,
        ))

        # Advance by spacing metres along the lane centre line.
        next_wps = current_wp.next(spacing)
        if not next_wps:
            # End of road / junction with no continuation.
            break

        # Always pick the first option (straight-ahead heuristic).
        current_wp = next_wps[0]
        accumulated += spacing

    return route


def print_route_summary(route, spacing=2.0):
    if not route:
        print("[ROUTE] Empty route — nothing to summarise.")
        return

    n = len(route)
    first = route[0]
    last = route[-1]
    approx_length = (n - 1) * spacing

    print("[ROUTE] Waypoints : {}".format(n))
    print("[ROUTE] First     : x={:.2f}m, y={:.2f}m".format(first.x, first.y))
    print("[ROUTE] Last      : x={:.2f}m, y={:.2f}m".format(last.x, last.y))
    print("[ROUTE] Approx length: {:.1f}m".format(approx_length))


# Internal helpers

def _get_driving_lane_type():
    """Return the carla.LaneType flag for drivable lanes."""
    try:
        import carla
        return carla.LaneType.Driving
    except Exception:
        return 1
