"""
vehicle_state.py

"""

import math


class VehicleState(object):

    def __init__(self, x, y, yaw_rad, yaw_deg, speed_ms, speed_kmh, steer):
        # Position in world frame
        self.x = x
        self.y = y

        # Heading angle
        self.yaw_rad = yaw_rad
        self.yaw_deg = yaw_deg

        # Longitudinal speed (planar, ignoring vertical component)
        self.speed_ms = speed_ms      # m/s
        self.speed_kmh = speed_kmh    # km/h

        # Latest steering command applied to the vehicle (-1.0 to 1.0)
        # None if the vehicle control has not been read yet.
        self.steer = steer

    def __repr__(self):
        return (
            "VehicleState("
            "x={:.2f}m, y={:.2f}m, "
            "yaw={:.2f}deg, "
            "speed={:.2f}m/s ({:.1f}km/h), "
            "steer={}"
            ")".format(
                self.x, self.y,
                self.yaw_deg,
                self.speed_ms, self.speed_kmh,
                "{:.3f}".format(self.steer) if self.steer is not None else "N/A",
            )
        )


def get_vehicle_state(vehicle):

    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()

    # Position
    x = transform.location.x
    y = transform.location.y

    # Yaw: CARLA gives degrees; convert to radians for controller use.
    yaw_deg = transform.rotation.yaw
    yaw_rad = math.radians(yaw_deg)
    # Normalise to (-pi, pi]
    yaw_rad = math.atan2(math.sin(yaw_rad), math.cos(yaw_rad))

    # Planar speed (ignore vertical velocity component)
    speed_ms = math.sqrt(velocity.x ** 2 + velocity.y ** 2)
    speed_kmh = speed_ms * 3.6

    # Read last steering command from vehicle control if available
    try:
        steer = vehicle.get_control().steer
    except Exception:
        steer = None

    return VehicleState(
        x=x,
        y=y,
        yaw_rad=yaw_rad,
        yaw_deg=yaw_deg,
        speed_ms=speed_ms,
        speed_kmh=speed_kmh,
        steer=steer,
    )
