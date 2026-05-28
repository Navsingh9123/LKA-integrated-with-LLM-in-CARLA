import math
import time
import carla

from core.vehicle_state import get_vehicle_state
from core.route_planner import generate_route, print_route_summary
from core.metrics import lateral_error, heading_error, calculate_summary_metrics, print_summary_table
from core.data_logger import DataLogger
from controllers.pure_pursuit import PurePursuitController
from controllers.stanley import StanleyController
from controllers.pid_lateral import PIDLateralController
from controllers.mpc_controller import MPCController
import json
import os

from llm_supervisor.live_command_supervisor import LiveCommandSupervisor
from llm_supervisor.live_llm_supervisor import LiveLLMSupervisor

# Select controller
#   Baseline      : "pure_pursuit" | "stanley" | "pid" | "mpc"
#   LLM pre-run   : "llm_pure_pursuit"
#   Rule-based live: "interactive_pure_pursuit"
#   AI-supervised : "ai_pure_pursuit" | "ai_stanley" | "ai_pid" | "ai_mpc"
#   LLM live      : "llm_interactive_pure_pursuit" | "llm_interactive_stanley"
#                   "llm_interactive_pid"           | "llm_interactive_mpc"
CONTROLLER = "ai_mpc"

LLM_PARAMS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "llm_supervisor", "llm_params.json"
)


# Experiment parameters
TARGET_SPEED_KMH = 15.0
TARGET_SPEED_MS  = TARGET_SPEED_KMH / 3.6

LOOP_DT          = 0.05    # seconds between control updates 20 hz
REAL_TIME_PACING = True    # sleep each step so sim time = wall-clock time

NORMAL_RUN_DURATION_S   = 120.0   # run duration
NORMAL_ROUTE_DISTANCE_M = 700.0   # route length

INTERACTIVE_RUN_DURATION_S   = 120.0   # run duration for interactive/AI modes
INTERACTIVE_ROUTE_DISTANCE_M = 700.0   # route length for interactive/AI modes

ROUTE_SPACING_M = 2.0   # distance between waypoints


# Visualisation settings
ROUTE_DRAW_EVERY_N   = 2
ROUTE_POINT_SIZE     = 0.12
ROUTE_POINT_COLOR    = carla.Color(r=0,   g=200, b=255)   # cyan

# Vehicle trajectory
TRAJ_POINT_SIZE      = 0.08
TRAJ_POINT_COLOR     = carla.Color(r=255, g=140, b=0)     # orange
TRAJ_POINT_LIFETIME  = 15.0

# camera placement
CAM_BEHIND_M = 8.0   # metres behind the vehicle
CAM_ABOVE_M  = 5.0   # metres above the vehicle
CAM_PITCH    = -20.0  # degrees (negative = looking slightly downward)

# Visualisation helpers
def draw_route(world, route, lifetime):
    """
    Draw route waypoints in the CARLA world as cyan dots
    """
    debug = world.debug
    for i, wp in enumerate(route):
        if i % ROUTE_DRAW_EVERY_N != 0:
            continue
        location = carla.Location(x=wp.x, y=wp.y, z=wp.location.z + 0.5)
        debug.draw_point(
            location,
            size=ROUTE_POINT_SIZE,
            color=ROUTE_POINT_COLOR,
            life_time=lifetime,
        )
    print("[VIS] Route drawn: {} markers".format(len(route) // ROUTE_DRAW_EVERY_N))


def draw_vehicle_position(world, state):
    """
    Draw a small orange dot at the vehicle's current position.
    """
    location = carla.Location(x=state.x, y=state.y, z=0.3)
    world.debug.draw_point(
        location,
        size=TRAJ_POINT_SIZE,
        color=TRAJ_POINT_COLOR,
        life_time=TRAJ_POINT_LIFETIME,
    )


def update_spectator_camera(world, vehicle):
    """
    Move the CARLA spectator to follow the ego vehicle from
    behind and above.
    """
    transform   = vehicle.get_transform()
    forward_vec = transform.get_forward_vector()
    vehicle_yaw = transform.rotation.yaw

    # Offset position
    cam_location = carla.Location(
        x=transform.location.x - CAM_BEHIND_M * forward_vec.x,
        y=transform.location.y - CAM_BEHIND_M * forward_vec.y,
        z=transform.location.z + CAM_ABOVE_M,
    )

    cam_rotation = carla.Rotation(
        pitch=CAM_PITCH,
        yaw=vehicle_yaw,
        roll=0.0,
    )

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(cam_location, cam_rotation))


# Controller factory and shared loop helpers

def make_controller(name):
    """Instantiate and return the selected controller."""
    if name == "pure_pursuit":
        return PurePursuitController(
            wheelbase=2.875,
            lookahead_base=6.0,
            lookahead_gain=0.5,
            max_steer_rad=0.7,
        )
    elif name == "stanley":
        return StanleyController(
            wheelbase=2.875,
            k=1.0,
            softening_speed=1.0,
            max_steer_rad=0.7,
        )
    elif name == "pid":
        return PIDLateralController(
            Kp_cte=0.25,
            Ki_cte=0.00,
            Kd_cte=0.10,
            K_heading=0.80,
            max_steer_rad=0.7,
            integral_limit=5.0,
        )
    elif name == "llm_pure_pursuit":
        # Parameters are filled in by the LLM before this factory is called.
        # make_controller() is not used for llm_pure_pursuit; see main().
        raise RuntimeError("llm_pure_pursuit must be built via make_llm_controller().")
    elif name == "interactive_pure_pursuit":
        return PurePursuitController(
            wheelbase=2.875,
            lookahead_base=6.0,
            lookahead_gain=0.5,
            max_steer_rad=0.7,
        )
    elif name == "mpc":
        return MPCController(
            wheelbase=2.875,
            horizon_steps=10,
            dt=0.10,
            max_steer_rad=0.7,
            q_lateral=1.0,
            q_heading=0.5,
            r_steer=0.1,
            r_steer_change=0.2,
        )
    elif name.startswith("ai_"):
        # AI-supervised mode: build the underlying base controller unchanged.
        return make_controller(name[3:])
    elif name.startswith("llm_interactive_"):
        # LLM interactive mode: build the underlying base controller unchanged.
        return make_controller(name[len("llm_interactive_"):])
    else:
        raise ValueError(
            "Unknown controller: '{}'. "
            "Baselines: 'pure_pursuit' | 'stanley' | 'pid' | 'mpc'. "
            "LLM pre-run: 'llm_pure_pursuit'. "
            "Rule-based live: 'interactive_pure_pursuit'. "
            "AI-supervised: 'ai_pure_pursuit' | 'ai_stanley' | 'ai_pid' | 'ai_mpc'. "
            "LLM live: 'llm_interactive_pure_pursuit' | 'llm_interactive_stanley' | "
            "'llm_interactive_pid' | 'llm_interactive_mpc'.".format(name)
        )


def make_llm_controller():

    if not os.path.isfile(LLM_PARAMS_PATH):
        raise RuntimeError(
            "FALLBACK - llm_params.json missing, not LLM-supervised.\n"
            "Run ollama_supervisor.py before starting CARLA to generate it."
        )

    with open(LLM_PARAMS_PATH, "r") as f:
        try:
            params = json.load(f)
        except ValueError as e:
            raise RuntimeError(
                "FALLBACK - llm_params.json is invalid JSON: {}".format(e)
            )

    print("[LLM] Loaded pre-run Ollama parameters from llm_params.json")
    print("[LLM] target_speed_kmh = {:.1f}".format(params["target_speed_kmh"]))
    print("[LLM] lookahead_base   = {:.2f}".format(params["lookahead_base"]))
    print("[LLM] lookahead_gain   = {:.2f}".format(params["lookahead_gain"]))
    print("[LLM] reason           = {}".format(params.get("reason", "")))

    controller = PurePursuitController(
        wheelbase=2.875,
        lookahead_base=float(params["lookahead_base"]),
        lookahead_gain=float(params["lookahead_gain"]),
        max_steer_rad=0.7,
    )
    target_speed_kmh = float(params["target_speed_kmh"])
    target_speed_ms  = target_speed_kmh / 3.6

    return controller, target_speed_ms, target_speed_kmh


KP_SPEED = 0.25   # proportional gain for longitudinal speed control

def compute_throttle(speed_ms, target_ms):
    """Proportional longitudinal speed controller for smoother acceleration."""
    error = target_ms - speed_ms
    if error < -0.3:
        # Coasting only — no brake applied yet.
        return 0.0
    throttle = KP_SPEED * error
    return max(0.0, min(0.45, throttle))


def _apply_ai_params_to_controller(controller, base_name, params):
    """Write AI supervisor parameters into the live controller instance."""
    if base_name == "pure_pursuit":
        controller.lookahead_base = params["lookahead_base"]
        controller.lookahead_gain = params["lookahead_gain"]
    elif base_name == "stanley":
        controller.k               = params["k"]
        controller.softening_speed = params["softening_speed"]
    elif base_name == "pid":
        controller.Kp_cte    = params["Kp_cte"]
        controller.Kd_cte    = params["Kd_cte"]
        controller.K_heading = params["K_heading"]
    elif base_name == "mpc":
        controller.q_lateral      = params["q_lateral"]
        controller.q_heading      = params["q_heading"]
        controller.r_steer        = params["r_steer"]
        controller.r_steer_change = params["r_steer_change"]


def _print_ai_params(base_name, ai_params, ai_last_command):
    """Print the current AI supervisor parameter state once per second."""
    cmd_str = ai_last_command if ai_last_command else "—"
    if base_name == "pure_pursuit":
        detail = "base={:.2f} gain={:.2f}".format(
            ai_params["lookahead_base"], ai_params["lookahead_gain"])
    elif base_name == "stanley":
        detail = "k={:.2f} soft={:.2f}".format(
            ai_params["k"], ai_params["softening_speed"])
    elif base_name == "pid":
        detail = "Kp={:.3f} Kd={:.3f} Kh={:.3f}".format(
            ai_params["Kp_cte"], ai_params["Kd_cte"], ai_params["K_heading"])
    elif base_name == "mpc":
        detail = "qL={:.2f} qH={:.2f} rS={:.3f} rC={:.3f}".format(
            ai_params["q_lateral"], ai_params["q_heading"],
            ai_params["r_steer"], ai_params["r_steer_change"])
    else:
        detail = ""
    print("         [AI] speed={:.1f}km/h  {}  cmd='{}'".format(
        ai_params["target_speed_kmh"], detail, cmd_str))


def _print_llm_params(base_name, llm_params, llm_status, llm_reason):
    """Print the current LLM interactive supervisor state (once per second)."""
    if base_name == "pure_pursuit":
        detail = "base={:.2f} gain={:.2f}".format(
            llm_params["lookahead_base"], llm_params["lookahead_gain"])
    elif base_name == "stanley":
        detail = "k={:.2f} soft={:.2f}".format(
            llm_params["k"], llm_params["softening_speed"])
    elif base_name == "pid":
        detail = "Kp={:.3f} Kd={:.3f} Kh={:.3f}".format(
            llm_params["Kp_cte"], llm_params["Kd_cte"], llm_params["K_heading"])
    elif base_name == "mpc":
        detail = "qL={:.2f} qH={:.2f} rS={:.3f} rC={:.3f}".format(
            llm_params["q_lateral"], llm_params["q_heading"],
            llm_params["r_steer"], llm_params["r_steer_change"])
    else:
        detail = ""
    reason_short = (llm_reason[:50] + "…") if len(llm_reason) > 50 else (llm_reason or "—")
    print("         [LLM] speed={:.1f}km/h  {}  status='{}'  reason='{}'".format(
        llm_params["target_speed_kmh"], detail, llm_status, reason_short))


def build_log_row(elapsed, state, result, controller_name, throttle, lat_err, head_err):
    """
    Build a CSV data row.
    """
    row = {
        "timestamp_s":       elapsed,
        "x_m":               state.x,
        "y_m":               state.y,
        "yaw_deg":           state.yaw_deg,
        "speed_ms":          state.speed_ms,
        "speed_kmh":         state.speed_kmh,
        "throttle":          throttle,
        "steer_cmd":         result.steer_cmd,
        "lateral_error_m":   lat_err,
        "heading_error_rad": head_err,
    }

    if controller_name in ("pure_pursuit", "llm_pure_pursuit",
                           "interactive_pure_pursuit", "ai_pure_pursuit",
                           "llm_interactive_pure_pursuit"):
        row["target_x"]    = result.target_x
        row["target_y"]    = result.target_y
        row["lookahead_m"] = result.lookahead_distance
        row["alpha_rad"]   = result.alpha_rad
    elif controller_name in ("stanley", "ai_stanley", "llm_interactive_stanley"):
        row["target_x"]    = result.nearest_x
        row["target_y"]    = result.nearest_y
        row["lookahead_m"] = 0.0
        row["alpha_rad"]   = result.heading_error_rad
    elif controller_name in ("pid", "ai_pid", "llm_interactive_pid"):
        row["target_x"]    = result.nearest_x
        row["target_y"]    = result.nearest_y
        row["lookahead_m"] = 0.0
        row["alpha_rad"]   = result.heading_error_rad
    elif controller_name in ("mpc", "ai_mpc", "llm_interactive_mpc"):
        row["target_x"]           = result.nearest_x
        row["target_y"]           = result.nearest_y
        row["lookahead_m"]        = 0.0
        row["alpha_rad"]          = result.heading_error_rad
        row["predicted_cost"]     = result.predicted_cost
        row["selected_steer_rad"] = result.selected_steer_rad

    return row


def print_loop_line(elapsed, state, result, controller_name, lat_err, head_err):
    """Print a one-line console status update."""
    if controller_name in ("pure_pursuit", "llm_pure_pursuit",
                           "interactive_pure_pursuit", "ai_pure_pursuit",
                           "llm_interactive_pure_pursuit"):
        extra = "Ld={:.2f}m".format(result.lookahead_distance)
    elif controller_name in ("mpc", "ai_mpc", "llm_interactive_mpc"):
        extra = "cost={:.4f} sel={:.3f}rad".format(
            result.predicted_cost, result.selected_steer_rad)
    else:
        extra = "cte={:.3f}m".format(result.cross_track_error)

    print(
        "[t={:05.1f}s] "
        "speed={:.2f}m/s ({:.1f}km/h) | "
        "lat_err={:.3f}m | "
        "head_err={:.3f}deg | "
        "steer={:.3f} | "
        "{}".format(
            elapsed,
            state.speed_ms, state.speed_kmh,
            lat_err,
            head_err * 57.2958,
            result.steer_cmd,
            extra,
        )
    )




# Main
def main():
    client        = None
    vehicle       = None
    logger        = None
    original_settings = None   # saved so we can restore them in finally

    controller_label   = CONTROLLER.replace("_", " ").title()
    is_ai_mode         = CONTROLLER.startswith("ai_")
    base_ctrl_name     = CONTROLLER[3:] if is_ai_mode else CONTROLLER
    is_llm_interactive = CONTROLLER.startswith("llm_interactive_")
    llm_base_ctrl      = CONTROLLER[len("llm_interactive_"):] if is_llm_interactive else ""

    try:
        print("[INFO] Controller selected: {}".format(controller_label))
        print("[INFO] Connecting to CARLA server at localhost:2000 ...")
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        print("[INFO] Connection established.")

        world    = client.get_world()
        map_name = world.get_map().name
        print("[INFO] Current map: {}".format(map_name))
        _REQUIRED_MAP = "Town10HD_Opt"
        if _REQUIRED_MAP not in map_name:
            print("[WARN] *** Expected map containing '{}' but got '{}'. ***".format(
                _REQUIRED_MAP, map_name))
            print("[WARN] *** Load Town10HD_Opt in CARLA before collecting final results. ***")

        original_settings = world.get_settings()
        sync_settings = world.get_settings()
        sync_settings.synchronous_mode  = True
        sync_settings.fixed_delta_seconds = LOOP_DT
        world.apply_settings(sync_settings)
        print("[INFO] Synchronous mode enabled (fixed_delta_seconds={}).".format(LOOP_DT))


        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
        print("[INFO] Blueprint loaded: {}".format(vehicle_bp.id))

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found in the current map.")

        spawn_transform = spawn_points[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
        print("[INFO] Vehicle spawned at: x={:.2f}, y={:.2f}, z={:.2f}".format(
            spawn_transform.location.x,
            spawn_transform.location.y,
            spawn_transform.location.z,
        ))

        # Advance one tick so the simulator places the vehicle before reading state.
        world.tick()

        # Route planning
        print("[INFO] Generating route ...")
        route_distance = (INTERACTIVE_ROUTE_DISTANCE_M
                          if (CONTROLLER == "interactive_pure_pursuit"
                              or is_llm_interactive
                              or is_ai_mode)
                          else NORMAL_ROUTE_DISTANCE_M)
        route = generate_route(world, spawn_transform.location,
                               distance=route_distance, spacing=ROUTE_SPACING_M)
        print_route_summary(route)

        if not route:
            raise RuntimeError("Route generation failed — cannot run {}.".format(controller_label))


        _marker_run_s = (INTERACTIVE_RUN_DURATION_S
                         if (CONTROLLER == "interactive_pure_pursuit"
                             or is_llm_interactive
                             or is_ai_mode)
                         else NORMAL_RUN_DURATION_S)
        route_marker_lifetime = _marker_run_s + 30.0
        draw_route(world, route, lifetime=route_marker_lifetime)

        # Position the spectator camera behind the vehicle before the run starts.
        update_spectator_camera(world, vehicle)

        # --- Controller and logger setup ---
        if CONTROLLER == "llm_pure_pursuit":
            controller, run_target_ms, run_target_kmh = make_llm_controller()
        else:
            controller = make_controller(CONTROLLER)
            if CONTROLLER == "interactive_pure_pursuit":
                run_target_kmh = 10.0          # slower start — user can type 'faster'
                run_target_ms  = run_target_kmh / 3.6
            else:
                run_target_ms  = TARGET_SPEED_MS
                run_target_kmh = TARGET_SPEED_KMH

        # Interactive mode uses a longer run duration and a live supervisor.
        if CONTROLLER == "interactive_pure_pursuit":
            run_duration_s = INTERACTIVE_RUN_DURATION_S
            cmd_supervisor = LiveCommandSupervisor()
            active_command = ""   # last command that changed params
            _cmd_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "llm_supervisor", "live_command.txt"
            )
            print("[INFO] Interactive mode active — {}s run, {:.0f}m route.".format(
                int(INTERACTIVE_RUN_DURATION_S), route_distance))
            print("[INFO] Edit the command file and save it:")
            print("[INFO]   {}".format(_cmd_path))
            print("[INFO] Valid commands: faster | slower | smoother | sharper | "
                  "more aggressive | normal | reset")
        elif is_llm_interactive:
            run_duration_s = INTERACTIVE_RUN_DURATION_S
            cmd_supervisor = None
            active_command = ""
            print("[INFO] LLM interactive mode: {}s run, {:.0f}m route.".format(
                int(INTERACTIVE_RUN_DURATION_S), route_distance))
        elif is_ai_mode:
            run_duration_s = INTERACTIVE_RUN_DURATION_S
            cmd_supervisor = None
            active_command = ""
            print("[INFO] AI supervised mode: {}s run, {:.0f}m route.".format(
                int(INTERACTIVE_RUN_DURATION_S), route_distance))
        else:
            run_duration_s = NORMAL_RUN_DURATION_S
            cmd_supervisor = None
            active_command = ""
            print("[INFO] Baseline mode: {}s run, {:.0f}m route.".format(
                int(NORMAL_RUN_DURATION_S), route_distance))

        #  AI supervisor setup
        if is_ai_mode:
            _ai_req_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "llm_supervisor", "user_request.txt"
            )
            # Seed user_request.txt with the standard dissertation request so all
            # ai_* runs start from identical conditions for fair comparison.
            _INITIAL_LLM_REQUEST = (
                "Improve tracking accuracy around bends while avoiding "
                "harsh steering corrections."
            )
            with open(_ai_req_path, "w") as _req_f:
                _req_f.write(_INITIAL_LLM_REQUEST)
            ai_supervisor      = LiveLLMSupervisor(controller_type=base_ctrl_name)
            ai_params          = ai_supervisor.active_params
            ai_status          = "idle"
            ai_reason          = ""
            ai_current_request = ""
            _apply_ai_params_to_controller(controller, base_ctrl_name, ai_params)
            print("[INFO] AI supervised mode: {} (base controller: {})".format(
                CONTROLLER, base_ctrl_name))
            print("[INFO] Initial request: {}".format(_INITIAL_LLM_REQUEST))
            print("[INFO] Request file:   {}".format(_ai_req_path))
            print("[LLM] Watching user_request.txt")
        else:
            ai_supervisor      = None
            ai_params          = {}
            ai_status          = "idle"
            ai_reason          = ""
            ai_current_request = ""


        # ── LLM interactive supervisor setup (llm_interactive_* modes only) ──
        if is_llm_interactive:
            llm_supervisor      = LiveLLMSupervisor(controller_type=llm_base_ctrl)
            llm_params          = llm_supervisor.active_params
            llm_status          = "idle"
            llm_reason          = ""
            llm_current_request = ""
            run_target_kmh      = llm_params["target_speed_kmh"]
            run_target_ms       = run_target_kmh / 3.6
            _apply_ai_params_to_controller(controller, llm_base_ctrl, llm_params)
            _req_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "llm_supervisor", "user_request.txt"
            )
            print("[INFO] LLM interactive mode: {} (base: {})".format(
                CONTROLLER, llm_base_ctrl))
            print("[INFO] Type a natural-language request into:")
            print("[INFO]   {}".format(_req_path))
            print("[INFO] Save the file — the LLM will interpret it and update "
                  "parameters while the simulation runs.")
        else:
            llm_supervisor      = None
            llm_params          = {}
            llm_status          = "idle"
            llm_reason          = ""
            llm_current_request = ""


        logger   = DataLogger(controller_name=CONTROLLER)
        log_rows = []
        print("[INFO] Logging to: {}".format(logger.filepath))

        print("[INFO] Run duration:     {:.0f} s".format(run_duration_s))
        print("[INFO] Route distance:   {:.0f} m".format(route_distance))
        print("[INFO] Expected rows:    {}".format(int(run_duration_s / LOOP_DT)))
        print("[INFO] Real-time pacing: {}".format(
            "enabled" if REAL_TIME_PACING else "disabled"))
        print("[INFO] Starting {} | target={:.0f} km/h | duration={:.0f}s".format(
            controller_label, run_target_kmh, run_duration_s
        ))


        num_steps     = int(run_duration_s / LOOP_DT)
        print_every_n = int(1.0 / LOOP_DT)   # = 20 steps → 1 simulated second
        traj_every_n  = int(0.2 / LOOP_DT)   # = 4 steps  → 0.2 simulated seconds

        for step in range(num_steps):
            elapsed    = step * LOOP_DT         # simulated seconds elapsed
            step_start = time.perf_counter()    # wall-clock start of this step


            world.tick()

            # ── Live command polling (interactive mode, once per sim second) ──
            if cmd_supervisor is not None and step % print_every_n == 0:
                changed, new_params, active_command_this_step = cmd_supervisor.poll()
                if changed:
                    active_command          = active_command_this_step
                    run_target_kmh          = new_params["target_speed_kmh"]
                    run_target_ms           = run_target_kmh / 3.6
                    controller.lookahead_base  = new_params["lookahead_base"]
                    controller.lookahead_gain  = new_params["lookahead_gain"]


            # AI supervisor polling (ai_* modes, once per sim second)
            if ai_supervisor is not None and step % print_every_n == 0:
                _ai_changed, ai_params, ai_status, ai_reason, ai_current_request = \
                    ai_supervisor.poll()
                if _ai_changed:
                    run_target_kmh = ai_params["target_speed_kmh"]
                    run_target_ms  = run_target_kmh / 3.6
                    _apply_ai_params_to_controller(
                        controller, base_ctrl_name, ai_params)


            #  LLM interactive supervisor polling (once per sim second)
            if llm_supervisor is not None and step % print_every_n == 0:
                _llm_changed, llm_params, llm_status, llm_reason, llm_current_request = \
                    llm_supervisor.poll()
                if _llm_changed:
                    run_target_kmh = llm_params["target_speed_kmh"]
                    run_target_ms  = run_target_kmh / 3.6
                    _apply_ai_params_to_controller(
                        controller, llm_base_ctrl, llm_params)
                    print("[LLM] Applied new parameters: speed={:.1f}km/h".format(
                        run_target_kmh))


            state = get_vehicle_state(vehicle)

            # PID variants use the fixed timestep for derivative/integral terms.
            if CONTROLLER in ("pid", "ai_pid", "llm_interactive_pid"):
                result = controller.compute(state, route, dt=LOOP_DT)
            else:
                result = controller.compute(state, route)

            throttle = compute_throttle(state.speed_ms, run_target_ms)

            #  Route-end protection (all modes)
            route_ended = False
            last_wp     = route[-1]
            dist_to_end = math.sqrt(
                (state.x - last_wp.x) ** 2 + (state.y - last_wp.y) ** 2
            )
            if dist_to_end < 5.0:
                print("[WARN] Vehicle near end of route — stopping safely.")
                throttle    = 0.0
                route_ended = True


            control = carla.VehicleControl()
            control.throttle = throttle
            control.steer    = result.steer_cmd
            vehicle.apply_control(control)

            lat_err  = lateral_error(state, route)
            head_err = heading_error(state, route)

            row = build_log_row(elapsed, state, result, CONTROLLER, throttle, lat_err, head_err)

            # Add interactive fields to the row (blank for non-interactive runs).
            if CONTROLLER == "interactive_pure_pursuit":
                row["active_target_speed_kmh"] = run_target_kmh
                row["active_lookahead_base"]    = controller.lookahead_base
                row["active_lookahead_gain"]    = controller.lookahead_gain
                row["command_text"]             = active_command

            # Add AI supervisor fields to the row (blank for baseline modes).
            if ai_supervisor is not None:
                row["latest_user_request"]    = ai_current_request
                row["llm_status"]             = ai_status
                row["llm_reason"]             = ai_reason
                row["active_target_speed_kmh"] = ai_params["target_speed_kmh"]
                if base_ctrl_name == "pure_pursuit":
                    row["active_lookahead_base"] = ai_params["lookahead_base"]
                    row["active_lookahead_gain"] = ai_params["lookahead_gain"]
                elif base_ctrl_name == "stanley":
                    row["active_stanley_k"]       = ai_params["k"]
                    row["active_softening_speed"]  = ai_params["softening_speed"]
                elif base_ctrl_name == "pid":
                    row["active_Kp_cte"]    = ai_params["Kp_cte"]
                    row["active_Kd_cte"]    = ai_params["Kd_cte"]
                    row["active_K_heading"] = ai_params["K_heading"]
                elif base_ctrl_name == "mpc":
                    row["active_q_lateral"]      = ai_params["q_lateral"]
                    row["active_q_heading"]       = ai_params["q_heading"]
                    row["active_r_steer"]         = ai_params["r_steer"]
                    row["active_r_steer_change"]  = ai_params["r_steer_change"]

            # Add LLM interactive fields to the row (blank for non-llm_interactive_* runs).
            if llm_supervisor is not None:
                row["latest_user_request"]   = llm_current_request
                row["llm_reason"]            = llm_reason
                row["llm_status"]            = llm_status
                row["active_target_speed_kmh"] = llm_params["target_speed_kmh"]
                if llm_base_ctrl == "pure_pursuit":
                    row["active_lookahead_base"] = llm_params["lookahead_base"]
                    row["active_lookahead_gain"] = llm_params["lookahead_gain"]
                elif llm_base_ctrl == "stanley":
                    row["active_stanley_k"]       = llm_params["k"]
                    row["active_softening_speed"]  = llm_params["softening_speed"]
                elif llm_base_ctrl == "pid":
                    row["active_Kp_cte"]    = llm_params["Kp_cte"]
                    row["active_Kd_cte"]    = llm_params["Kd_cte"]
                    row["active_K_heading"] = llm_params["K_heading"]
                elif llm_base_ctrl == "mpc":
                    row["active_q_lateral"]      = llm_params["q_lateral"]
                    row["active_q_heading"]      = llm_params["q_heading"]
                    row["active_r_steer"]        = llm_params["r_steer"]
                    row["active_r_steer_change"] = llm_params["r_steer_change"]

            logger.log(row)
            log_rows.append(row)

            # Draw trajectory dot every 0.2 simulated seconds.
            if step % traj_every_n == 0:
                draw_vehicle_position(world, state)

            # Keep the spectator camera locked behind the vehicle.
            update_spectator_camera(world, vehicle)

            # Print status once per simulated second.
            if step % print_every_n == 0:
                print_loop_line(elapsed, state, result, CONTROLLER, lat_err, head_err)
                if CONTROLLER == "interactive_pure_pursuit":
                    print("         [PARAMS] speed={:.1f}km/h  base={:.2f}  "
                          "gain={:.2f}  cmd='{}'".format(
                              run_target_kmh,
                              controller.lookahead_base,
                              controller.lookahead_gain,
                              active_command if active_command else "—",
                          ))
                if ai_supervisor is not None:
                    _print_llm_params(base_ctrl_name, ai_params, ai_status, ai_reason)
                if llm_supervisor is not None:
                    _print_llm_params(llm_base_ctrl, llm_params, llm_status, llm_reason)

            # Real-time pacing: sleep to match wall-clock LOOP_DT.
            if REAL_TIME_PACING:
                step_wall = time.perf_counter() - step_start
                time.sleep(max(0.0, LOOP_DT - step_wall))

            # Stop the run cleanly when the interactive route end is reached.
            if route_ended:
                print("[INFO] Route end reached — stopping run early.")
                break

        #End-of-run summary
        metrics  = calculate_summary_metrics(log_rows)
        total_s  = metrics.get("total_duration_s", 0.0) if metrics else 0.0
        print("[INFO] {} run complete. Rows logged: {}  |  Duration: {:.1f}s".format(
            controller_label, logger.row_count, total_s
        ))
        print_summary_table(metrics, controller_name=controller_label)
        print("[INFO] CSV saved to: {}".format(logger.filepath))

    except Exception as e:
        print("[ERROR] {}".format(e))

    finally:
        # Restore original world settings before exiting so CARLA is left in
        # the same state it was in before the script ran.
        if original_settings is not None and client is not None:
            try:
                world = client.get_world()
                world.apply_settings(original_settings)
                print("[INFO] CARLA world settings restored.")
            except Exception as restore_err:
                print("[WARN] Could not restore world settings: {}".format(restore_err))
        if logger is not None:
            logger.close()
        if vehicle is not None:
            vehicle.destroy()
            print("[INFO] Vehicle destroyed. Clean exit.")


if __name__ == "__main__":
    main()
