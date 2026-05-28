# AI-Integrated Lane-Keeping Assist in CARLA

This repository contains the implementation and simulation results for a CARLA-based Lane-Keeping Assist (LKA) project. The work compares classical path-tracking controllers with AI/LLM-supervised variants, where a local language model interprets high-level user requests and adjusts bounded controller parameters during simulation.

The project was developed for a final-year aerospace engineering individual project and uses CARLA simulation rather than physical robot-car testing.

---

## Project Overview

The aim of this project is to evaluate autonomous lane-keeping/path-tracking performance using deterministic controllers and AI-supervised controller variants in a repeatable simulation environment.

The system uses a CARLA ego vehicle following a generated lane-centre reference route. The controllers minimise lateral error and heading error while maintaining a target speed. The AI/LLM layer does not directly command steering. Instead, it supervises the controller by modifying safe, bounded parameters such as lookahead distance, Stanley gain, PID gains, MPC weights, and target speed.

---

## Controllers Implemented

The repository includes four baseline controllers:

- **Pure Pursuit** geometric path-tracking controller
- **Stanley** cross-track and heading-error controller
- **PID lateral controller** using cross-track and heading-error feedback
- **Sampled kinematic MPC-style controller** using finite-horizon candidate steering evaluation

It also supports AI/LLM-supervised variants:

- **AI Pure Pursuit**
- **AI Stanley**
- **AI PID**
- **AI MPC**

The AI-supervised modes use an Ollama-based local LLM supervisor to interpret natural-language requests such as:

```text
Improve tracking accuracy around bends while avoiding harsh steering corrections.
```

The supervisor then updates bounded controller parameters while the deterministic controller remains responsible for generating steering commands.

---

## Key Features

- CARLA 0.9.15 simulation environment
- Python 3.7 compatible implementation
- Synchronous simulation mode with fixed time step
- Lane-centre route generation from the CARLA map
- Multiple lateral controllers implemented in Python
- Live/interactive parameter supervision
- Local Ollama LLM integration
- CSV logging of vehicle state, steering, speed, lateral error, heading error, and controller-specific variables
- Final plots and comparison data for controller performance analysis

---

## Repository Structure

```text
CARLA_LKA_Project/
├── main.py
├── requirements.txt
├── controllers/
│   ├── pure_pursuit.py
│   ├── stanley.py
│   ├── pid_lateral.py
│   └── mpc_controller.py
├── core/
│   ├── vehicle_state.py
│   ├── route_planner.py
│   ├── metrics.py
│   └── data_logger.py
├── llm_supervisor/
│   ├── live_llm_supervisor.py
│   ├── live_command_supervisor.py
│   ├── universal_supervisor.py
│   ├── ollama_supervisor.py
│   ├── user_request.txt
│   ├── live_command.txt
│   └── live_llm_params.json
├── results/
│   ├── final_logs/
│   └── final_plots/
└── documentation/
    ├── FINAL_RESULTS_NOTES.txt
    ├── PLOT_LABEL_CHECKLIST.txt
    └── PROJECT_FILE_TREE.txt
```

---

## Software Requirements

The project was developed using:

- Windows with CARLA 0.9.15
- Python 3.7
- CARLA Python API
- NumPy
- Pandas
- Matplotlib
- Requests
- Ollama for local LLM supervision

Install the Python dependencies using:

```bash
pip install -r requirements.txt
```

For CARLA compatibility, Python 3.7 is recommended.

---

## Running the Simulation

1. Start the CARLA simulator and load the required map:

```text
Town10HD_Opt
```

2. Open the project folder in a terminal.

3. Select the controller in `main.py` by editing the `CONTROLLER` variable. Example:

```python
CONTROLLER = "ai_mpc"
```

Supported controller names include:

```text
pure_pursuit
stanley
pid
mpc
ai_pure_pursuit
ai_stanley
ai_pid
ai_mpc
interactive_pure_pursuit
llm_interactive_pure_pursuit
llm_interactive_stanley
llm_interactive_pid
llm_interactive_mpc
```

4. Run the simulation:

```bash
python main.py
```

On Windows with Python 3.7 installed separately, the command may be:

```powershell
py -3.7 main.py
```

---

## LLM/AI Supervisor Use

The AI supervisor reads natural-language instructions from:

```text
llm_supervisor/user_request.txt
```

For the final AI-supervised runs, the standard request used was:

```text
Improve tracking accuracy around bends while avoiding harsh steering corrections.
```

The LLM adjusts only bounded high-level parameters. It does not output steering commands, throttle commands, or direct vehicle-control actions. This keeps the deterministic control loop responsible for safety-critical actuation.

Example parameters adjusted by the supervisor include:

| Controller | Parameters Adjusted |
|---|---|
| Pure Pursuit | Target speed, lookahead base, lookahead gain |
| Stanley | Target speed, Stanley gain, softening speed |
| PID | Target speed, cross-track and heading gains |
| MPC | Target speed, lateral-error weight, heading-error weight, steering penalties |

---

## Results

Final accepted simulation logs and plots are stored in:

```text
results/final_logs/
results/final_plots/
```

The final comparison considered eight controller modes:

1. Pure Pursuit
2. Stanley
3. PID lateral control
4. MPC
5. AI Pure Pursuit
6. AI Stanley
7. AI PID
8. AI MPC

The main performance metrics were:

- RMS lateral error
- Mean absolute lateral error
- Maximum absolute lateral error
- RMS heading error
- Mean absolute steering command
- Mean speed
- Maximum speed

The results show that AI supervision can alter the accuracy, speed, and smoothness trade-off, but it should not be interpreted as universally improving every metric. The key contribution is the bounded supervisory architecture, where high-level user intent is translated into safe controller-parameter adaptation.

---

## Methodology Summary

The simulations use a fixed CARLA setup for fair comparison:

- Map: `Town10HD_Opt`
- Ego vehicle: Tesla Model 3
- Simulation mode: synchronous
- Fixed time step: 0.05 s
- Route: lane-centre waypoints generated from the CARLA map
- Validation: CSV logging and post-processing of lateral error, heading error, steering demand, and speed

---

## Academic Basis

The control and simulation approach is based on established path-tracking and autonomous-vehicle literature, including:

- Coulter, R. C. (1992). *Implementation of the Pure Pursuit Path Tracking Algorithm*. Carnegie Mellon University Robotics Institute Technical Report CMU-RI-TR-92-01.
- Thrun, S., Montemerlo, M., Dahlkamp, H., et al. (2006). Stanley: The robot that won the DARPA Grand Challenge. *Journal of Field Robotics*, 23(9), 661–692.
- Falcone, P., Borrelli, F., Asgari, J., Tseng, H. E. and Hrovat, D. (2007). Predictive active steering control for autonomous vehicle systems. *IEEE Transactions on Control Systems Technology*, 15(3), 566–580.
- Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A. and Koltun, V. (2017). CARLA: An open urban driving simulator. *Proceedings of the Conference on Robot Learning*.

---

## Notes and Limitations

- The project is validated in simulation only.
- The LLM is used as a bounded supervisor, not as a direct controller.
- LLM response time and model reliability are practical limitations.
- Results are dependent on CARLA version, map, route, spawn point, and controller tuning.
- Further work could include repeated trials, disturbance testing, sensor-noise modelling, and hardware validation.

---

## Author

Navneet Singh  
Final Year Aerospace Engineering Individual Project  
University of Surrey
