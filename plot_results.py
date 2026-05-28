"""
plot_results.py — LKA controller performance plots and comparison tables.

Usage:
    python plot_results.py                  # latest CSV in results/final_logs/
    python plot_results.py path/to/run.csv  # specific file
    python plot_results.py --all            # all CSVs + comparison_table.csv
"""

import os
import sys
import glob

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(_SCRIPT_DIR, "results", "final_logs")
PLOT_DIR = os.path.join(_SCRIPT_DIR, "results", "final_plots")

DPI = 300


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

def find_latest_csv(log_dir):
    """Return the path of the most recently modified CSV in log_dir."""
    pattern = os.path.join(log_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            "No CSV files found in {}. Run main.py first.".format(log_dir)
        )
    return max(files, key=os.path.getmtime)


# Acronyms that capitalize() gets wrong — applied to plot titles only,
# not to filename prefixes or CSV controller_name values.
_LABEL_FIXES = {
    "Ai":  "AI",
    "Pid": "PID",
    "Mpc": "MPC",
}


def _fix_label(raw_label):
    """Correct known acronyms in a capitalize()-generated label string."""
    return " ".join(_LABEL_FIXES.get(word, word) for word in raw_label.split())


def load_csv(path):
    """
    Load a run CSV into a pandas DataFrame and derive a controller label and prefix.

    Returns:
        df     – pandas DataFrame
        label  – human-readable title string, e.g. "Pure Pursuit", "AI MPC"
        prefix – filename-safe slug, e.g. "pure_pursuit", "ai_mpc"
    """
    df = pd.read_csv(path)

    # Derive label and prefix from the filename.
    # e.g. "pure_pursuit_20260430_191500" -> label="Pure Pursuit", prefix="pure_pursuit"
    basename = os.path.splitext(os.path.basename(path))[0]
    parts = basename.split("_")
    # Drop the trailing date/time tokens (last two parts are YYYYMMDD and HHMMSS).
    name_parts = parts[:-2] if len(parts) > 2 else parts
    label  = _fix_label(" ".join(p.capitalize() for p in name_parts))
    prefix = "_".join(name_parts)

    return df, label, prefix


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def _save(fig, filename):
    """Save figure to PLOT_DIR and print its path."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] Saved: {}".format(path))


def plot_lateral_error(df, label, prefix):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp_s"], df["lateral_error_m"],
            color="steelblue", linewidth=1.0, label="Lateral error")
    ax.axhline(df["lateral_error_m"].mean(), color="orange", linestyle="--",
               linewidth=1.0, label="Mean = {:.3f} m".format(df["lateral_error_m"].mean()))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lateral Error (m)")
    ax.set_title("{} — Lateral Error vs Time".format(label))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _save(fig, "{}_lateral_error_vs_time.png".format(prefix))


def plot_heading_error(df, label, prefix):
    # Convert radians to degrees for readability
    heading_deg = df["heading_error_rad"].apply(lambda r: r * 57.2958)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp_s"], heading_deg,
            color="darkorange", linewidth=1.0, label="Heading error")
    ax.axhline(0.0, color="grey", linestyle="-", linewidth=0.8)
    ax.axhline(heading_deg.mean(), color="purple", linestyle="--",
               linewidth=1.0, label="Mean = {:.3f} deg".format(heading_deg.mean()))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading Error (deg)")
    ax.set_title("{} — Heading Error vs Time".format(label))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _save(fig, "{}_heading_error_vs_time.png".format(prefix))


def plot_steering_command(df, label, prefix):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp_s"], df["steer_cmd"],
            color="crimson", linewidth=1.0, label="Steer command")
    ax.axhline(0.0, color="grey", linestyle="-", linewidth=0.8)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Steering Command (normalised, −1 to 1)")
    ax.set_title("{} — Steering Command vs Time".format(label))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _save(fig, "{}_steering_command_vs_time.png".format(prefix))


def plot_speed(df, label, prefix):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp_s"], df["speed_kmh"],
            color="seagreen", linewidth=1.0, label="Vehicle speed")
    ax.axhline(df["speed_kmh"].mean(), color="navy", linestyle="--",
               linewidth=1.0, label="Mean = {:.2f} km/h".format(df["speed_kmh"].mean()))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("{} — Speed vs Time".format(label))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _save(fig, "{}_speed_vs_time.png".format(prefix))


def plot_trajectory(df, label, prefix):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot route reference points if they are present and non-trivial.
    has_route = (
        "target_x" in df.columns
        and "target_y" in df.columns
        and df["target_x"].notna().any()
        and df["target_x"].abs().max() > 0.01
    )
    if has_route:
        # Deduplicate consecutive identical target points for a cleaner reference line.
        route_df = df[["target_x", "target_y"]].drop_duplicates()
        ax.plot(route_df["target_x"], route_df["target_y"],
                color="lightcoral", linewidth=1.5, linestyle="--",
                label="{} route reference".format(label))

    # Vehicle trajectory
    ax.plot(df["x_m"], df["y_m"],
            color="steelblue", linewidth=1.5, label="Vehicle trajectory")

    # Mark start and end
    ax.plot(df["x_m"].iloc[0], df["y_m"].iloc[0],
            "go", markersize=8, label="Start")
    ax.plot(df["x_m"].iloc[-1], df["y_m"].iloc[-1],
            "rs", markersize=8, label="End")

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("{} — Vehicle Trajectory (XY)".format(label))
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.6)
    _save(fig, "{}_trajectory_xy.png".format(prefix))


# ---------------------------------------------------------------------------
# Metrics extraction (pandas-based, mirrors core/metrics.py for CSV data)
# ---------------------------------------------------------------------------

def compute_metrics_from_df(df):
    """
    Compute summary metrics from a loaded DataFrame.

    Returns a dict with the same keys as core/metrics.calculate_summary_metrics,
    or an empty dict if df is empty.
    """
    if df.empty:
        return {}

    lat  = df["lateral_error_m"].fillna(0.0)
    head = df["heading_error_rad"].fillna(0.0)
    steer = df["steer_cmd"].fillna(0.0)
    speed = df["speed_kmh"].fillna(0.0)
    t    = df["timestamp_s"].fillna(0.0)

    head_deg = head * 57.2958

    return {
        "rms_lateral_error_m":        float((lat ** 2).mean() ** 0.5),
        "mean_abs_lateral_error_m":   float(lat.abs().mean()),
        "max_abs_lateral_error_m":    float(lat.abs().max()),
        "rms_heading_error_deg":      float((head_deg ** 2).mean() ** 0.5),
        "mean_abs_heading_error_deg": float(head_deg.abs().mean()),
        "mean_abs_steer_cmd":         float(steer.abs().mean()),
        "max_abs_steer_cmd":          float(steer.abs().max()),
        "mean_speed_kmh":             float(speed.mean()),
        "max_speed_kmh":              float(speed.max()),
        "total_duration_s":           float(t.iloc[-1] - t.iloc[0]) if len(t) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def generate_comparison_table(csv_paths, output_dir):
    """
    Build a one-row-per-run comparison CSV from a list of log file paths
    and save it to output_dir/comparison_table.csv.
    """
    _COLS = [
        "controller_name",
        "rms_lateral_error_m",
        "mean_abs_lateral_error_m",
        "max_abs_lateral_error_m",
        "rms_heading_error_deg",
        "mean_abs_heading_error_deg",
        "mean_abs_steer_cmd",
        "max_abs_steer_cmd",
        "mean_speed_kmh",
        "max_speed_kmh",
        "total_duration_s",
        "csv_filepath",
    ]

    records = []
    for path in sorted(csv_paths):
        try:
            df, label, prefix = load_csv(path)
        except Exception as exc:
            print("[COMPARE] Skipping {}: {}".format(os.path.basename(path), exc))
            continue
        m = compute_metrics_from_df(df)
        if not m:
            print("[COMPARE] No data in {} — skipped.".format(os.path.basename(path)))
            continue
        m["controller_name"] = prefix
        m["csv_filepath"]    = path
        records.append({c: m.get(c, "") for c in _COLS})

    if not records:
        print("[COMPARE] No valid runs found — comparison table not written.")
        return

    comparison_df = pd.DataFrame(records, columns=_COLS)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "comparison_table.csv")
    comparison_df.to_csv(out_path, index=False, float_format="%.4f")
    print("\n[COMPARE] Comparison table ({} runs) saved: {}".format(
        len(records), out_path))
    print(comparison_df[_COLS[:-1]].to_string(index=False))


# ---------------------------------------------------------------------------
# Per-CSV helper
# ---------------------------------------------------------------------------

def _plot_one(csv_path):
    """Load one CSV, generate all five plots, and return (df, label, prefix)."""
    print("[INFO] Loading: {}".format(csv_path))
    df, label, prefix = load_csv(csv_path)
    print("[INFO] Rows: {:5d}  |  Controller: {}".format(len(df), label))
    plot_lateral_error(df, label, prefix)
    plot_heading_error(df, label, prefix)
    plot_steering_command(df, label, prefix)
    plot_speed(df, label, prefix)
    plot_trajectory(df, label, prefix)
    return df, label, prefix


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # ── Batch mode: process every CSV in the log directory ───────────
        csv_paths = sorted(glob.glob(os.path.join(LOG_DIR, "*.csv")))
        if not csv_paths:
            print("[ERROR] No CSV files found in {}.".format(LOG_DIR))
            sys.exit(1)
        print("[INFO] Found {} CSV file(s). Saving plots to: {}".format(
            len(csv_paths), PLOT_DIR))
        for csv_path in csv_paths:
            _plot_one(csv_path)
        generate_comparison_table(csv_paths, PLOT_DIR)
        print("[INFO] Batch complete.")

    else:
        # ── Single-file mode ─────────────────────────────────────────────
        if len(sys.argv) > 1:
            csv_path = sys.argv[1]
            if not os.path.isfile(csv_path):
                print("[ERROR] File not found: {}".format(csv_path))
                sys.exit(1)
        else:
            csv_path = find_latest_csv(LOG_DIR)

        print("[INFO] Saving plots to: {}".format(PLOT_DIR))
        _plot_one(csv_path)
        print("[INFO] All plots saved.")


if __name__ == "__main__":
    main()
