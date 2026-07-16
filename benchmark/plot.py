from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def collect_csv_files(*paths: Path) -> list[Path]:
    """Collect CSV files from paths or from the default data directory.

    Args:
        paths: CSV files or directories containing CSV files.

    Returns:
        List of CSV file paths.
    """
    default_data_folder = Path(__file__).parent / "data"
    inputs = [Path(p) for p in paths] if paths else [default_data_folder]
    csv_files: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            csv_files.extend(sorted(input_path.glob("*.csv")))
        elif input_path.is_file() and input_path.suffix.lower() == ".csv":
            csv_files.append(input_path)
        else:
            raise ValueError(f"Expected a CSV file or directory, got: {input_path}")
    if not csv_files:
        raise ValueError("No CSV files found for plotting")
    return csv_files


def plot_fps_data(data_folder: Path):
    """Read all CSVs from the data folder and plot the latest gym and sim fps by device.

    Args:
        data_folder: pathlib.Path object pointing to the folder containing CSV files
    """
    # Colors for different devices. Deep blue for CPU, NVIDIA green for GPU
    colors = {"cpu": "#0000AA", "gpu": "#76B900"}
    device_names = {"cpu": "Intel Core i9-13900KF", "gpu": "NVIDIA RTX 4090"}
    dfs = {}

    # Read all CSV files in the data folder
    for csv_file in sorted(data_folder.glob("*.csv")):
        df = pd.read_csv(csv_file)
        device = df["device"].iloc[-1].lower()
        # Group data by test_type and n_worlds
        dfs["sim_" + device] = df[df["test_type"] == "simulator"]
        dfs["gym_" + device] = df[df["test_type"] == "gym_env"]

    if not dfs:
        print("No valid data found for plotting")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Add a super title (suptitle) for the entire figure
    fig.suptitle("Crazyflow Performance", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the suptitle

    # Plot gym FPS
    for key, df in dfs.items():
        if key.startswith("gym_"):
            device = key.split("_")[1]
            color = colors[device]

            # Plot FPS
            ax1.plot(
                df["n_worlds"],
                df["fps"],
                marker="o",
                linestyle="-",
                color=color,
                label=f"{device_names[device]}",
            )

    ax1.set_title("Steps per second: Gym envs")
    ax1.set_xlabel("Number of Worlds")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    format_log_axes(ax1, dfs, "gym_")

    # Plot sim FPS
    for key, df in dfs.items():
        if key.startswith("sim_"):
            device = key.split("_")[1]
            color = colors[device]

            # Plot FPS
            ax2.plot(
                df["n_worlds"],
                df["fps"],
                marker="o",
                linestyle="-",
                color=color,
                label=f"{device_names[device]}",
            )

    ax2.set_title("Steps per second: Crazyflow")
    ax2.set_xlabel("Number of Worlds")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True)
    format_log_axes(ax2, dfs, "sim_")
    # Add legend for the axis
    ax2.legend(loc="upper left")

    plt.tight_layout()

    # Save the plot
    output_path = data_folder / "performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def plot_splat_data(*paths: Path):
    """Plot splat rendering throughput and save render.png.

    Args:
        paths: CSV files or directories containing CSV files. If empty, uses benchmark/data.
    """
    csv_files = collect_csv_files(*paths)
    required = {"test_type", "n_worlds", "fps", "device"}
    series: list[dict[str, str | pd.DataFrame]] = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if not required.issubset(df.columns):
            missing = sorted(required.difference(df.columns))
            raise ValueError(f"CSV {csv_file} is missing required columns: {missing}")
        splat = df[df["test_type"] == "splat"].copy()
        if splat.empty:
            continue
        splat = splat.sort_values("n_worlds")
        device = str(splat["device"].iloc[-1]).lower()
        date_token = csv_file.stem.split("_")[-2]
        date_label = datetime.strptime(date_token, "%Y%m%d").strftime("%d.%m.%y")
        series.append({"device": device, "date": date_label, "df": splat})

    if not series:
        raise ValueError("No splat benchmark data found in provided CSV files")

    colors = {"cpu": "#0000AA", "gpu": "#76B900"}
    device_names = {"cpu": "Intel Core i9-13900KF", "gpu": "NVIDIA RTX 4090"}
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.suptitle("Crazyflow Splat Rendering", fontsize=16, fontweight="bold", y=0.98)

    for item in series:
        device = str(item["device"])
        date_label = str(item["date"])
        df = item["df"]
        label = f"{device_names.get(device, device.upper())} ({date_label})"
        ax.plot(df["n_worlds"], df["fps"], marker="o", linestyle="-", color=colors.get(device), label=label)

    ax.set_title("Images per second: Splat renderer")
    ax.set_xlabel("Number of Worlds")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(loc="upper left")
    format_log_axes(
        ax,
        {f"splat_{idx}": item["df"] for idx, item in enumerate(series)},
        "splat_",
    )
    plt.tight_layout()

    output_dir = csv_files[0].parent
    output_path = output_dir / "render.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def format_log_axes(ax: plt.Axes, dfs: dict[str, pd.DataFrame], prefix: str):
    """Format logarithmic axes with nice labels.

    Args:
        ax: matplotlib axis to format
        dfs: dictionary of dataframes
        prefix: prefix for filtering dataframes (e.g., "gym_" or "sim_")
    """
    # Rename the axes labels
    xticks = np.array([1, 10, 100, 1000, 10000, 100000, 1000000])
    min_x = min([df["n_worlds"].min() for key, df in dfs.items() if key.startswith(prefix)])
    max_x = max([df["n_worlds"].max() for key, df in dfs.items() if key.startswith(prefix)])
    mask = (xticks >= min_x) & (xticks <= max_x)
    valid_indices = np.nonzero(mask)[0]
    ax.set_xticks(xticks[valid_indices])
    xticklabels = ["1", "10", "100", "1K", "10K", "100K", "1M"]
    ax.set_xticklabels([xticklabels[i] for i in valid_indices])

    # Get min and max y values for plots
    min_y = min([df["fps"].min() for key, df in dfs.items()])
    max_y = max([df["fps"].max() for key, df in dfs.items()])

    # Create logarithmic y-ticks
    # Generate yticks based on data range
    min_power = int(np.floor(np.log10(min_y)))
    max_power = int(np.ceil(np.log10(max_y)))
    yticks = np.array([10**i for i in range(min_power, max_power + 1)])
    ax.set_yticks(yticks)
    yticklabels = []
    abbrev = {1e9: "B", 1e6: "M", 1e3: "K"}
    for i in yticks:
        for divisor, suffix in sorted(abbrev.items(), reverse=True):
            if i >= divisor:
                yticklabels.append(f"{int(i // divisor)}{suffix}")
                break
        else:
            yticklabels.append(f"{int(i)}")
    ax.set_yticklabels(yticklabels)

    # Remove minor ticks for cleaner appearance
    ax.minorticks_off()


if __name__ == "__main__":
    data_folder = Path(__file__).parent / "data"
    plot_fps_data(data_folder)
    plot_splat_data(data_folder)
