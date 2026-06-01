import argparse
import json
import os
import matplotlib.pyplot as plt

# Parse command line arguments
# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Plot reward and velocity logs from REvolve runs."
)
parser.add_argument(
    "--run",
    type=str,
    default="test_run_ant",
    help="Name of the run directory (e.g., test_run or test_run_ant)"
)
parser.add_argument(
    "--individual",
    type=str,
    default="0_0",
    help="Identifier of the individual to plot (e.g., 0_0 or 4_1)"
)
args = parser.parse_args()

base_dir = os.path.dirname(os.path.abspath(__file__))
reward_file = os.path.join(
    base_dir,
    f"database/revolve_auto/{args.run}/island_0/reward_history/{args.individual}.json"
)
velocity_file = os.path.join(
    base_dir,
    f"database/revolve_auto/{args.run}/island_0/velocity_logs/velocity_{args.individual}.txt"
)
output_plot = os.path.join(
    base_dir,
    f"training_results_{args.run}_{args.individual}.png"
)

# Load reward history
rewards = []
component_data = {}

if os.path.exists(reward_file):
    with open(reward_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    rewards.append(data["total_reward"])
                    components = data.get("episode_components", {})
                    for key, val in components.items():
                        if key not in component_data:
                            component_data[key] = []
                        component_data[key].append(val)
                except Exception as e:
                    print(f"Error parsing line: {e}")

# Load velocity logs
velocities = []
if os.path.exists(velocity_file):
    with open(velocity_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    velocities.append(float(line.strip()))
                except Exception as e:
                    print(f"Error parsing velocity: {e}")

# Generate plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Rewards
if rewards:
    episodes = range(1, len(rewards) + 1)
    ax1.plot(
        episodes,
        rewards,
        label="Total Reward",
        color="#1f77b4",
        linewidth=2
    )
    for name, vals in component_data.items():
        ax1.plot(episodes, vals, label=name, linestyle="--")
    ax1.set_title(
        f"Reward per Episode ({args.run})",
        fontsize=14,
        fontweight="bold",
        pad=10
    )
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Reward Value", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)
else:
    ax1.text(
        0.5,
        0.5,
        "No reward data found",
        ha="center",
        va="center",
        fontsize=12
    )

# Plot Velocities
if velocities:
    steps = range(1, len(velocities) + 1)
    ax2.plot(
        steps,
        velocities,
        color="#d62728",
        linewidth=1.5,
        label="Velocity (x-axis)"
    )
    ax2.set_title(
        f"Forward Velocity over Steps ({args.run})",
        fontsize=14,
        fontweight="bold",
        pad=10
    )
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Velocity (m/s)", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)
else:
    ax2.text(
        0.5,
        0.5,
        "No velocity data found",
        ha="center",
        va="center",
        fontsize=12
    )

plt.tight_layout()
plt.savefig(output_plot, dpi=300)
print(f"Plot saved successfully to {output_plot}")
