import argparse
import os
import imageio
import numpy as np
from stable_baselines3 import SAC

from rl_agent.HumanoidEnv import HumanoidEnv
from rl_agent.AntEnv import AntEnv

# Parse arguments
parser = argparse.ArgumentParser(
    description="Record a GIF video of a trained REvolve agent."
)
parser.add_argument(
    "--run",
    type=str,
    default="test_run_ant",
    help="Name of the run directory (e.g., test_run or test_run_ant)"
)
parser.add_argument(
    "--env",
    type=str,
    default="AntEnv",
    help="Environment class name (AntEnv or HumanoidEnv)"
)
parser.add_argument(
    "--steps",
    type=int,
    default=200,
    help="Number of steps to record"
)
parser.add_argument(
    "--individual",
    type=str,
    default="0_0",
    help="Identifier of the individual to record (e.g., 0_0 or 4_1)"
)
args = parser.parse_args()

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(
    base_dir,
    f"database/revolve_auto/{args.run}/island_0/model_checkpoints/SAC_{args.individual}_1000.zip"
)
reward_fn_path = os.path.join(
    base_dir,
    f"database/revolve_auto/{args.run}/island_0/generated_fns/{args.individual}.txt"
)
output_gif = os.path.join(
    base_dir,
    f"recorded_agent_{args.run}_{args.individual}.gif"
)

# Check file existence
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
if not os.path.exists(reward_fn_path):
    raise FileNotFoundError(f"Reward function file not found at: {reward_fn_path}")

# Load reward function string
with open(reward_fn_path, "r") as f:
    reward_func_str = f.read()

# Initialize environment
print(f"Initializing {args.env}...")
reward_history_file = os.path.join(base_dir, "temp_reward_history.json")
velocity_file = os.path.join(base_dir, "temp_velocity.txt")
model_checkpoint_file = os.path.join(base_dir, "temp_checkpoint.zip")

if args.env == "AntEnv":
    gymenv = AntEnv(
        reward_func_str=reward_func_str,
        counter=0,
        generation_id=0,
        island_id="0",
        reward_history_file=reward_history_file,
        velocity_file=velocity_file,
        model_checkpoint_file=model_checkpoint_file,
        render_mode="rgb_array"
    )
else:
    gymenv = HumanoidEnv(
        reward_func_str=reward_func_str,
        counter=0,
        generation_id=0,
        island_id="0",
        reward_history_file=reward_history_file,
        velocity_file=velocity_file,
        model_checkpoint_file=model_checkpoint_file,
        render_mode="rgb_array"
    )

# Load model weights
print("Loading SAC model...")
model = SAC.load(checkpoint_path, env=gymenv)

# Record frames
print(f"Running policy for {args.steps} steps and recording...")
obs, _ = gymenv.reset()
frames = []

for step in range(args.steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = gymenv.step(action)
    
    # Render frame (rgb_array)
    if args.env == "HumanoidEnv":
        frame = gymenv.render("rgb_array")
    else:
        frame = gymenv.render()
    if frame is not None:
        frames.append(frame)
        
    if terminated or truncated:
        obs, _ = gymenv.reset()

gymenv.close()

# Cleanup temp files if created
for temp_file in [reward_history_file, velocity_file, model_checkpoint_file]:
    if os.path.exists(temp_file):
        os.remove(temp_file)

# Save as GIF
if frames:
    print(f"Saving {len(frames)} frames to {output_gif}...")
    imageio.mimsave(output_gif, frames, fps=20)
    print("GIF saved successfully!")
else:
    print("No frames were captured.")
