import argparse
import os
import torch
import pandas as pd
import numpy as np
from pytransform3d.rotations import matrix_from_axis_angle

from model import MLP
from utils import getAzimuthElevation, lookat, matrix_from_axis_angle


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default=r"cvpr\version_1", help='checkpoint path of the trained model')
    parser.add_argument('--tracking_file', type=str, default=r"tracking.csv", help='CSV file containing tracking data')
    args = parser.parse_args()

    # Load the model from the checkpoint
    model_dir = os.path.join("lightning_logs", args.checkpoint_dir)
    for file in os.listdir(model_dir):
        if file.endswith(".ckpt"):
            checkpoint_path = os.path.join(model_dir, file)
            break
    if not checkpoint_path:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    model = MLP.load_from_checkpoint(checkpoint_path).to(device)
    print(f"Loading model from {checkpoint_path}")
    print(model)

    # Load CSV file using Pandas
    tracking_file = args.tracking_file
    if not os.path.exists(tracking_file):
        raise FileNotFoundError(f"File {tracking_file} not found")
    df = pd.read_csv(tracking_file)

    focal_length=0.0036
    sensor_size=(0.00367, 0.00274)
    image_size=(640, 480)

    # Define camera pose
    camera_position = np.array([5, 5, 5])
    up = np.array([0, 0, 1])
    from_point = camera_position
    to_point = np.array([0, 0, 0])    # the camera always points at the origin of the world coordinates

    # Compute Rotation Matrix
    R_C2W, _ = lookat(from_point, to_point, up)
    R_C2W = R_C2W @ matrix_from_axis_angle((1, 0, 0, np.pi))

    # Initialize new columns in DataFrame
    df["x_world_new"] = np.nan
    df["y_world_new"] = np.nan

    # Iterate through each row and perform inference
    for idx, row in df.iterrows():

        # Compute Bounding Box Center, Azimuth and Elevation
        bb_center = [(row["bbox_x_center"]*image_size[0], row["bbox_y_center"]*image_size[1])]
        azimuth, elevation, _ = getAzimuthElevation(focal_length, sensor_size, image_size, bb_center, R_C2W)

        # Forward pass through the model
        output = model(torch.tensor([row["bbox_x_center"], row["bbox_y_center"], row["bbox_width"], row["bbox_height"], azimuth, elevation], dtype=torch.float32).to(device))
        x_world_new = output[0].item()
        y_world_new = output[1].item()

        # Store results in DataFrame
        df.at[idx, "x_world_new"] = x_world_new
        df.at[idx, "y_world_new"] = y_world_new

# Save updated DataFrame to a new CSV file
df.to_csv("tracking_and_localization.csv", index=False)

print("Inference complete. Results saved to tracking_updated.csv")