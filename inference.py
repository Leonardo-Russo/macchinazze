import argparse
import os
import torch
import pandas as pd
import numpy as np
from pytransform3d.rotations import matrix_from_axis_angle
import json

from model import MLP
from utils import getAzimuthElevation, lookat, matrix_from_axis_angle, euler_to_rotation_matrix


def inference(model, tracking_file, config_file, output_path, debug=False):

    # Load CSV file using Pandas
    tracking_file = args.tracking_file
    if not os.path.exists(tracking_file):
        raise FileNotFoundError(f"File {tracking_file} not found")
    df = pd.read_csv(tracking_file)

    # Load configuration parameters from the JSON file
    config_file = args.config_file
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"File {config_file} not found")
    with open(config_file, 'r') as f:
        config = json.load(f)

    focal_length = config["focal_length"]
    sensor_size = tuple(config["sensor_size"])
    image_size = tuple(config["image_size"])

    # Extract Camera Position and Orientation
    camera_position = np.array(config["camera_position"])
    roll = config["camera_orientation"][0]
    pitch = config["camera_orientation"][1]
    yaw = config["camera_orientation"][2]
    R_C2W = euler_to_rotation_matrix(roll, pitch, yaw)

    if debug:
        print(f"Camera Position: {camera_position}")
        print(f"Camera Rotation Matrix: {R_C2W}")

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
    df.to_csv(output_path, index=False)
    print(f"Inference complete. Results saved to {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default=r"cvpr\version_1", help='checkpoint path of the trained model')
    parser.add_argument('--tracking_file', type=str, default=r"data\tracking.csv", help='CSV file containing tracking data')
    parser.add_argument('--output_path', type=str, default=r'data\tracking_and_localization.csv', help='output path for the updated tracking data')
    parser.add_argument('--config_file', type=str, default=r"data\config.json", help='Path to the config.json file')
    parser.add_argument('--debug', type=bool, default=False, help='debugging mode')
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

    inference(model, args.tracking_file, args.config_file, args.output_path, args.debug)