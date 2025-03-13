import cv2
import pandas as pd
import torch
from ultralytics import YOLO
from tqdm import tqdm
import argparse

import torch
import torchvision


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=r'data/rendering.mkv')
    parser.add_argument("--tracking_path", type=str, default=r'data/tracking.csv')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(torch.__version__)
    print(torchvision.__version__)
    print(f"Using device: {device}")

    model = YOLO("yolo11n.pt").to(device)
    results = model.track(args.video_path, show=False, save=False, stream=True)

    # Prepare a list to store results
    tracking_data = []

    # Iterate over frames
    for frame_idx, result in enumerate(results):
        if result.boxes is not None:
            for box in result.boxes:
                track_id = box.id if box.id is not None else -1  # Some versions don't assign IDs
                x_center, y_center, width, height = box.xywhn[0].tolist()  # Get bounding box info
                
                # Append results
                tracking_data.append([int(track_id), frame_idx, x_center, y_center, width, height])

    # Convert to DataFrame
    df_tracking = pd.DataFrame(tracking_data, columns=["track_id", "frame_id", "bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height"])

    # Save to CSV
    df_tracking.to_csv("tracking_results.csv", index=False)
    print("Tracking results saved to tracking_results.csv")
    
