import pandas as pd
from scipy.spatial import cKDTree
import argparse

def match(gt_path, track_path, output_path):

    # Load CSV files
    gt_df = pd.read_csv(gt_path)
    track_df = pd.read_csv(track_path)

    # Create KDTree for fast nearest neighbor lookup
    gt_tree = cKDTree(gt_df[['x', 'y']].values)

    # Find the nearest ground truth match for each tracking entry
    distances, indices = gt_tree.query(track_df[['x_world_new', 'y_world_new']].values)

    # Assign correct track_id based on the nearest ground truth
    track_df['track_id'] = gt_df.iloc[indices]['track_id'].values

    # Save the corrected tracking data
    track_df.to_csv(output_path, index=False)

    print(f"Matched tracking results saved to {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, default=r'data\ground_truth.csv')
    parser.add_argument("--track_path", type=str, default=r'data\tracking_and_localization.csv')
    parser.add_argument("--output_path", type=str, default=r'data\tracking_and_localization_matched.csv')
    args = parser.parse_args()

    match(args.gt_path, args.track_path, args.output_path)
