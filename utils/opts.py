"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Detection Model Arguments")

    # Add basic arguments
    parser.add_argument('--cfg', type=str, default=None, help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=224, help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, default='ETHZFOOD101', help='Name of the dataset')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--tag', type=str, default='', help='Tag for the experiment')
    parser.add_argument('--eval', action='store_true', help='Enable evaluation mode')

    return parser
