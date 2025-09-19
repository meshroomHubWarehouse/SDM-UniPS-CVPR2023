"""
Debug script to test SDM-UniPS API vs direct usage
Replicates main.py functionality but uses api.py
"""

from __future__ import print_function, division
import sys
import argparse
import time
import torch

sys.path.append('..')

from sdm_unips.api import SDMUniPS_API

# Argument parser - same as main.py
parser = argparse.ArgumentParser()

# Properties
parser.add_argument('--session_name', default='sdm_unips')
parser.add_argument('--target', default='normal_and_brdf', choices=['normal', 'brdf', 'normal_and_brdf'])
parser.add_argument('--checkpoint', default='checkpoint')

# Data Configuration
parser.add_argument('--max_image_res', type=int, default=4096)
parser.add_argument('--max_image_num', type=int, default=10)
parser.add_argument('--test_ext', default='.data')
parser.add_argument('--test_dir', default='DefaultTest')
parser.add_argument('--test_prefix', default='L*')
parser.add_argument('--mask_margin', type=int, default=8)

# Network Configuration
parser.add_argument('--canonical_resolution', type=int, default=256)
parser.add_argument('--pixel_samples', type=int, default=10000)
parser.add_argument('--scalable', action='store_true')


def main():
    args = parser.parse_args()
    print(f'\nStarting a session: {args.session_name}')
    print(f'target: {args.target}\n')
    
    # Initialize SDM-UniPS API instead of direct builder
    sdm_api = SDMUniPS_API(
        checkpoint_path=args.checkpoint,
        target=args.target,
        canonical_resolution=args.canonical_resolution,
        pixel_samples=args.pixel_samples,
        scalable=args.scalable,
        session_name=args.session_name
    )

    start_time = time.time()
    
    # Process dataset using API
    result = sdm_api.process_dataset(
        test_dir=args.test_dir,
        max_image_res=args.max_image_res,
        max_image_num=args.max_image_num,
        test_ext=args.test_ext,
        test_prefix=args.test_prefix,
        mask_margin=args.mask_margin
    )
    
    end_time = time.time()
    
    if result['success']:
        print(f"Prediction finished (Elapsed time is {result['elapsed_time']:.3f} sec)")
        print(f"Output saved to: {result['output_path']}")
        print(f"Object name: {result['object_name']}")
        print(f"Number of objects processed: {result['num_objects']}")
        print("\nExecute the following script to render a video under new lighting conditions based on the generated BRDF and normal map.\n")
        print(f"        python sdm_unips/relighting.py --datadir {result['output_path']}\n")
    else:
        print("Processing failed!")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())