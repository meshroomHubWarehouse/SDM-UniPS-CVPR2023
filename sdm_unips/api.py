"""
API wrapper for SDM-UniPS to use it as a library instead of command-line tool
"""

import torch
import time
from .modules.builder import builder
from .modules.io import dataio


class SDMUniPS_API:
    """
    SDM-UniPS API wrapper for programmatic usage
    """
    
    def __init__(self, 
                 checkpoint_path='checkpoint',
                 target='normal_and_brdf',
                 canonical_resolution=256,
                 pixel_samples=10000,
                 scalable=False,
                 session_name='sdm_unips_api'):
        """
        Initialize SDM-UniPS
        
        Args:
            checkpoint_path (str): Path to checkpoint directory
            target (str): What to estimate - 'normal', 'brdf', or 'normal_and_brdf'
            canonical_resolution (int): Canonical resolution for processing
            pixel_samples (int): Number of pixel samples
            scalable (bool): Use scalable processing for large images
            session_name (str): Session name for output organization
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Create args-like object
        self.args = type('Args', (), {
            'checkpoint': checkpoint_path,
            'target': target,
            'canonical_resolution': canonical_resolution,
            'pixel_samples': pixel_samples,
            'scalable': scalable,
            'session_name': session_name
        })()
        
        # Initialize the builder
        self.sdm_unips = builder.builder(self.args, self.device)
        
    def process_dataset(self,
                       test_dir,
                       max_image_res=4096,
                       max_image_num=10,
                       test_ext='.data',
                       test_prefix='L*',
                       mask_margin=8):
        """
        Process a dataset directory
        
        Args:
            test_dir (str): Directory containing test data
            max_image_res (int): Maximum image resolution
            max_image_num (int): Maximum number of images to process
            test_ext (str): File extension for dataset directories
            test_prefix (str): Prefix for image files
            mask_margin (int): Margin for mask processing
            
        Returns:
            dict: Processing results and timing information
        """
        
        # Update args for data loading
        self.args.test_dir = test_dir
        self.args.max_image_res = max_image_res
        self.args.max_image_num = max_image_num
        self.args.test_ext = test_ext
        self.args.test_prefix = test_prefix
        self.args.mask_margin = mask_margin
        
        # Create data loader
        test_data = dataio.dataio('Test', self.args)
        
        # Process
        start_time = time.time()
        self.sdm_unips.run(
            testdata=test_data,
            max_image_resolution=max_image_res,
            canonical_resolution=self.args.canonical_resolution,
        )
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        return {
            'success': True,
            'elapsed_time': elapsed_time,
            'object_name': test_data.data.objname,
            'output_path': test_data.data.data_workspace,
            'num_objects': len(test_data.objlist)
        }
    
    