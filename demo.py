#!/usr/bin/venv python
# -*- coding: utf-8 -*-
#
"""
FoRM Demo Script
Single sample inference, read preprocessed data and output SMPL parameters

Input data format:
{
    "forces_4x2": [window_size, 8],     # Necessary
    "imu_raw_acc": [window_size, 6],    # Necessary
    "imu_raw_gyro": [window_size, 6],   # Necessary
    "gt_betas": [10],                   # Optional, default zeros
    "framerate": scalar                 # Optional, default 60
}
Output data format:
{
    "betas": [10],                      # SMPL shape parameters
    "poses": [24],                      # SMPL pose parameters
    "trans": [3]                        # SMPL translation parameters
    "framerate": scalar                 # Frame rate
}
"""
import os
import torch
import argparse
import smplx
import numpy as np
from loguru import logger
import sys

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(current_file_path))
from lib.config import config
from lib.network.net import FoRMNet
from lib.math.angular import r6d_to_rotation_matrix, rotation_matrix_to_axis_angle


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FoRMDemo:
    """FoRM Inference Demo"""
    def __init__(self, device='cuda:0', ckpt_path=None, smpl_model_path=None):
        """
        Args:
            device: Running device ('cuda:0' or 'cpu')
            ckpt_path: Model checkpoint path
            smpl_model_path: SMPL model path
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set model checkpoint path
        self.ckpt_path = ckpt_path if ckpt_path else config.PATH_CHECKPOINTS
        self.smpl_model_path = smpl_model_path if smpl_model_path else config.PATH_MODEL_SMPL
        
        # Load SMPL model
        self.smpl = smplx.create(
            model_type='smpl', 
            model_path=self.smpl_model_path, 
            gender="neutral"
        ).to(self.device)
        logger.info("SMPL model loaded")
        
        # Load FoRMNet model
        self.model = FoRMNet(16, 24, 48, 23, 6, 3, 'LSTM', 2, 512, self.smpl)
        self._load_model_weights()
        self.model.to(self.device)
        self.model.eval()
        logger.info("FoRM model loaded and set to evaluation mode")
    
    def _load_model_weights(self):
        """Load the model weights from the checkpoint path."""
        path_weight = os.path.join(self.ckpt_path, "FoRM.pth")
        
        if not os.path.exists(path_weight):
            raise FileNotFoundError(f"Model weights not found at {path_weight}")
        
        logger.info(f"Loading model weights from {path_weight}")
        model_state_dict = torch.load(path_weight, map_location=self.device)
        self.model.shoe_poser.load_state_dict(model_state_dict['shoe_poser'])
        self.model.shoe_rooter.load_state_dict(model_state_dict['shoe_rooter'])
        self.model.shoe_yawer.load_state_dict(model_state_dict['shoe_yawer'])
        logger.info("Model weights loaded successfully")

    @staticmethod
    def load_preprocessed_data(data_path):
        """Load preprocessed data from a file

        Args:
            data_path: Preprocessed data file path (.pth, .pt, .npz, etc.)
        
        Returns:
            data_dict: Dictionary containing the following keys
                - forces_4x2: [window_size, 8] Force sensor data
                - imu_raw_acc: [window_size, 6] IMU accelerometer data
                - imu_raw_gyro: [window_size, 6] IMU gyroscope data
                - gt_betas: [10] SMPL shape parameters (optional)
                - framerate: Scalar, frame rate (optional, default 60 fps)
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        logger.info(f"Loading preprocessed data from {data_path}")

        # Load data based on file extension
        ext = os.path.splitext(data_path)[1]
        if ext in ['.pth', '.pt']:
            data = torch.load(data_path)
        elif ext == '.npz':
            data_npz = np.load(data_path)
            data = {key: torch.from_numpy(data_npz[key]) for key in data_npz.files}
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .pth, .pt, or .npz")
        
        # Validate necessary keys
        required_keys = ['forces_4x2', 'imu_raw_acc', 'imu_raw_gyro']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Set default values for optional keys
        if 'gt_betas' not in data:
            logger.warning("gt_betas not found, using default zero betas")
            data['gt_betas'] = torch.zeros(10)
        if 'framerate' not in data:
            logger.warning("framerate not found, using default 60 fps")
            data['framerate'] = torch.tensor(60.0)
        
        logger.info(f"Data loaded: window_size={data['forces_4x2'].shape[0]}")
        return data
    
    def _prepare_input(self, data_dict):
        """
        Prepare input data for the model
        
        Args:
            data_dict: Raw data dictionary
        
        Returns:
            input_dict: Model input dictionary
        """
        # Get window size
        window_size = data_dict['forces_4x2'].shape[0]

        # Prepare input data (batch_size=1)
        input_dict = {}
        
        # Concatenate sensor data: forces(8) + acc(6) + gyro(6) = 20 dimensions
        input_dict['measure'] = torch.cat([
            data_dict['forces_4x2'].view(1, window_size, -1),
            data_dict['imu_raw_acc'].view(1, window_size, -1),
            data_dict['imu_raw_gyro'].view(1, window_size, -1),
        ], dim=-1).float().to(self.device)

        # Additional input (for ablation study, default is zero)
        input_dict['addition'] = torch.zeros(1, window_size, 48).float().to(self.device)

        # SMPL shape parameters
        input_dict['gt_betas'] = data_dict['gt_betas'].unsqueeze(0).unsqueeze(0).repeat(
            1, window_size, 1
        ).float().to(self.device)
        
        # Framerate
        input_dict['framerate'] = data_dict['framerate'].unsqueeze(0).float().to(self.device)
        
        # Initial pose and translation (initialized to zero)
        input_dict['gt_poses'] = torch.zeros(1, window_size, 72).float().to(self.device)
        input_dict['gt_trans'] = torch.zeros(1, window_size, 3).float().to(self.device)
        
        return input_dict
    
    @staticmethod
    def _root_ori_to_theta(root_ori):
        """
        Convert root orientation from 6D format to axis-angle format

        Args:
            root_ori: [batch_size, window_size, 6] 6D rotation representation
        
        Returns:
            root_theta: [batch_size, window_size, 3] axis-angle representation
        """
        batch_size, window_size = root_ori.shape[:2]
        root_ori = root_ori.reshape(-1, 6)
        root_theta = rotation_matrix_to_axis_angle(
            r6d_to_rotation_matrix(root_ori)
        ).view(batch_size, window_size, 3)
        return root_theta
    
    @torch.no_grad()
    def inference(self, data_dict):
        """
        Args:
            data_dict: Input data dictionary
        
        Returns:
            result_dict: Inference result dictionary
                - poses: [window_size, 72] SMPL pose parameters (axis-angle format)
                - trans: [window_size, 3] SMPL root translation
                - betas: [10] SMPL shape parameters
                - framerate: scalar, frame rate
        """
        logger.info("Starting inference...")
        
        # Prepare input
        input_dict = self._prepare_input(data_dict)
        
        # Run model
        pred_result = self.model(
            input_dict["measure"],
            input_dict["addition"],
            input_dict["gt_poses"],
            input_dict["gt_trans"],
            input_dict["framerate"],
            input_dict["gt_betas"]
        )
        
        # Extract prediction results
        pred_poses = torch.cat([
            self._root_ori_to_theta(pred_result["pred_root_ori_refine"]),
            pred_result["pred_body_poses"],
        ], dim=-1)  # [1, window_size, 72]
        
        pred_trans = pred_result["pred_root_trans_refine"]  # [1, window_size, 3]

        # Remove batch dimension and move to CPU
        result_dict = {
            "poses": pred_poses.squeeze(0).cpu(),  # [window_size, 72]
            "trans": pred_trans.squeeze(0).cpu(),  # [window_size, 3]
            "betas": data_dict['gt_betas'].cpu(),  # [10]
            "framerate": data_dict['framerate'].cpu().item(),  # scalar
        }
        
        logger.info(f"Inference completed: {result_dict['poses'].shape[0]} frames")
        return result_dict

    @staticmethod
    def save_results(result_dict, output_path):
        """
        Args:
            result_dict: Inference result dictionary
            output_path: Output file path
        """
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save results
        torch.save(result_dict, output_path)
        logger.info(f"Results saved to {output_path}")
    
    def run(self, input_path, output_path):
        """
        Complete inference pipeline: load data -> inference -> save results

        Args:
            input_path: Input data file path
            output_path: Output result file path
        
        Returns:
            result_dict: Inference result dictionary
        """
        # Load data
        data_dict = self.load_preprocessed_data(input_path)
        
        # Run inference
        result_dict = self.inference(data_dict)
        
        # Save results
        self.save_results(result_dict, output_path)
        
        return result_dict


def main():
    parser = argparse.ArgumentParser(description="FoRM Demo - Single Sample Inference")
    
    # Input/output paths
    parser.add_argument("--input", type=str, required=True,
                        help="Input preprocessed data file path (.pth, .pt, .npz)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output SMPL parameters file path (.pth)")
    
    # Model configuration
    parser.add_argument("--device", type=str, 
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="FoRMNet model checkpoint path")
    parser.add_argument("--smpl_path", type=str, default=None,
                        help="SMPL model path")
    
    # Other configurations
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    # ======== Inference Pipeline ========
    set_seed(args.seed)

    demo = FoRMDemo(
        device=args.device,
        ckpt_path=args.ckpt_path,
        smpl_model_path=args.smpl_path
    )
    result = demo.run(args.input, args.output)
    
    # Print summary
    logger.info("=" * 50)
    logger.info("Inference Summary:")
    logger.info(f"  Frames: {result['poses'].shape[0]}")
    logger.info(f"  Poses shape: {result['poses'].shape}")
    logger.info(f"  Trans shape: {result['trans'].shape}")
    logger.info(f"  Betas shape: {result['betas'].shape}")
    logger.info(f"  Framerate: {result['framerate']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    # Basic usage:
    #   python demo.py --input /path/to/input_data.pth --output /path/to/output_result.pth
    # Advanced usage:
    #   python demo.py \
    #     --input /path/to/input_data.pth \
    #     --output /path/to/output_result.pth \
    #     --ckpt_path /path/to/checkpoints \
    #     --smpl_path /path/to/smpl_models
    main()
