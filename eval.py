#!/usr/bin/venv python
# -*- coding: utf-8 -*-
#
import os
import time
import torch
import argparse

import smplx
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import sys

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(current_file_path))
from lib.config import config, constant
from lib.network.net import FoRMNet
from lib.dataset.dataset import DatasetWindowFoRM
from lib.math.angular import r6d_to_rotation_matrix, rotation_matrix_to_axis_angle
from lib.eval.eval_pose import PoseEvaluator


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DataPreparation:
    @staticmethod
    def prepare_dataloader(dataset_dir, windows_size, batch_size, shuffle, window_test,
                           split_strategy='specify_file:specify', valid_config=None, test_config=None,):
        # Prepare the dataset
        split_params = ('forces_raw', 'forces_4x2',
                        'imu_raw_acc', 'imu_raw_gyro',
                        'gt_poses', 'gt_trans')
        return_params = split_params + ('gt_betas', 'framerate')
        logger.info(f'Windows Size: {windows_size}')
        
        # Main Experiment Split
        _v2_valid_actions = ['S01_FreeMove', 'S01_FrontArmRaise', 'S02_CrissCross', 'S02_LateralArmRaise',
                             'S03_BackStep', 'S03_ChestElbowTwist', 'S04_StraightStep', 'S04_LateralArmTwist',
                             'S05_SquareStep', 'S05_LeftKneeLift', 'S06_SideStep', 'S06_RightKneeLift',
                             'S07_Running', 'S07_LeftLunge', 'S08_ObstacleCircling', 'S08_RightLunge',
                             'S09_JumpJacks', 'S09_SquatStand', 'S10_CrissCross', 'S10_Boxing',
                             'S11_FreeMove', 'S11_ChairSitStand', 'S12_BackStep', 'S12_DoorOpening',
                             'S13_SquareStep', 'S13_LeftKneeLift', 'S14_SideStep', 'S14_RightKneeLift',
                             'S15_Running', 'S15_LeftLunge', 'S16_ObstacleCircling', 'S16_RightLunge',
                             'S17_JumpJacks', 'S17_SquatStand', 'S18_CrissCross', 'S18_Boxing',
                             ]
        _v2_test_actions = None
        
        if split_strategy == 'specify_file:specify' and valid_config is None and test_config is None:
            valid_config, test_config = _v2_valid_actions, _v2_test_actions
        if test_config is None:
            test_config = valid_config
        
        dataset_test = DatasetWindowFoRM(
            dataset_dir=dataset_dir, windows_length=windows_size,
            split_strategy=split_strategy, valid_config=valid_config, test_config=test_config,
            used_param=('forces_raw', 'forces_4x2',
                        'imu_raw_acc', 'imu_raw_gyro',
                        'gt_poses', 'gt_betas', 'gt_trans', 'gt_gender',
                        'framerate', 'vid'),
            splt_param=split_params,
            out_param=return_params,
            window_test=window_test,
        )
        
        test_dataset = dataset_test
        test_dataset.choose_data_type('valid')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return test_loader


class Evaluator:
    def __init__(self, **kwargs):
        self.device = kwargs["device"]
        
        self.ckpt_path = config.PATH_CHECKPOINTS
        self.ckpt_title = kwargs["ckpt_title"]
        
        # Gender of SMPL model is 'neutral' as default
        self.smpl = smplx.create(model_type='smpl', model_path=config.PATH_MODEL_SMPL, gender="neutral").to(self.device)
        
        # Load the model
        self.model = FoRMNet(16, 24, 48, 23, 6, 3, 'LSTM', 2, 512, self.smpl)
        self._load_model_weights()
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from {self.ckpt_title}")
        
        # Prepare the dataloader
        data_preparation = DataPreparation()
        self.test_loader = data_preparation.prepare_dataloader(
            dataset_dir=kwargs["dataset_dir"],
            windows_size=kwargs["window_size"],
            batch_size=kwargs["batch_size"],
            shuffle=False,
            window_test=kwargs["window_test"],
            split_strategy=kwargs["split_strategy"],
            valid_config=kwargs["valid_config"],
            test_config=kwargs["test_config"],
        )
    
    def _load_model_weights(self):
        """Load the model weights from the checkpoint path."""
        path_weight = os.path.join(self.ckpt_path, "FoRM.pth")
        
        if os.path.exists(path_weight):
            logger.info(f"Loading model weights from {path_weight}")
            model_state_dict = torch.load(path_weight, map_location=self.device)
            self.model.shoe_poser.load_state_dict(model_state_dict['shoe_poser'])
            self.model.shoe_rooter.load_state_dict(model_state_dict['shoe_rooter'])
            self.model.shoe_yawer.load_state_dict(model_state_dict['shoe_yawer'])
        else:
            logger.warning(f"Model weights not found at {path_weight}")

    def _prepare_data_dict(self, data_set, device=torch.device('cpu')):
        """
        Convert the data in the dataset to the format required by the model.
        Args:
            data_set: Raw dataset, diction.
            device: Device to which the data is moved.
        Returns:
            new_data_set: The data in the format required by the model.
        """
        new_data_set = dict()
        
        batch_size, window_size, _ = data_set['gt_trans'].shape
        # Process the input data
        new_data_set['measure'] = torch.cat([
            data_set['forces_4x2'].view(batch_size, window_size, -1),
            data_set['imu_raw_acc'].view(batch_size, window_size, -1),
            data_set['imu_raw_gyro'].view(batch_size, window_size, -1),
            ],
            dim=-1).detach().clone().float().requires_grad_(False).to(device)
        
        # Additional inputs for ablation study, only used during development
        new_data_set['addition'] = torch.zeros(batch_size, window_size, 48).detach().clone().float().requires_grad_(False).to(device)
        
        # Process the ground truth data
        for key, value in data_set.items():
            if 'gt_' in key:
                if 'betas' in key:
                    new_data_set[key] = value.unsqueeze(1).repeat(1, window_size, 1).detach().clone().float().requires_grad_(False).to(device)
                else:
                    new_data_set[key] = value.detach().clone().float().requires_grad_(False).to(device)
        
        new_data_set['framerate'] = data_set['framerate'].detach().clone().float().requires_grad_(False).to(device)

        return new_data_set

    @staticmethod
    def _root_ori_to_theta(root_ori):
        """
        Convert the root orientation to axis-angle format, maintaining the batch_size and window_size.
        Args:
            root_ori: [batch_size, window_size, 6], root orientation in 6D format
        Returns:
            root_theta: [batch_size, window_size, 3], root angle
        """
        batch_size, window_size = root_ori.shape[:2]
        root_ori = root_ori.reshape(-1, 6)

        root_theta = rotation_matrix_to_axis_angle(r6d_to_rotation_matrix(root_ori)).view(batch_size, window_size, 3)
        return root_theta
    
    def _run_model(self, batch_data):
        """
        Run the model and get the prediction results.
        Args:
            batch_data: Batched data dictionary.
        Returns:
            pred_dict: Prediction results dictionary
                pred_dict["poses"]: [batch_size, window_size, 72], root orientation and body pose in 6D format
                pred_dict["trans"]: [batch_size, window_size, 3], root translation
            gt_dict: Ground truth results dictionary
                gt_dict["poses"]: [batch_size, window_size, 72], root orientation and body pose in 6D format
                gt_dict["trans"]: [batch_size, window_size, 3], root translation
                gt_dict["betas"]: [batch_size, window_size, 10], SMPL shape parameters
        """
        batch4network = self._prepare_data_dict(batch_data, self.device)
        
        # Run the model
        pred_result = self.model(batch4network["measure"],
                                 batch4network["addition"],
                                 batch4network["gt_poses"],
                                 batch4network["gt_trans"],
                                 batch4network["framerate"],
                                 batch4network["gt_betas"].detach().clone().float().requires_grad_(False))
        
        # Extract the prediction results
        pred_dict = {
            "poses": torch.cat([
                self._root_ori_to_theta(pred_result["pred_root_ori_refine"]),
                pred_result["pred_body_poses"],
            ], dim=-1),
            "trans": pred_result["pred_root_trans_refine"],
        }
        
        # Extract the ground truth results
        gt_dict = {
            "poses": batch4network["gt_poses"],
            "trans": batch4network["gt_trans"],
            "betas": batch4network["gt_betas"],
            "framerate": batch4network["framerate"],
        }
        
        return pred_dict, gt_dict
    
    def _update_result(self, result_dict, pred_dict, gt_dict):
        """
        Update the result dictionary.
        Args:
            result_dict: The result dictionary to be updated, storing the prediction and ground truth results.
            pred_dict: The prediction results dictionary for current batch.
            gt_dict: The ground truth results dictionary for current batch.
        Returns:
            result_dict: The updated result dictionary.
        """
        for key, value in pred_dict.items():
            result_dict['pred_'+key].append(value.clone().detach().cpu())

        for key, value in gt_dict.items():
            if key == 'framerate':
                continue
            result_dict['gt_' + key].append(value.clone().detach().cpu())
            
        return result_dict
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate the model and store the result."""
        logger.info("Starting evaluation...")
        result_dict = {
            "pred_poses": [], "pred_trans": [],
            "gt_poses": [], "gt_trans": [], "gt_betas": [],
        }
        
        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
            # Run the model
            pred_dict, gt_dict = self._run_model(batch)
            
            # Update the result dictionary
            result_dict = self._update_result(result_dict, pred_dict, gt_dict)
        
        # # Store the result in a file
        # result_file = os.path.join(self.result_path, f"{self.ckpt_title}_eval_result.pth")
        # logger.info(f"Saving evaluation results to {result_file}")
        # torch.save(result_dict, result_file)
        
        logger.info("Evaluation completed, starting computing error...")
        eval_error = PoseEvaluator(device=self.device, smpl_model=self.smpl)  
        err = eval_error.evaluate_batch(result_dict)
        for key, value in err.items():
            logger.info(f"{key}: {value:.4f}")
        
        return result_dict
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    # Dataset and dataloader configuration
    parser.add_argument("--dataset_dir", type=str, 
                        default=config.PATH_IMPLY_DATASET,
                        help="Dataset directory")
    parser.add_argument("--window_test", action="store_true",
                        help="Whether to use window test")
    parser.add_argument("--window_size", type=int, default=120,
                        help="Window size")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--split_strategy", type=str, default="specify_file:specify",
                        help="Dataset split strategy")
    parser.add_argument("--valid_config", type=str, default=None,
                        help="Validation set configuration, don't modify if you want to eval the paper result")
    parser.add_argument("--test_config", type=str, default=None,
                        help="Test set configuration, don't modify if you want to eval the paper result")
    
    # Model configuration
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on")
    parser.add_argument("--ckpt_title", type=str, default="FoRM",
                        help="Checkpoint title")
    
    args = parser.parse_args()
    
    # Set random seed here
    set_seed(args.seed)
    
    # Evaluate the FoRMNet model
    evaluator = Evaluator(**vars(args))
    evaluator.evaluate()
