import os
import torch
import smplx
import numpy as np
from tqdm import trange
from loguru import logger

import sys
current_file_path = os.path.abspath(__file__)
PATH_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if PATH_PROJECT_ROOT not in sys.path:
    sys.path.append(PATH_PROJECT_ROOT)

from lib.config import config, constant
from lib.math.angular import axis_angle_to_rotation_matrix
from lib.eval.utils import batch_compute_similarity_transform_torch


class PoseEvaluator:
    def __init__(self, device='cuda:0', smpl_model=None):
        """
        Initialize the PoseEvaluator.
        
        """
        self.device = device
        self._m2mm = 1000  # meter to millimeter
        
        # initialize SMPL model
        if smpl_model is None:
            self.smpl = smplx.create(
                config.PATH_MODEL_SMPL, type='smpl', gender='neutral').to(self.device)
        else:
            self.smpl = smpl_model.to(self.device)
    
    def _get_smpl_params(self, smpl_params):
        """
        Get SMPL joints and vertices from SMPL parameters.
        
        Args:
            smpl_params: SMPL parameter dictionary, including poses, trans, betas
        
        Returns:
            joints: (B, N, 24, 3) Joints position (m)
            vertices: (B, N, 6890, 3) Vertices position (m)
        """
        # Ensure all inputs are on the same device as SMPL model
        device = self.device
        batch_size, seq_len = smpl_params['poses'].shape[:2]
        
        smpl_output = self.smpl(
            betas=smpl_params['betas'].view(-1, 10).to(device),
            body_pose=smpl_params['poses'][:, :, 3: 72].view(-1, 69).to(device),
            global_orient=smpl_params['poses'][:, :, :3].view(-1, 3).to(device),
            transl=smpl_params['trans'].view(-1, 3).to(device),
            return_verts=True,
        )

        joints = smpl_output.joints.view(batch_size, seq_len, -1, 3)
        vertices = smpl_output.vertices.view(batch_size, seq_len, -1, 3)

        return joints[:, :, :24, :], vertices
    
    # ----------------- Stability Metrics ----------------- #
    @staticmethod
    def _compute_acceleration(joints, frame_rate=60):
        """
        Calculate acceleration of joints.
        
        Args:
            joints: (B, N, 24, 3) joints location (m)
            frame_rate: frame rate (Hz)
        
        Returns:
            acceleration: (B, N-2, 24, 3) Acceleration (m/s^2)
        """
        dt = 1 / frame_rate
        velocity = (joints[:, 1:] - joints[:, :-1]) / dt
        acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt
        return acceleration
    
    @staticmethod
    def _compute_jitter(joints):
        """
        Calculate jitter of joints.
        
        Args:
            joints: (B, N, 24, 3) Joints location (m)
        
        Returns:
            jitter: average jitter value (m/s^3)
        """
        jerk = torch.diff(joints, n=3, dim=-2)
        return torch.mean(torch.norm(jerk, dim=-1))
    
    def compute_mpjpe(self, pred_pose, gt_pose, pred_trans, gt_trans, betas, with_alignment=False):
        """
        Calculate MPJPE (Mean Per Joint Position Error)
        
        Args:
            pred_pose: (B, N, 72) predicted pose parameters
            gt_pose: (B, N, 72) ground truth pose parameters
            pred_trans: (B, N, 3) predicted translation parameters
            gt_trans: (B, N, 3) ground truth translation parameters
            betas: (B, N, 10) shape parameters
            with_alignment: whether to perform Procrustes alignment
        
        Returns:
            mpjpe: MPJPE value (mm)
        """
        # 将数据移至指定设备
        pred_pose = pred_pose.clone().detach().to(self.device)
        gt_pose = gt_pose.clone().detach().to(self.device)
        pred_trans = pred_trans.clone().detach().to(self.device)
        gt_trans = gt_trans.clone().detach().to(self.device)
        betas = betas.clone().detach().to(self.device)
        
        # 获取关节位置
        if with_alignment:
            # 对于PA-MPJPE，不考虑平移
            joints_pred, _ = self._get_smpl_params(
                smpl_params={'poses': pred_pose, 'trans': torch.zeros_like(pred_trans), 'betas': betas}
            )
            joints_gt, _ = self._get_smpl_params(
                smpl_params={'poses': gt_pose, 'trans': torch.zeros_like(gt_trans), 'betas': betas}
            )
            
            # 进行Procrustes对齐
            S1_hat = batch_compute_similarity_transform_torch(
                joints_pred.reshape(-1, 24, 3), joints_gt.reshape(-1, 24, 3)
            )
            mpjpe = self._m2mm * torch.mean(torch.norm(joints_gt.reshape(-1, 24, 3) - S1_hat, dim=-1))
        else:
            # 对于MPJPE或W-MPJPE，考虑平移
            joints_pred, _ = self._get_smpl_params(
                smpl_params={'poses': pred_pose, 'trans': pred_trans, 'betas': betas}
            )
            joints_gt, _ = self._get_smpl_params(
                smpl_params={'poses': gt_pose, 'trans': gt_trans, 'betas': betas}
            )
            
            mpjpe = self._m2mm * torch.mean(torch.norm(joints_pred - joints_gt, dim=-1))
        
        return mpjpe
    
    def compute_mpvpe(self, pred_pose, gt_pose, pred_trans, gt_trans, betas):
        """
        Calculate MPVPE (Mean Per Vertex Position Error)
        
        Args:
            pred_pose: (B, N, 72) predicted pose parameters
            gt_pose: (B, N, 72) ground truth pose parameters
            pred_trans: (B, N, 3) predicted translation parameters
            gt_trans: (B, N, 3) ground truth translation parameters
            betas: (B, N, 10) shape parameters
        
        Returns:
            mpvpe: MPVPE value (mm)
        """
        # Move data to specified device
        pred_pose = pred_pose.clone().detach().to(self.device)
        gt_pose = gt_pose.clone().detach().to(self.device)
        pred_trans = pred_trans.clone().detach().to(self.device)
        gt_trans = gt_trans.clone().detach().to(self.device)
        betas = betas.clone().detach().to(self.device)
        
        # Get vertex positions
        _, verts_pred = self._get_smpl_params(
            smpl_params={'poses': pred_pose, 'trans': pred_trans, 'betas': betas}
        )
        _, verts_gt = self._get_smpl_params(
            smpl_params={'poses': gt_pose, 'trans': gt_trans, 'betas': betas}
        )
        
        mpvpe = self._m2mm * torch.mean(torch.norm(verts_pred - verts_gt, dim=-1))
        return mpvpe
    
    def compute_mpjre(self, pred_pose, gt_pose):
        """
        Calculate MPJRE (Mean Per Joint Rotation Error)
        
        Args:
            pred_pose: (B, N, 72) predicted pose parameters
            gt_pose: (B, N, 72) ground truth pose parameters
        
        Returns:
            mpjre: MPJRE value (degrees)
        """
        # Move data to specified device
        pred_pose = pred_pose.clone().detach().to(self.device)
        gt_pose = gt_pose.clone().detach().to(self.device)
        
        batch_size, seq_len, _ = pred_pose.shape
        
        # Convert axis-angle representation to rotation matrices
        rot_mat_pred = axis_angle_to_rotation_matrix(pred_pose).view(batch_size, seq_len, 24, 3, 3)
        rot_mat_gt = axis_angle_to_rotation_matrix(gt_pose).view(batch_size, seq_len, 24, 3, 3)

        # Initialize global rotation matrices
        global_rot_mat_pred = torch.eye(3).to(pred_pose.device).unsqueeze(0).repeat(batch_size, seq_len, 24, 1, 1)
        global_rot_mat_gt = torch.eye(3).to(gt_pose.device).unsqueeze(0).repeat(batch_size, seq_len, 24, 1, 1)

        # Initialize rotation error accumulation variable
        total_rotation_error = 0.0

        # SMPL parent joint indices
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

        # Calculate global rotation matrices
        for i in range(24):
            if parents[i] == -1:  # Root joint
                global_rot_mat_pred[:, :, i] = rot_mat_pred[:, :, i]
                global_rot_mat_gt[:, :, i] = rot_mat_gt[:, :, i]
            else:
                parent = parents[i]
                global_rot_mat_pred[:, :, i] = torch.matmul(global_rot_mat_pred[:, :, parent], rot_mat_pred[:, :, i])
                global_rot_mat_gt[:, :, i] = torch.matmul(global_rot_mat_gt[:, :, parent], rot_mat_gt[:, :, i])

        # Calculate rotation error for each joint
        for i in range(24):
            R_diff = torch.matmul(global_rot_mat_pred[:, :, i].transpose(-1, -2), global_rot_mat_gt[:, :, i])
            angles = torch.acos(
                torch.clamp((R_diff[:, :, 0, 0] + R_diff[:, :, 1, 1] + R_diff[:, :, 2, 2] - 1) / 2, -1 + 1e-6, 1 - 1e-6)
            )
            total_rotation_error += torch.mean(angles) * 180 / np.pi

        mpjre = total_rotation_error / 24  # Average rotation error per joint
        return mpjre
    
    def compute_rte(self, pred_trans, gt_trans):
        """
        Calculate RTE (Root Translation Error)
        
        Args:
            pred_trans: (B, N, 3) predicted translation parameters
            gt_trans: (B, N, 3) ground truth translation parameters
        
        Returns:
            rte: RTE value (meters)
        """
        # Move data to specified device
        pred_trans = pred_trans.clone().detach().to(self.device)
        gt_trans = gt_trans.clone().detach().to(self.device)
        
        # Calculate difference between trajectory endpoint and startpoint
        loc_pred = pred_trans[:, -1] - pred_trans[:, 0]
        loc_gt = gt_trans[:, -1] - gt_trans[:, 0]
        
        # Calculate error
        rte = torch.mean(torch.norm(loc_pred - loc_gt, dim=-1))
        return rte
    
    def compute_accel_err(self, pred_pose, gt_pose, pred_trans, gt_trans, betas, frame_rate=60):
        """
        Calculate acceleration error
        
        Args:
            pred_pose: (B, N, 72) predicted pose parameters
            gt_pose: (B, N, 72) ground truth pose parameters
            pred_trans: (B, N, 3) predicted translation parameters
            gt_trans: (B, N, 3) ground truth translation parameters
            betas: (B, N, 10) shape parameters
            frame_rate: frame rate
        
        Returns:
            accel_err: acceleration error (m/s^2)
        """
        # Move data to specified device
        pred_pose = pred_pose.clone().detach().to(self.device)
        gt_pose = gt_pose.clone().detach().to(self.device)
        pred_trans = pred_trans.clone().detach().to(self.device)
        gt_trans = gt_trans.clone().detach().to(self.device)
        betas = betas.clone().detach().to(self.device)
        
        # Get joint positions
        joints_pred, _ = self._get_smpl_params(
            smpl_params={'poses': pred_pose, 'trans': pred_trans, 'betas': betas}
        )
        joints_gt, _ = self._get_smpl_params(
            smpl_params={'poses': gt_pose, 'trans': gt_trans, 'betas': betas}
        )
        
        # Calculate acceleration
        accel_pred = self._compute_acceleration(joints_pred, frame_rate)
        accel_gt = self._compute_acceleration(joints_gt, frame_rate)
        
        # Calculate acceleration error
        accel_err = torch.mean(torch.norm(accel_pred - accel_gt, dim=-1))
        return accel_err
    
    def compute_jitter(self, pred_pose, gt_pose, pred_trans, gt_trans, betas):
        """
        Calculate jitter ratio
        
        Args:
            pred_pose: (B, N, 72) predicted pose parameters
            gt_pose: (B, N, 72) ground truth pose parameters
            pred_trans: (B, N, 3) predicted translation parameters
            gt_trans: (B, N, 3) ground truth translation parameters
            betas: (B, N, 10) shape parameters
        
        Returns:
            jitter: jitter ratio
        """
        # Move data to specified device
        pred_pose = pred_pose.clone().detach().to(self.device)
        gt_pose = gt_pose.clone().detach().to(self.device)
        pred_trans = pred_trans.clone().detach().to(self.device)
        gt_trans = gt_trans.clone().detach().to(self.device)
        betas = betas.clone().detach().to(self.device)
        
        # Get joint positions
        joints_pred, _ = self._get_smpl_params(
            smpl_params={'poses': pred_pose, 'trans': pred_trans, 'betas': betas}
        )
        joints_gt, _ = self._get_smpl_params(
            smpl_params={'poses': gt_pose, 'trans': gt_trans, 'betas': betas}
        )
        
        # Calculate jitter ratio
        jitter = self._compute_jitter(joints_pred) / self._compute_jitter(joints_gt)
        return jitter
    
    @torch.no_grad()
    def evaluate_sequence(self, pred_pose, gt_pose, pred_trans, gt_trans, betas, frame_rate=60):
        """
        Evaluate the predicted sequence against the ground truth sequence.
        
        Args:
            pred_pose: (B, N, 72) predicted pose parameters
            gt_pose: (B, N, 72) ground truth pose parameters
            pred_trans: (B, N, 3) predicted translation parameters
            gt_trans: (B, N, 3) ground truth translation parameters
            betas: (B, N, 10) shape parameters
            frame_rate: frame rate

        Returns:
            metrics: a dictionary containing all evaluation metrics
        """
        # Pre-compute SMPL outputs to avoid redundant computation
        # With translation
        joints_pred_with_trans, verts_pred_with_trans = self._get_smpl_params(
            smpl_params={'poses': pred_pose, 'trans': pred_trans, 'betas': betas}
        )
        joints_gt_with_trans, verts_gt_with_trans = self._get_smpl_params(
            smpl_params={'poses': gt_pose, 'trans': gt_trans, 'betas': betas}
        )
        
        # Without translation
        zero_trans_pred = torch.zeros_like(pred_trans, device=self.device)
        zero_trans_gt = torch.zeros_like(gt_trans, device=self.device)
        joints_pred_no_trans, verts_pred_no_trans = self._get_smpl_params(
            smpl_params={'poses': pred_pose, 'trans': zero_trans_pred, 'betas': betas}
        )
        joints_gt_no_trans, verts_gt_no_trans = self._get_smpl_params(
            smpl_params={'poses': gt_pose, 'trans': zero_trans_gt, 'betas': betas}
        )
        
        # Calculate PA-MPJPE (Procrustes-aligned MPJPE)
        S1_hat = batch_compute_similarity_transform_torch(
            joints_pred_no_trans.reshape(-1, 24, 3), joints_gt_no_trans.reshape(-1, 24, 3)
        )
        pa_mpjpe = self._m2mm * torch.mean(torch.norm(joints_gt_no_trans.reshape(-1, 24, 3) - S1_hat, dim=-1))
        
        # Calculate MPJPE (without translation)
        mpjpe = self._m2mm * torch.mean(torch.norm(joints_pred_no_trans - joints_gt_no_trans, dim=-1))
        
        # Calculate MPVPE (without translation)
        mpvpe = self._m2mm * torch.mean(torch.norm(verts_pred_no_trans - verts_gt_no_trans, dim=-1))
        
        # Calculate W-MPJPE (with translation)
        w_mpjpe = self._m2mm * torch.mean(torch.norm(joints_pred_with_trans - joints_gt_with_trans, dim=-1))
        
        # Calculate W-MPVPE (with translation)
        w_mpvpe = self._m2mm * torch.mean(torch.norm(verts_pred_with_trans - verts_gt_with_trans, dim=-1))
        
        # Calculate MPJRE
        mpjre = self.compute_mpjre(pred_pose, gt_pose)
        
        # Calculate RTE
        rte = self.compute_rte(pred_trans, gt_trans)
        
        # Calculate acceleration error
        accel_pred = self._compute_acceleration(joints_pred_with_trans, frame_rate)
        accel_gt = self._compute_acceleration(joints_gt_with_trans, frame_rate)
        accel_err = torch.mean(torch.norm(accel_pred - accel_gt, dim=-1))
        
        # Calculate jitter ratio
        jitter = self._compute_jitter(joints_pred_with_trans) / self._compute_jitter(joints_gt_with_trans)
        
        # Compile all metrics
        metrics = {
            'pa_mpjpe': pa_mpjpe.detach().cpu().numpy(),
            'mpjpe': mpjpe.detach().cpu().numpy(),
            'mpvpe': mpvpe.detach().cpu().numpy(),
            'w_mpjpe': w_mpjpe.detach().cpu().numpy(),
            'w_mpvpe': w_mpvpe.detach().cpu().numpy(),
            'mpjre': mpjre.detach().cpu().numpy(),
            'rte': rte.detach().cpu().numpy(),
            'accel_err': accel_err.detach().cpu().numpy(),
            'jitter': jitter.detach().cpu().numpy()
        }
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        
        return metrics
    
    def evaluate_batch(self, result_dict, batch_size=None):
        """
        Evaluate all metrics for a batch of sequences
        
        Args:
            result_dict: Dictionary containing predictions and ground truth
                - pred_poses: List of predicted pose parameters
                - pred_trans: List of predicted translation parameters
                - gt_poses: List of ground truth pose parameters
                - gt_trans: List of ground truth translation parameters
                - gt_betas: List of shape parameters
            batch_size: Batch size, if None process all data
        
        Returns:
            metrics: Dictionary containing all evaluation metrics
        """
        # Initialize metrics dictionary
        metrics = {
            'pa_mpjpe': [],
            'mpjpe': [],
            'mpvpe': [],
            'w_mpjpe': [],
            'w_mpvpe': [],
            'mpjre': [],
            'rte': [],
            'accel_err': [],
            'jitter': []
        }
        
        # Get sequence lengths as weights
        batch_length = [len(result_dict['gt_trans'][i]) for i in range(len(result_dict['gt_trans']))]
        
        # Determine number of batches to process
        num_batches = len(result_dict['gt_trans']) if batch_size is None else min(batch_size, len(result_dict['gt_trans']))
        
        # Evaluate batch by batch
        for batch_idx in trange(num_batches, desc='Evaluating'):
            pred_pose = result_dict['pred_poses'][batch_idx]
            pred_trans = result_dict['pred_trans'][batch_idx]
            gt_pose = result_dict['gt_poses'][batch_idx]
            gt_trans = result_dict['gt_trans'][batch_idx]
            betas = result_dict['gt_betas'][batch_idx]
            
            # Get frame rate if available
            frame_rate = 60  # Default frame rate
            if 'framerate' in result_dict and result_dict['framerate'][batch_idx] is not None:
                frame_rate = result_dict['framerate'][batch_idx].item()
            
            # Evaluate current batch
            batch_metrics = self.evaluate_sequence(pred_pose, gt_pose, pred_trans, gt_trans, betas, frame_rate)
            
            # Update metrics dictionary
            for key, value in batch_metrics.items():
                metrics[key].append(value)
        
        # Calculate weighted average metrics
        avg_metrics = {}
        for key, values in metrics.items():
            avg_metrics[key] = np.average(values, weights=batch_length[:num_batches])
        
        return avg_metrics


def evaluate_form_results(result_dict, device='cuda:0'):
    """
    Evaluate FoRM model results
    
    Args:
        result_dict: Dictionary containing predictions and ground truth
            - pred_poses: List of predicted pose parameters
            - pred_trans: List of predicted translation parameters
            - gt_poses: List of ground truth pose parameters
            - gt_trans: List of ground truth translation parameters
            - gt_betas: List of shape parameters
        device: Computation device
    
    Returns:
        metrics: Dictionary containing all evaluation metrics
    """
    # Initialize evaluator
    evaluator = PoseEvaluator(device=device)
    
    # Evaluate results
    metrics = evaluator.evaluate_batch(result_dict)
    
    # Print evaluation results
    print("\n===== Evaluation Results =====")
    print(f"PA-MPJPE: {metrics['pa_mpjpe']:.2f} mm")
    print(f"MPJPE: {metrics['mpjpe']:.2f} mm")
    print(f"MPVPE: {metrics['mpvpe']:.2f} mm")
    print(f"W-MPJPE: {metrics['w_mpjpe']:.2f} mm")
    print(f"W-MPVPE: {metrics['w_mpvpe']:.2f} mm")
    print(f"MPJRE: {metrics['mpjre']:.2f} degrees")
    print(f"RTE: {metrics['rte']:.4f} m")
    print(f"Accel-Err: {metrics['accel_err']:.4f} m/s²")
    print(f"Jitter: {metrics['jitter']:.4f}")
    print("==============================\n")
    
    return metrics
