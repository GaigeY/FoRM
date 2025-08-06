#!/usr/bin/venv python
# -*- coding: utf-8 -*-
#
import os
import torch
import torch.nn as nn

import sys
current_file_path = os.path.abspath(__file__)
PATH_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if PATH_PROJECT_ROOT not in sys.path:
    sys.path.append(PATH_PROJECT_ROOT)
    
from lib.math.angular import axis_angle_to_rotation_matrix, rotation_matrix_to_r6d, r6d_to_rotation_matrix
from lib.config import config
from lib.network.module import Regressor, NeuralInitialization
from lib.network.utils import rollout_global_motion, reset_foot_contact_velocity, reset_foot_contact_velocity_dynamic, decouple_yaw_axis_angle, decouple_yaw_r6d, remix_yaw_r6d


class ShoePoser(nn.Module):
    def __init__(self, in_dim, d_embed, embed_drop, rnn_type, n_layers, n_joints):
        super(ShoePoser, self).__init__()

        self.n_joints = n_joints
        self.embedding = nn.Linear(in_dim, d_embed)
        self.embed_drop = nn.Dropout(embed_drop)

        self.neural_init = NeuralInitialization(in_dim + n_joints*3, d_embed, rnn_type, n_layers)
        self.reg = Regressor(d_embed, d_embed, [n_joints * 3], n_joints * 3, rnn_type, n_layers)

    def forward(self, x, init_params):
        batch_size, frame_size = x.shape[:2]
        init = torch.cat([init_params.reshape(batch_size, frame_size, -1),
                          x.reshape(batch_size, frame_size, -1)], dim=-1)
        # Note! Only the first frame is used for initialization!
        h0 = self.neural_init(init[:, 0])
        pred_list = [init_params[:, [0]]]
        motion_context_list = []
        
        x = self.embedding(x.reshape(batch_size, frame_size, -1))
        x = self.embed_drop(x)

        for i in range(frame_size):
            (pred_pose, ), motion_context, h0 = self.reg(x[:, [i]], pred_list[-1:], h0)
            motion_context_list.append(motion_context)
            pred_list.append(pred_pose)

        pred_poses = torch.cat(pred_list[1:], dim=1).view((batch_size, frame_size, -1, 3))
        if self.n_joints == 24:
            # Attention: Remove yaw for root pose, if output 24 joints rather than 23 joints
            pred_poses[:, :, 0] = decouple_yaw_axis_angle(pred_poses[:, :, 0])[0]
        motion_context = torch.cat(motion_context_list, dim=1)
        motion_context = torch.cat((motion_context, pred_poses.reshape(batch_size, frame_size, -1)), dim=-1)
        return pred_poses, motion_context


class Trajectory(nn.Module):
    def __init__(self, in_dim, d_embed, embed_drop, rnn_type, n_layers, root_r_dim=6):
        super().__init__()
        self.embedding = nn.Linear(in_dim, d_embed)
        self.embed_drop = nn.Dropout(embed_drop)
        
        self.neural_init = NeuralInitialization(in_dim + root_r_dim, d_embed, rnn_type, n_layers)
        self.reg = Regressor(d_embed, d_embed, [3, root_r_dim], root_r_dim, rnn_type, n_layers)
        
    def process_yaw(self, pred_root_r):
        pred_root_wo_yaw, pred_yaw = decouple_yaw_r6d(pred_root_r)
        pred_yaw = torch.zeros_like(pred_yaw).to(pred_yaw.device).requires_grad_(False)
        return pred_root_wo_yaw, pred_yaw
    
    def forward(self, x, init_params, h0=None):
        # x: [shoe, addition, motion_context]
        batch_size, frame_size = x.shape[:2]
        init = torch.cat([init_params.reshape(batch_size, frame_size, -1),
                          x.reshape(batch_size, frame_size, -1)], dim=-1)
        if h0 is None:
            # Note! Only the first frame is used for initialization!
            h0 = self.neural_init(init[:, 0])
        
        first_frame_root, first_frame_yaw = decouple_yaw_r6d(init_params[:, [0]])
        pred_root_list, pred_yaw_list, pred_vel_list, pred_contact_list = [first_frame_root], [first_frame_yaw], [], []
        
        x = self.embedding(x.reshape(batch_size, frame_size, -1))
        x = self.embed_drop(x)

        for i in range(frame_size):
            (pred_root_v, pred_root_r), _, h0 = self.reg(x[:, [i]], pred_root_list[-1:], h0)
            
            pred_root_wo_yaw, pred_yaw = self.process_yaw(pred_root_r)
            
            pred_root_list.append(pred_root_wo_yaw)
            pred_yaw_list.append(pred_yaw)
            pred_vel_list.append(pred_root_v)
        # Root orientation is longer by 1 frame, which is used for computing the first frame's global motion
        pred_root = torch.cat(pred_root_list, dim=1).view(batch_size, frame_size + 1, -1)
        pred_yaw = torch.cat(pred_yaw_list, dim=1).view(batch_size, frame_size + 1)
        pred_vel = torch.cat(pred_vel_list, dim=1).view(batch_size, frame_size, -1)
        
        return pred_root, pred_yaw, pred_vel


class YawTrajectory(Trajectory):
    def __init__(self, in_dim, d_embed, embed_drop, rnn_type, n_layers, root_r_dim=6):
        super().__init__(in_dim, d_embed, embed_drop, rnn_type, n_layers, root_r_dim)
        
    def process_yaw(self, pred_root_r):
        pred_root_wo_yaw, pred_yaw = decouple_yaw_r6d(pred_root_r)
        return pred_root_wo_yaw, pred_yaw


class FoRMNet(nn.Module):
    def __init__(self, shoe_pres_dim, shoe_imu_dim, shoe_add_dim, gt_joint_num,
                 gt_root_orient_dim, gt_root_trans_dim=3,
                 rnn_type='LSTM', n_layers=2, d_embed=512,
                 smpl_model=None):
        super(FoRMNet, self).__init__()
        poser_context_dim = d_embed + gt_joint_num * 3
        
        self.smpl_model = smpl_model
        self.framerate = None

        self.shoe_poser = ShoePoser(shoe_pres_dim + shoe_imu_dim + shoe_add_dim, d_embed,
                                    0.1, rnn_type, n_layers, gt_joint_num)
        self.shoe_rooter = Trajectory(shoe_pres_dim + shoe_imu_dim + shoe_add_dim + poser_context_dim, d_embed,
                                      0.1, rnn_type, n_layers, gt_root_orient_dim)
        self.shoe_yawer = YawTrajectory(shoe_pres_dim + shoe_imu_dim + shoe_add_dim + poser_context_dim + gt_root_orient_dim + 3, d_embed,
                                        0.1, rnn_type, n_layers, gt_root_orient_dim)
        
        self.output = {}
    
    def forward(self, measurement, addition, poses, init_trans, framerate=60, betas=None):
        """
        Args:
            measurement: (batch_size, window_size, shoe_pres_dim + shoe_imu_dim)
            addition: (batch_size, window_size, shoe_add_dim)
            poses: (batch_size, window_size, 72)
            init_trans: (batch_size, window_size, 3)
            betas: (batch_size, window_size, 10) Used for root velocity reset only
        Returns:
            pred_poses: (batch_size, window_size, 69), used for smpl body pose fitting
        """
        # ------------- Preprocess -------------
        batch_size, window_size = poses.shape[:2]
        init_root_theta = poses[..., :3]
        init_body_poses = poses[..., 3:]
        init_glb_ori = rotation_matrix_to_r6d(axis_angle_to_rotation_matrix(init_root_theta)).view(batch_size, window_size, 6)
        
        # ------------- Inference -------------
        # Stage1: Estimate body pose
        # Init param only used first frame for initialization
        pose_init_param = torch.cat([init_body_poses, ], dim=-1)
        _, init_yaw = decouple_yaw_axis_angle(init_root_theta)
        pose_input = torch.cat([measurement, addition], dim=-1)
        
        pred_poses, motion_context = self.shoe_poser(pose_input, pose_init_param)
        pred_body_poses = pred_poses
        
        # Stage2: Estimate root motion without yaw
        root_input = torch.cat([pose_input, motion_context], dim=-1)
        root_init_param = torch.cat([init_glb_ori, ], dim=-1)
        pred_pitch_roll_freeze, _, pred_vel_freeze = self.shoe_rooter(root_input, root_init_param)
        # Caution: pred_pitch_roll_freeze and init_yaw is longer by 1 frame in window
        init_yaw = torch.cat([init_yaw[..., :1], init_yaw], dim=-1)
        # pred_root_ori_freeze = remix_yaw_r6d(pred_pitch_roll_freeze, init_yaw)
        
        # Stage3: Estimate yaw and refine root motion
        yaw_input = torch.cat([root_input, pred_pitch_roll_freeze[:, 1:], pred_vel_freeze], dim=-1)
        pred_pitch_roll_refine, pred_yaw, pred_vel_refine = self.shoe_yawer(yaw_input, root_init_param)
        # Caution: pred_pitch_roll_refine and pred_yaw is longer by 1 frame in window
        pred_root_ori_refine = remix_yaw_r6d(pred_pitch_roll_refine, pred_yaw)
        
        # ------------- Register output -------------        
        self.output["pred_body_poses"] = pred_body_poses.view(batch_size, window_size, -1)
        
        # Refine the trajectory and get root_ori[:, 1:]
        # pred_root_ori_freeze, pred_trans_freeze = rollout_global_motion(pred_root_ori_freeze, pred_vel_freeze, init_trans[:, 0], framerate)
        pred_root_ori_refine, pred_trans_refine = rollout_global_motion(pred_root_ori_refine, pred_vel_refine, init_trans[:, 0], framerate)
        
        # ------------- Register refined root -------------
        self.output["pred_yaw"] = pred_yaw[:, 1:] / torch.pi
        # self.output["pred_root_ori_freeze"] = pred_root_ori_freeze  # (batch_size, window_size, 6)
        self.output["pred_root_ori_refine"] = pred_root_ori_refine  # (batch_size, window_size, 6)
        # self.output["pred_root_trans_freeze"] = pred_trans_freeze  # (batch_size, window_size, 3)
        self.output["pred_root_trans_refine"] = pred_trans_refine  # (batch_size, window_size, 3)
        
        # ------------- Output -------------
        return self.output
