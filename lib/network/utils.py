#!/usr/bin/venv python
# -*- coding: utf-8 -*-
#
import os
import torch

import sys

current_file_path = os.path.abspath(__file__)
PATH_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if PATH_PROJECT_ROOT not in sys.path:
    sys.path.append(PATH_PROJECT_ROOT)
from lib.math.angular import r6d_to_rotation_matrix, rotation_matrix_to_r6d, rotation_matrix_to_axis_angle, axis_angle_to_rotation_matrix
from lib.config import constant


def reset_foot_contact_velocity(smpl, body_pose, pred_root_ori, pred_root_vel, betas=None, contact_mask=None, framerate=60, contact_thr=0.7):
    """
    Reset foot contact velocity.
    :param
        - smpl: smpl model
        - body_pose: (N, window_size, 69)
        - pred_root_ori: (N, window_size + 1, 6)
        - pred_root_vel: (N, window_size, 3)
        - betas: (N, window_size, 10) or None.
        - contact_mask: (N, window_size, 2) or None. sigmoid output, need to convert to 0/1, where 1 means contact, 0 means no contact
        - framerate: int, frame rate
        - contact_thr: float, contact threshold
    :return:
        - pred_root_vel: (N, window_size, 3)
    """
    batch_size, window_size = pred_root_vel.shape[:2]
    dt = 1 / framerate
    
    if contact_mask is None:
        return pred_root_vel
    if betas is None:
        betas = torch.zeros((batch_size, window_size, 10),
                            dtype=pred_root_vel.dtype,
                            device=pred_root_vel.device)
    
    # 1. Convert root orientation from r6d to axis-angle
    root_ori_mat = r6d_to_rotation_matrix(pred_root_ori[:, 1:]).view(batch_size, window_size, 3, 3)
    root_ori_axis = rotation_matrix_to_axis_angle(root_ori_mat).view(batch_size, window_size, -1)
    # 2. Get global root velocity
    global_root_vel = (root_ori_mat @ pred_root_vel.unsqueeze(-1)).squeeze(-1)
    
    # 3. Get foot joint location
    joints = smpl(betas=betas.view(-1, 10), 
                  body_pose=body_pose.view(-1, 69), 
                  global_orient=root_ori_axis.view(-1, 3)).joints
    foot_joints = joints[:, [10, 7, 11, 8], :].view(batch_size, window_size, 4, 3)
    
    # 4. Get foot joint world velocity via pred_root_vel
    foot_vel_local = (foot_joints[:, 1:] - foot_joints[:, :-1]) / dt.view(-1, 1, 1, 1)
    foot_vel_world = foot_vel_local + global_root_vel[:, 1:].unsqueeze(2)
    foot_vel_world = torch.cat((torch.zeros_like(foot_vel_world[:, :1], dtype=foot_vel_world.dtype, device=foot_vel_world.device), foot_vel_world), dim=1)
    
    # 5. Update global root velocity with removing contact weight
    contact_weight = (contact_mask > contact_thr).clone().detach().float().unsqueeze(-1)
    updated_global_root_vel = global_root_vel - (foot_vel_world * contact_weight).sum(dim=2) / (contact_weight.sum(dim=2) + 1e-8)
    
    # 6. Convert global root velocity back to local root velocity
    updated_local_root_vel = (root_ori_mat.transpose(-1, -2) @ updated_global_root_vel.unsqueeze(-1)).squeeze(-1)
    
    return updated_local_root_vel


def reset_foot_contact_velocity_dynamic(smpl, body_pose, pred_root_ori, pred_root_vel, foot_offset, betas=None, contact_mask=None, framerate=60, contact_thr=0.7):
    """
    Reset foot contact velocity.
    :param
        - smpl: smpl model
        - body_pose: (N, window_size, 69)
        - pred_root_ori: (N, window_size + 1, 6)
        - pred_root_vel: (N, window_size, 3)
        - foot_offset: (N, window_size, 4, 3)
        - betas: (N, window_size, 10) or None.
        - contact_mask: (N, window_size, 2) or None. sigmoid output, need to convert to 0/1, where 1 means contact, 0 means no contact
        - framerate: int, frame rate
        - contact_thr: float, contact threshold
    :return:
        - pred_root_vel: (N, window_size, 3)
    """
    batch_size, window_size = pred_root_vel.shape[:2]
    dt = 1 / framerate
    
    if contact_mask is None:
        return pred_root_vel
    if betas is None:
        betas = torch.zeros((batch_size, window_size, 10),
                            dtype=pred_root_vel.dtype,
                            device=pred_root_vel.device)
    
    # 1. Convert root orientation from r6d to axis-angle
    root_ori_mat = r6d_to_rotation_matrix(pred_root_ori[:, 1:]).view(batch_size, window_size, 3, 3)
    root_ori_axis = rotation_matrix_to_axis_angle(root_ori_mat).view(batch_size, window_size, -1)
    # 2. Get global root velocity
    global_root_vel = (root_ori_mat @ pred_root_vel.unsqueeze(-1)).squeeze(-1)
    
    # 3. Get foot joint location
    joints = smpl(betas=betas.view(-1, 10), 
                  body_pose=body_pose.view(-1, 69), 
                  global_orient=root_ori_axis.view(-1, 3)).joints
    foot_joints = joints[:, [10, 7, 11, 8], :].view(batch_size, window_size, 4, 3)
    foot_joints = foot_joints + foot_offset
    
    # 4. Get foot joint world velocity via pred_root_vel
    foot_vel_local = (foot_joints[:, 1:] - foot_joints[:, :-1]) / dt.view(-1, 1, 1, 1)
    foot_vel_world = foot_vel_local + global_root_vel[:, 1:].unsqueeze(2)
    foot_vel_world = torch.cat((torch.zeros_like(foot_vel_world[:, :1], dtype=foot_vel_world.dtype, device=foot_vel_world.device), foot_vel_world), dim=1)
    
    # 5. Update global root velocity with removing contact weight
    contact_weight = (contact_mask > contact_thr).clone().detach().float().unsqueeze(-1)
    updated_global_root_vel = global_root_vel - (foot_vel_world * contact_weight).sum(dim=2) / (contact_weight.sum(dim=2) + 1e-8)
    
    # 6. Convert global root velocity back to local root velocity
    updated_local_root_vel = (root_ori_mat.transpose(-1, -2) @ updated_global_root_vel.unsqueeze(-1)).squeeze(-1)
    
    return updated_local_root_vel


def rollout_global_motion(root_r, root_v, init_trans=None, framerate=60):
    batch_size, window_size = root_r.shape[:2]
    dt = 1 / framerate
    
    root_rmat = r6d_to_rotation_matrix(root_r).view(batch_size, window_size, 3, 3)
    vel_world = (root_rmat[:, :-1] @ (root_v * dt.view(-1, 1, 1)).unsqueeze(-1)).squeeze(-1)
    trans = torch.cumsum(vel_world, dim=1)
    if init_trans is not None: trans = trans + init_trans.view(batch_size, 1, 3)
    
    return root_r[:, 1:], trans


def decouple_yaw_matrix(global_ori):
    """
    Remove yaw from global orientation.
    Args:
        global_ori: [batch_size, window_size, 3, 3] SMPL global orientation, rotation matrix format
    Returns:
        global_ori: [batch_size, window_size, 3, 3] SMPL global orientation, rotation matrix format
        yaw: [batch_size, window_size] yaw angle
    """
    batch_size, window_size = global_ori.shape[:2]
    
    # remove z rotation
    yaw = torch.atan2(global_ori[..., 1, 0], global_ori[..., 0, 0]).view(batch_size, window_size)
    # build inverse yaw rotation matrix
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    inv_yaw_mat = torch.eye(3, device=global_ori.device).repeat(batch_size, window_size, 1, 1)
    inv_yaw_mat[..., 0, 0] = cos_yaw
    inv_yaw_mat[..., 0, 1] = sin_yaw
    inv_yaw_mat[..., 1, 0] = -sin_yaw
    inv_yaw_mat[..., 1, 1] = cos_yaw
    
    global_ori = inv_yaw_mat @ global_ori
    
    return global_ori, yaw


def decouple_yaw_axis_angle(global_ori):
    """
    Remove yaw from global orientation.
    Args:
        global_ori: [batch_size, window_size, 3] SMPL global orientation, axis-angle format
    Returns:
        global_ori: [batch_size, window_size, 3] SMPL global orientation, axis-angle format
        yaw: [batch_size, window_size] yaw angle
    """
    batch_size, window_size = global_ori.shape[:2]
    global_ori_mat = axis_angle_to_rotation_matrix(global_ori).view(batch_size, window_size, 3, 3)
    
    new_global_ori_mat, yaw = decouple_yaw_matrix(global_ori_mat)
    new_global_ori = rotation_matrix_to_axis_angle(new_global_ori_mat).view(batch_size, window_size, 3)
    
    return new_global_ori, yaw


def decouple_yaw_r6d(global_ori):
    """
    Remove yaw from global orientation.
    Args:
        global_ori: [batch_size, window_size, 6] SMPL global orientation, r6d format
    Returns:
        global_ori: [batch_size, window_size, 6] SMPL global orientation, r6d format
        yaw: [batch_size, window_size] yaw angle
    """ 
    batch_size, window_size = global_ori.shape[:2]
    global_ori_mat = r6d_to_rotation_matrix(global_ori).view(batch_size, window_size, 3, 3)
    new_global_ori_mat, yaw = decouple_yaw_matrix(global_ori_mat)
    new_global_ori = rotation_matrix_to_r6d(new_global_ori_mat).view(batch_size, window_size, 6)
    
    return new_global_ori, yaw


def remix_yaw_matrix(global_ori, yaw):
    """
    Couple yaw to global orientation.
    Args:
        global_ori: [batch_size, window_size, 3, 3] SMPL global orientation, rotation matrix format
        yaw: [batch_size, window_size] yaw angle
    Returns:
        global_ori: [batch_size, window_size, 3, 3] SMPL global orientation, rotation matrix format
    """
    batch_size, window_size = global_ori.shape[:2]
    
    # build yaw rotation matrix
    yaw = yaw.view(batch_size, window_size)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    yaw_mat = torch.eye(3, device=global_ori.device).repeat(batch_size, window_size, 1, 1)
    yaw_mat[..., 0, 0] = cos_yaw
    yaw_mat[..., 0, 1] = -sin_yaw
    yaw_mat[..., 1, 0] = sin_yaw
    yaw_mat[..., 1, 1] = cos_yaw

    global_ori = yaw_mat @ global_ori

    return global_ori


def remix_yaw_axis_angle(global_ori, yaw):
    """
    Couple yaw to global orientation.
    Args:
        global_ori: [batch_size, window_size, 3] SMPL global orientation, axis-angle format
        yaw: [batch_size, window_size] yaw angle
    Returns:
        global_ori: [batch_size, window_size, 3] SMPL global orientation, axis-angle format
    """
    # print(global_ori.shape)
    batch_size, window_size = global_ori.shape[:2]
    global_ori_mat = axis_angle_to_rotation_matrix(global_ori).view(batch_size, window_size, 3, 3)
    new_global_ori_mat = remix_yaw_matrix(global_ori_mat, yaw)
    new_global_ori = rotation_matrix_to_axis_angle(new_global_ori_mat).view(batch_size, window_size, 3)
    
    return new_global_ori


def remix_yaw_r6d(global_ori, yaw):
    """
    Couple yaw to global orientation.
    Args:
        global_ori: [batch_size, window_size, 6] SMPL global orientation, r6d format
        yaw: [batch_size, window_size] yaw angle
    Returns:
        global_ori: [batch_size, window_size, 6] SMPL global orientation, r6d format
    """
    batch_size, window_size = global_ori.shape[:2]
    global_ori_mat = r6d_to_rotation_matrix(global_ori).view(batch_size, window_size, 3, 3)
    new_global_ori_mat = remix_yaw_matrix(global_ori_mat, yaw)
    new_global_ori = rotation_matrix_to_r6d(new_global_ori_mat).view(batch_size, window_size, 6)
    
    return new_global_ori


def get_gyro_from_r6d(rot_r6d, framerate=60):
    """
    Convert r6D orientation to gyro.
    Args:
        rot_r6d: [batch_size, window_size, joint_num, 6], rotation in 6D format
        framerate: int, frame rate
    Returns:
        gyro_r6d: [batch_size, window_size - 1, joint_num, 6], gyro in 6D format
    """
    batch_size, window_size = rot_r6d.shape[:2]
    dt = 1 / framerate
    rot_mat = r6d_to_rotation_matrix(rot_r6d).view(batch_size, window_size, -1, 3, 3)
    gyro_mat = (rot_mat[:, 1:] @ rot_mat[:, :-1].transpose(-1, -2))
    gyro_r6d = rotation_matrix_to_r6d(gyro_mat).view(batch_size, window_size - 1, -1, 6) / dt.view(-1, 1, 1, 1)
    return gyro_r6d
