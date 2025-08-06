#!/usr/bin/venv python
# -*- coding: utf-8 -*-
#
import torch

skeleton_link = [
    ('Hips', 'RightUpLeg'),
    ('Hips', 'LeftUpLeg'),
    ('RightUpLeg', 'RightLeg'),
    ('LeftUpLeg', 'LeftLeg'),
    ('RightLeg', 'RightFoot'),
    ('LeftLeg', 'LeftFoot'),
    # ('RightFoot', 'RightToeBase'),
    # ('LeftFoot', 'LeftToeBase'),
    ('Hips', 'Spine'),
    ('Spine', 'Spine1'),
    ('Spine1', 'Spine2'),
    ('Spine2', 'Neck'),
    ('Neck', 'Head'),
    ('Spine2', 'RightShoulder'),
    ('Spine2', 'LeftShoulder'),
    ('RightShoulder', 'RightArm'),
    ('LeftShoulder', 'LeftArm'),
    ('RightArm', 'RightForeArm'),
    ('LeftArm', 'LeftForeArm'),
    ('RightForeArm', 'RightHand'),
    ('LeftForeArm', 'LeftHand'),
]

JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

NoitomSMPLJoints = ['Hips', 'LeftUpLeg', 'RightUpLeg',
                    'Spine', 'LeftLeg', 'RightLeg',
                    'Spine1', 'LeftFoot', 'RightFoot',
                    'Spine2', 'LeftFoot', 'RightFoot',
                    'Neck', 'LeftShoulder', 'RightShoulder',
                    'Head', 'LeftArm', 'RightArm',
                    'LeftForeArm', 'RightForeArm',
                    'LeftHand', 'RightHand',
                    'LeftHand', 'RightHand']
joint_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
NoitomMaskJoints = [NoitomSMPLJoints[i] for i in joint_mask]


class RotationConst:
    dip2amass = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]])
    amass2dip = dip2amass.t()
