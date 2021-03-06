# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:30:24 2020

@author: dubs
"""
import numpy as np
import pandas as pd
import os
import torch
"""
    0 - Head
    1 - Neck
    2 - Lshoulder
    3 - Lelbow
    4 - Lwrist
    5 - Rshoulder
    6 - Relbow
    7 - Rwrist
    8 - Pelvis
    9 - Lhip
    10 - Lknee
    11 - Lankle
    12 - Rhip
    13 - Rknee
    14 - Rankle
"""

def trans_motion3d(motion3d,local3d=None,unit=128):
    motion3d = motion3d*unit
    if local3d is not None:
        motion_proj = local3d[[0,2],:] @ motion3d
    else:
        motion_proj = motion3d[:,[0,2],:]
    motion_proj[:,1,:] = -motion_proj[:,1,:]
    motion_proj = trans_motion2d(motion_proj)
    return motion_proj

def trans_motion2d(motion2d):
    centers = motion2d[8,:,:]
    motion_proj = motion2d - centers
    velocity = np.c_[np.zeros((2, 1)), centers[:, 1:] - centers[:, :-1]].reshape(1, 2, -1)
    motion_proj = np.r_[motion_proj[:8], motion_proj[9:], velocity]

    return motion_proj

def trans_motion_inv(motion, sx=256, sy=256, velocity=None):
    if velocity is None:
        velocity = motion[-1].copy()
    motion_inv = np.r_[motion[:8], np.zeros((1, 2, motion.shape[-1])), motion[8:-1]]

    # restore centre position
    centers = np.zeros_like(velocity)
    sum = 0
    for i in range(motion.shape[-1]):
        sum += velocity[:, i]
        centers[:, i] = sum
    centers += np.array([[sx], [sy]])

    return motion_inv + centers.reshape((1, 2, -1))

def normalize_motion(motion, mean_pose, std_pose):
    """
    :param motion: (J, 2, T)
    :param mean_pose: (J, 2)
    :param std_pose: (J, 2)
    :return:
    """
    return (motion - mean_pose[:, :, np.newaxis]) / std_pose[:, :, np.newaxis]

def normalize_motion_inv(motion, mean_pose, std_pose):
    if len(motion.shape) == 2:
        motion = motion.reshape(-1, 2, motion.shape[-1])
    return motion * std_pose[:, :, np.newaxis] + mean_pose[:, :, np.newaxis]


def preprocess_motion2d(motion, mean_pose, std_pose):
    motion_trans = normalize_motion(trans_motion2d(motion), mean_pose, std_pose)
    motion_trans = motion_trans.reshape((-1, motion_trans.shape[-1]))
    return torch.Tensor(motion_trans).unsqueeze(0)


def postprocess_motion2d(motion, mean_pose, std_pose, sx=256, sy=256):
    motion = motion.detach().cpu().numpy()[0].reshape(-1, 2, motion.shape[-1])
    motion = trans_motion_inv(normalize_motion_inv(motion, mean_pose, std_pose), sx, sy)
    return motion

def get_local3d(motion3d, angles=None):
    """
    Get the unit vectors for local rectangular coordinates for given 3D motion
    :param motion3d: numpy array. 3D motion from 3D joints positions, shape (nr_joints, 3, nr_frames).
    :param angles: tuple of length 3. Rotation angles around each axis.
    :return: numpy array. unit vectors for local rectangular coordinates's , shape (3, 3).
    """
    # 2 RightArm 5 LeftArm 9 RightUpLeg 12 LeftUpLeg
    horizontal = (motion3d[2] - motion3d[5] + motion3d[9] - motion3d[12]) / 2
    horizontal = np.mean(horizontal, axis=1)
    horizontal = horizontal / np.linalg.norm(horizontal)
    local_z = np.array([0, 0, 1])
    local_y = np.cross(horizontal, local_z)  # bugs!!!, horizontal and local_Z may not be perpendicular
    local_y = local_y / np.linalg.norm(local_y)
    local_x = np.cross(local_y, local_z)
    local = np.stack([local_x, local_y, local_z], axis=0)

    if angles is not None:
        local = rotate_coordinates(local, angles)

    return local


def rotate_coordinates(local3d, angles):
    """
    Rotate local rectangular coordinates from given view_angles.
    :param local3d: numpy array. Unit vectors for local rectangular coordinates's , shape (3, 3).
    :param angles: tuple of length 3. Rotation angles around each axis.
    :return:
    """
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    x = local3d[0]
    x_cpm = np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ], dtype='float')
    x = x.reshape(-1, 1)
    mat33_x = cx * np.eye(3) + sx * x_cpm + (1.0 - cx) * np.matmul(x, x.T)

    mat33_z = np.array([
        [cz, sz, 0],
        [-sz, cz, 0],
        [0, 0, 1]
    ], dtype='float')

    local3d = local3d @ mat33_x.T @ mat33_z
    return local3d


def rotation_matrix_along_axis(x, angle):
    cx = np.cos(angle)
    sx = np.sin(angle)
    x_cpm = np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ], dtype='float')
    x = x.reshape(-1, 1)
    mat33_x = cx * np.eye(3) + sx * x_cpm + (1.0 - cx) * np.matmul(x, x.T)
    return mat33_x

def get_foot_vel(batch_motion, foot_idx):
    return batch_motion[:, foot_idx, 1:] - batch_motion[:, foot_idx, :-1] + batch_motion[:, -2:, 1:].repeat(1, 2, 1)