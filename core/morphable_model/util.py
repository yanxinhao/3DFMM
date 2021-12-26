# coding=utf-8
"""
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2021-05-10 10:28:50
LastEditors: yanxinhao
Description: 
reference : https://github.com/HavenFeng/photometric_optimization/blob/master/util.py
"""
import numpy as np
import torch
import torch.nn.functional as F
import math
from collections import OrderedDict
import os
from scipy.ndimage import morphology

# from skimage.io import imsave
import cv2


def dict2obj(d):
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


def check_mkdir(path):
    if not os.path.exists(path):
        print("making %s" % path)
        os.makedirs(path)


def l2_distance(verts1, verts2):
    return torch.sqrt(((verts1 - verts2) ** 2).sum(2)).mean(1).mean()


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def batch_rodrigues(theta):
    # theta N x 3
    batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)


# def batch_orth_proj(X, camera):
#     '''
#         X is N x num_points x 3
#     '''
#     camera = camera.clone().view(-1, 1, 3)
#     X_trans = X[:, :, :2] + camera[:, :, 1:]
#     X_trans = torch.cat([X_trans, X[:, :, 2:]], 2)
#     shape = X_trans.shape
#     # Xn = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
#     Xn = (camera[:, :, 0:1] * X_trans)
#     return Xn


def batch_orth_proj(X, camera, t):
    """
    X is N x num_points x 3
    """
    X_trans = X[:, :, :2] + t
    X_trans = torch.cat([X_trans, X[:, :, 2:]], 2)
    # shape = X_trans.shape
    # Xn = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    Xn = camera.scale * X_trans
    Xn[..., 1:] = -Xn[..., 1:]
    return Xn


def batch_persp_proj(vertices, cam, t, eps=1e-9):
    """
    Calculate projective transformation of vertices given a camera and it's translation
    Input parameters:
    cam:
    t: batch_size * 1 * 3 xyz translation in world coordinate
    t: batch_size * 1 * 3 extrinsic calibration parameters
    Returns: For each point [X,Y,Z] in world coordinates [x,y,z] where u,v are the coordinates of the projection in
    NDC and z is the depth
    """
    t = t.unsqueeze(1)
    vertices = vertices + t
    # camera looks at -z direction
    x, y, z = vertices[:, :, 0], vertices[:, :, 1], -vertices[:, :, 2]
    x_ = x / (z + eps)
    y_ = y / (z + eps)

    vertices = torch.stack([x_, y_], dim=-1)
    x_n, y_n = (
        vertices[:, :, 0] * cam.focal_length[0] + cam.principal_point[0],
        vertices[:, :, 1] * cam.focal_length[0] + cam.principal_point[1],
    )
    vertices = torch.stack([x_n, -y_n, z], dim=-1)
    return vertices


# def batch_persp_proj(vertices, cam, f, t, orig_size=256, eps=1e-9):
#     '''
#     Calculate projective transformation of vertices given a projection matrix
#     Input parameters:
#     f: torch tensor of focal length
#     t: batch_size * 1 * 3 xyz translation in world coordinate
#     K: batch_size * 3 * 3 intrinsic camera matrix
#     R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
#     dist_coeffs: vector of distortion coefficients
#     orig_size: original size of image captured by the camera
#     Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
#     pixels and z is the depth
#     '''
#     device = vertices.device

#     K = torch.tensor([f, 0., cam['c'][0], 0., f, cam['c'][1], 0., 0., 1.]).view(3, 3)[None, ...].repeat(
#         vertices.shape[0], 1).to(device)
#     R = batch_rodrigues(cam['r'][None, ...].repeat(
#         vertices.shape[0], 1)).to(device)
#     dist_coeffs = cam['k'][None, ...].repeat(vertices.shape[0], 1).to(device)

#     vertices = torch.matmul(vertices, R.transpose(2, 1)) + t
#     x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
#     x_ = x / (z + eps)
#     y_ = y / (z + eps)

#     # Get distortion coefficients from vector
#     k1 = dist_coeffs[:, None, 0]
#     k2 = dist_coeffs[:, None, 1]
#     p1 = dist_coeffs[:, None, 2]
#     p2 = dist_coeffs[:, None, 3]
#     k3 = dist_coeffs[:, None, 4]

#     # we use x_ for x' and x__ for x'' etc.
#     r = torch.sqrt(x_ ** 2 + y_ ** 2)
#     x__ = x_ * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6)
#                 ) + 2 * p1 * x_ * y_ + p2 * (r ** 2 + 2 * x_ ** 2)
#     y__ = y_ * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6)
#                 ) + p1 * (r ** 2 + 2 * y_ ** 2) + 2 * p2 * x_ * y_
#     vertices = torch.stack([x__, y__, torch.ones_like(z)], dim=-1)
#     vertices = torch.matmul(vertices, K.transpose(1, 2))
#     u, v = vertices[:, :, 0], vertices[:, :, 1]
#     v = orig_size - v
#     # map u,v from [0, img_size] to [-1, 1] to be compatible with the renderer
#     u = 2 * (u - orig_size / 2.) / orig_size
#     v = 2 * (v - orig_size / 2.) / orig_size
#     vertices = torch.stack([u, v, z], dim=-1)

#     return vertices


def tensor_vis_landmarks(images, landmarks, gt_landmarks=None, color="g", isScale=True):
    # visualize landmarks
    vis_landmarks = []
    images = images.cpu().numpy()
    predicted_landmarks = landmarks.detach().cpu().numpy()
    if gt_landmarks is not None:
        gt_landmarks_np = gt_landmarks.detach().cpu().numpy()
    for i in range(images.shape[0]):
        image = images[i]
        image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]].copy()
        image = image * 255
        predicted_landmark = np.ones_like(predicted_landmarks[i])
        if isScale:
            predicted_landmark[..., 0] = (
                predicted_landmarks[i, :, 0] * image.shape[1] / 2 + image.shape[1] / 2
            )
            predicted_landmark[..., 1] = (
                predicted_landmarks[i, :, 1] * image.shape[0] / 2 + image.shape[0] / 2
            )
        else:
            predicted_landmark = predicted_landmarks[i]

        if predicted_landmark.shape[0] == 68:
            image_landmarks = plot_kpts(image, predicted_landmark, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(
                    image_landmarks,
                    gt_landmarks_np[i] * image.shape[0] / 2 + image.shape[0] / 2,
                    "r",
                )
        else:
            image_landmarks = plot_verts(image, predicted_landmark, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(
                    image_landmarks,
                    gt_landmarks_np[i] * image.shape[0] / 2 + image.shape[0] / 2,
                    "r",
                )

        vis_landmarks.append(image_landmarks)

    vis_landmarks = np.stack(vis_landmarks)
    vis_landmarks = (
        torch.from_numpy(vis_landmarks[:, :, :, [2, 1, 0]].transpose(0, 3, 1, 2))
        / 255.0
    )  # , dtype=torch.float32)
    return vis_landmarks


end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1


def plot_kpts(image, kpts, color="r"):
    """Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    """
    if color == "r":
        c = (255, 0, 0)
    elif color == "g":
        c = (0, 255, 0)
    elif color == "b":
        c = (255, 0, 0)
    image = image.copy()
    kpts = kpts.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        if kpts.shape[1] == 4:
            if kpts[i, 3] > 0.5:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)
        image = cv2.circle(image, (st[0], st[1]), 1, c, 2)
        if i in end_list:
            continue
        ed = kpts[i + 1, :2]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)

    return image
