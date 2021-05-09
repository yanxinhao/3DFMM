# coding=utf-8
'''
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2021-05-08 12:36:00
LastEditors: yanxinhao
Description:
reference : https://github.com/HavenFeng/photometric_optimization/blob/master/models/FLAME.py
            https://github.com/HavenFeng/photometric_optimization/blob/master/photometric_fitting.py
'''

# Modified from smplx code for FLAME
import os
import json
import cv2
from tqdm import tqdm
import datetime
import torch
import torch.nn as nn
import torchvision
import numpy as np
import pickle
import torch.nn.functional as F
from ..util import l2_distance, batch_orth_proj, batch_persp_proj, tensor_vis_landmarks
from ..camera import Camera
from .lbs import lbs, batch_rodrigues, vertices2landmarks
from renderer import Renderer


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, config, image_size=None, device="cuda"):
        super(FLAME, self).__init__()
        self.config = config
        self.device = device
        print("creating the FLAME Decoder")
        with open(config.flame_model_path, 'rb') as f:
            # flame_model = Struct(**pickle.load(f, encoding='latin1'))
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(
            to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(
            to_np(flame_model.v_template), dtype=self.dtype))
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:, :, :config.shape_params],
                               shapedirs[:, :, 300:300 + config.expression_params]], 2)
        self.register_buffer('shapedirs', shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(
            to_np(posedirs), dtype=self.dtype))
        #
        self.register_buffer('J_regressor', to_tensor(
            to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(
            to_np(flame_model.weights), dtype=self.dtype))

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros(
            [1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                         requires_grad=False))
        default_neck_pose = torch.zeros(
            [1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                          requires_grad=False))

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(
            config.flame_lmk_embedding_path, allow_pickle=True, encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer('lmk_faces_idx', torch.tensor(
            lmk_embeddings['static_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer('lmk_bary_coords', torch.tensor(
            lmk_embeddings['static_lmk_bary_coords'], dtype=self.dtype))
        self.register_buffer('dynamic_lmk_faces_idx', torch.tensor(
            lmk_embeddings['dynamic_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer('dynamic_lmk_bary_coords', torch.tensor(
            lmk_embeddings['dynamic_lmk_bary_coords'], dtype=self.dtype))
        self.register_buffer('full_lmk_faces_idx', torch.tensor(
            lmk_embeddings['full_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer('full_lmk_bary_coords', torch.tensor(
            lmk_embeddings['full_lmk_bary_coords'], dtype=self.dtype))

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))
        # setup camera
        self.setup_camera()

        # setup render
        self.image_size = image_size
        self._setup_renderer(image_size=image_size)

    def initialize_params(self, batch_size=1):
        self.shape = nn.Parameter(torch.zeros(
            batch_size, self.config.shape_params).float().to(self.device))
        self.tex = nn.Parameter(torch.zeros(
            batch_size, self.config.tex_params).float().to(self.device))
        self.exp = nn.Parameter(torch.zeros(
            batch_size, self.config.expression_params).float().to(self.device))
        self.pose = nn.Parameter(torch.zeros(
            batch_size, self.config.pose_params).float().to(self.device))
        self.lights = nn.Parameter(torch.zeros(
            batch_size, 9, 3).float().to(self.device))
        if self.is_perspective:
            self.cam_t = nn.Parameter(torch.tensor(
                [0, 0, 1.]).repeat(batch_size, 1).float().to(self.device))
        else:
            self.cam_t = nn.Parameter(torch.zeros(
                batch_size, 2).float().to(self.device))

        self.params = {}
        self.params['shape_params'] = self.shape
        self.params['tex_params'] = self.tex
        self.params['expression_params'] = self.exp
        self.params['light_params'] = self.lights
        self.params['pose_params'] = self.pose
        self.params['cam_t'] = self.cam_t

    def setup_camera(self, camera=None, camera_path=None, is_perspective=True):
        if camera is not None:
            self.cam = camera
            self.is_perspective = camera.is_perspective
            return

        # setup camera
        self.is_perspective = is_perspective
        if self.is_perspective:
            self.cam = Camera(camera_path=camera_path, is_perspective=is_perspective,
                              principal_point=nn.Parameter(
                                  torch.zeros(2).float().to(self.device)),
                              focal_length=nn.Parameter(
                                  5 * torch.ones(2).float().to(self.device))
                              )
            self.project_fun = batch_persp_proj
        else:
            self.cam = Camera(camera_path=camera_path,
                              is_perspective=is_perspective,
                              scale=nn.Parameter(
                                  5.0 * torch.ones(1).float().to(self.device))
                              )
            self.project_fun = batch_orth_proj

    def get_params(self):
        return self.params

    def get_camera(self):
        return self.cam

    def load_shape(self, shape_path):
        with open(shape_path, 'r') as f:
            shape = json.load(f)['shape']
        self.shape = nn.Parameter(torch.tensor(
            shape).unsqueeze(0).to(self.device))

    def _setup_renderer(self, image_size):
        mesh_file = './Data/morphable_model/FLAME/head_template_mesh.obj'
        self.render = Renderer(
            (image_size[1], image_size[0]), obj_filename=mesh_file).to(self.device)

    def _find_dynamic_lmk_idx_and_bcoords(self, pose, dynamic_lmk_faces_idx,
                                          dynamic_lmk_b_coords,
                                          neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=pose.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
            Calculates landmarks by barycentric interpolation
            Input:
                vertices: torch.tensor NxVx3, dtype = torch.float32
                    The tensor of input vertices
                faces: torch.tensor (N*F)x3, dtype = torch.long
                    The faces of the mesh
                lmk_faces_idx: torch.tensor N X L, dtype = torch.long
                    The tensor with the indices of the faces used to calculate the
                    landmarks.
                lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
                    The tensor of barycentric coordinates that are used to interpolate
                    the landmarks

            Returns:
                landmarks: torch.tensor NxLx3, dtype = torch.float32
                    The coordinates of the landmarks for each mesh in the batch
        """
        # Extract the indices of the vertices for each face
        # NxLx3
        batch_size, num_verts = vertices.shape[:dd2]
        lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
            1, -1, 3).view(batch_size, lmk_faces_idx.shape[1], -1)

        lmk_faces += torch.arange(batch_size, dtype=torch.long).view(-1, 1, 1).to(
            device=vertices.device) * num_verts

        lmk_vertices = vertices.view(-1, 3)[lmk_faces]
        landmarks = torch.einsum(
            'blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
        return landmarks

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                         self.full_lmk_faces_idx.repeat(
                                             vertices.shape[0], 1),
                                         self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1))
        return landmarks3d

    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat([pose_params[:, :3], self.neck_pose.expand(
            batch_size, -1), pose_params[:, 3:], eye_pose_params], dim=1)
        template_vertices = self.v_template.unsqueeze(
            0).expand(batch_size, -1, -1)

        # import ipdb; ipdb.set_trace()
        vertices, _ = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(
            dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(
            dim=0).expand(batch_size, -1, -1)

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose, self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain, dtype=self.dtype)
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(vertices, self.faces_tensor,
                                         lmk_faces_idx,
                                         lmk_bary_coords)
        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                         self.full_lmk_faces_idx.repeat(bz, 1),
                                         self.full_lmk_bary_coords.repeat(bz, 1, 1))

        return vertices, landmarks2d, landmarks3d

    def optimize_by_landmark(self, landmarks, max_iterations=200, photometric=False,
                             images=None, image_masks=None, savefolder="./Results"):
        bz = landmarks.shape[0]
        if photometric:
            self.flametex = FLAMETex(self.config)
        # setup camera
        self.setup_camera(is_perspective=False)
        # initialize params and optimizer
        self.initialize_params(batch_size=bz)
        landmarks = torch.from_numpy(landmarks).to(self.device)
        e_opt = torch.optim.Adam(
            [self.shape, self.exp, self.pose, self.tex, self.cam_t,
                self.lights] + self.cam.get_params(),
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )
        e_opt_rigid = torch.optim.Adam(
            [self.pose, self.cam_t] + self.cam.get_params(),
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )

        gt_landmark = landmarks
        # rigid fitting of pose and camera with 51 static face landmarks,
        # this is due to the non-differentiable attribute of contour landmarks trajectory
        for k in range(max_iterations):
            losses = {}
            vertices, landmarks2d, landmarks3d = self.forward(
                shape_params=self.shape, expression_params=self.exp, pose_params=self.pose)
            trans_vertices = self.project_fun(vertices, self.cam, self.cam_t)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = self.project_fun(landmarks2d, self.cam, self.cam_t)
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = self.project_fun(landmarks3d, self.cam, self.cam_t)
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['landmark'] = l2_distance(
                landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) * self.config.w_lmks

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt_rigid.zero_grad()
            all_loss.backward()
            e_opt_rigid.step()

            loss_info = '----iter: {}, time: {}\n'.format(
                k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + \
                    '{}: {}, '.format(key, float(losses[key]))
            if k % 10 == 0:
                print(loss_info)

            if k % 10 == 0:
                grids = {}
                visind = range(bz)  # [0]
                grids['images'] = torchvision.utils.make_grid(
                    images[visind]).detach().cpu()
                grids['landmarks_gt'] = torchvision.utils.make_grid(
                    tensor_vis_landmarks(images[visind], landmarks[visind]))
                grids['landmarks2d'] = torchvision.utils.make_grid(
                    tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                grids['landmarks3d'] = torchvision.utils.make_grid(
                    tensor_vis_landmarks(images[visind], landmarks3d[visind]))
                shape_images = self.render.render_shape(
                    vertices, trans_vertices, images)
                grids['shape'] = torchvision.utils.make_grid(
                    F.interpolate(shape_images[visind], (self.image_size[1], self.image_size[0]))).detach().float().cpu()

                grid = torch.cat(list(grids.values()), 1)
                grid_image = (grid.numpy().transpose(
                    1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                grid_image = np.minimum(np.maximum(
                    grid_image, 0), 255).astype(np.uint8)
                cv2.imwrite('{}/{}.jpg'.format(savefolder, k), grid_image)

        for k in range(200, 1000):
            losses = {}
            vertices, landmarks2d, landmarks3d = self.forward(
                shape_params=self.shape, expression_params=self.exp, pose_params=self.pose)
            trans_vertices = self.project_fun(vertices, self.cam, self.cam_t)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = self.project_fun(landmarks2d, self.cam, self.cam_t)
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = self.project_fun(landmarks3d, self.cam, self.cam_t)
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['landmark'] = l2_distance(
                landmarks2d[:, :, :2], gt_landmark[:, :, :2]) * self.config.w_lmks
            losses['shape_reg'] = (
                torch.sum(self.shape ** 2) / 2) * self.config.w_shape_reg  # *1e-4
            losses['expression_reg'] = (
                torch.sum(self.exp ** 2) / 2) * self.config.w_expr_reg  # *1e-4
            losses['pose_reg'] = (
                torch.sum(self.pose ** 2) / 2) * self.config.w_pose_reg

            # render
            if photometric:
                albedos = self.flametex(self.tex) / 255.
                ops = self.render(vertices, trans_vertices,
                                  albedos, self.lights)
                predicted_images = ops['images']
                losses['photometric_texture'] = (
                    image_masks * (ops['images'] - images).abs()).mean() * self.config.w_pho

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt.zero_grad()
            all_loss.backward()
            e_opt.step()

            loss_info = '----iter: {}, time: {}\n'.format(
                k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            for key in losses.keys():
                loss_info = loss_info + \
                    '{}: {}, '.format(key, float(losses[key]))

            if k % 10 == 0:
                print(loss_info)

            # visualize
            if k % 10 == 0:
                grids = {}
                visind = range(bz)  # [0]
                grids['images'] = torchvision.utils.make_grid(
                    images[visind]).detach().cpu()
                grids['landmarks_gt'] = torchvision.utils.make_grid(
                    tensor_vis_landmarks(images[visind], landmarks[visind]))
                grids['landmarks2d'] = torchvision.utils.make_grid(
                    tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                grids['landmarks3d'] = torchvision.utils.make_grid(
                    tensor_vis_landmarks(images[visind], landmarks3d[visind]))
                # grids['albedoimage'] = torchvision.utils.make_grid(
                #     (ops['albedo_images'])[visind].detach().cpu())
                # grids['render'] = torchvision.utils.make_grid(
                #     predicted_images[visind].detach().float().cpu())
                shape_images = self.render.render_shape(
                    vertices, trans_vertices, images)
                grids['shape'] = torchvision.utils.make_grid(
                    F.interpolate(shape_images[visind], (self.image_size[1], self.image_size[0]))).detach().float().cpu()

                # grids['tex'] = torchvision.utils.make_grid(F.interpolate(albedos[visind], [224, 224])).detach().cpu()
                grid = torch.cat(list(grids.values()), 1)
                grid_image = (grid.numpy().transpose(
                    1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                grid_image = np.minimum(np.maximum(
                    grid_image, 0), 255).astype(np.uint8)

                cv2.imwrite('{}/{}.jpg'.format(savefolder, k), grid_image)

        single_params = {
            'shape': self.shape.detach().cpu().numpy(),
            'exp': self.exp.detach().cpu().numpy(),
            'pose': self.pose.detach().cpu().numpy(),
            'cam': self.cam.detach().cpu().numpy(),
            'verts': trans_vertices.detach().cpu().numpy(),
            # 'albedos': albedos.detach().cpu().numpy(),
            'tex': self.tex.detach().cpu().numpy(),
            'lit': self.lights.detach().cpu().numpy()
        }
        return single_params

    def camera_calib(self, dataloader, epochs=20, savefolder="./"):
        print('----------------------camera calib-----------------------')
        for i in range(epochs):
            print(f"epoch {i}")
            for batch_idx, data in enumerate(tqdm(dataloader)):
                self.batch_idx = batch_idx
                landmarks, indices, images = data
                if images[0] is None:
                    images = None
                bz = landmarks.shape[0]
                shape = self.shape[0].expand(bz, -1)
                tex = self.tex[indices[0]:indices[-1] + 1]
                expression = self.exp[indices[0]:indices[-1] + 1]
                light = self.lights[indices[0]:indices[-1] + 1]
                pose = self.pose[indices[0]:indices[-1] + 1]
                t = self.cam_t[indices[0]:indices[-1] + 1]
                self._fit_landmarks(landmarks, self.cam, shape, tex, expression, light,
                                    pose, t, images=images, update_camera=True, update_shape=True)
            print(self.cam)
            self.cam.save(os.path.join(savefolder, "camera.json"))

    def identity_fitting(self, dataloader, epochs=15, savefolder="./"):
        print('--------------------identity fitting---------------------')
        for i in range(epochs):
            print(f"epoch {i}")
            for batch_idx, data in enumerate(tqdm(dataloader)):
                self.batch_idx = batch_idx
                landmarks, indices, images = data
                if images[0] is None:
                    images = None
                bz = landmarks.shape[0]
                shape = self.shape[0].expand(bz, -1)
                tex = self.tex[indices[0]:indices[-1] + 1]
                expression = self.exp[indices[0]:indices[-1] + 1]
                light = self.lights[indices[0]:indices[-1] + 1]
                pose = self.pose[indices[0]:indices[-1] + 1]
                t = self.cam_t[indices[0]:indices[-1] + 1]
                self._fit_landmarks(landmarks, self.cam, shape, tex, expression, light,
                                    pose, t, images=images, update_camera=False, update_shape=True)
            # save shape parameters
            results = {
                "shape": self.shape[0].cpu().detach().numpy().tolist()
            }
            with open(os.path.join(savefolder, "identity.json"), "w") as f:
                json.dump(results, f, indent=4)

    def fit(self, dataloader):
        print('--------------------fitting---------------------')
        results = {}
        for batch_idx, data in enumerate(tqdm(dataloader)):
            self.batch_idx = batch_idx
            images, landmarks, indices, image_names = data
            if images[0] is None:
                images = None
            bz = landmarks.shape[0]
            shape = self.shape[0].expand(bz, -1)
            tex = self.tex[indices[0]:indices[-1] + 1]
            expression = self.exp[indices[0]:indices[-1] + 1]
            light = self.lights[indices[0]:indices[-1] + 1]
            pose = self.pose[indices[0]:indices[-1] + 1]
            t = self.cam_t[indices[0]:indices[-1] + 1]
            self._fit_landmarks(landmarks, self.cam, shape, tex, expression, light,
                                pose, t, images=images, update_camera=False, update_shape=False,
                                rigid_iter=100, full_iter=200)
            for idx, name in zip(indices, image_names):
                results[name] = {
                    "expression": self.exp[idx].cpu().detach().numpy().tolist(),
                    "pose": self.pose[idx].cpu().detach().numpy().tolist(),
                    "cam_t": self.cam_t[idx].cpu().detach().numpy().tolist(),
                    "shape": self.shape[0].cpu().detach().numpy().tolist()
                }
        return results
        

    def _fit_landmarks(self, landmarks, camera,
                       shape_params, tex_params, expression_params, light_params,
                       pose_params, cam_t,
                       photometric=False, update_camera=False, update_shape=True,
                       images=None, log=True,
                       rigid_iter=20, full_iter=50
                       ):
        bz = landmarks.shape[0]

        shape_params = nn.Parameter(shape_params)
        tex_params = nn.Parameter(tex_params)
        expression_params = nn.Parameter(expression_params)
        light_params = nn.Parameter(light_params)
        pose_params = nn.Parameter(pose_params)
        cam_t = nn.Parameter(cam_t)
        # set the parameters that needs to be updated
        opt_params = [tex_params, expression_params, light_params,
                      pose_params, cam_t]
        opt_rigid_params = [pose_params, cam_t]
        if update_camera:
            opt_params += camera.get_params()
            opt_rigid_params += camera.get_params()
        if update_shape:
            opt_params += [shape_params]
        # init optimizer
        e_opt = torch.optim.Adam(
            opt_params,
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )
        e_opt_rigid = torch.optim.Adam(
            opt_rigid_params,
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )
        gt_landmark = landmarks
        for k in range(rigid_iter):
            losses = {}
            vertices, landmarks2d, landmarks3d = self.forward(
                shape_params=shape_params, expression_params=expression_params, pose_params=pose_params)
            trans_vertices = self.project_fun(vertices, camera, cam_t)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = self.project_fun(landmarks2d, camera, cam_t)
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = self.project_fun(landmarks3d, camera, cam_t)
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['landmark'] = l2_distance(
                landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) * self.config.w_lmks

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt_rigid.zero_grad()
            all_loss.backward()
            e_opt_rigid.step()

        for k in range(rigid_iter, full_iter):
            losses = {}
            vertices, landmarks2d, landmarks3d = self.forward(
                shape_params=shape_params, expression_params=expression_params, pose_params=pose_params)
            trans_vertices = self.project_fun(vertices, camera, cam_t)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]
            landmarks2d = self.project_fun(landmarks2d, camera, cam_t)
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]
            landmarks3d = self.project_fun(landmarks3d, camera, cam_t)
            landmarks3d[..., 1:] = - landmarks3d[..., 1:]

            losses['landmark'] = l2_distance(
                landmarks2d[:, :, :2], gt_landmark[:, :, :2]) * self.config.w_lmks
            losses['shape_reg'] = (
                torch.sum(self.shape ** 2) / 2) * self.config.w_shape_reg  # *1e-4
            losses['expression_reg'] = (
                torch.sum(self.exp ** 2) / 2) * self.config.w_expr_reg  # *1e-4
            losses['pose_reg'] = (
                torch.sum(self.pose ** 2) / 2) * self.config.w_pose_reg

            # render
            if photometric:
                albedos = self.flametex(self.tex) / 255.
                ops = self.render(vertices, trans_vertices,
                                  albedos, self.lights)
                predicted_images = ops['images']
                # losses['photometric_texture'] = (
                #     image_masks * (ops['images'] - images).abs()).mean() * self.config.w_pho

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt.zero_grad()
            all_loss.backward()
            e_opt.step()

        # visualize
        if log:
            grids = {}
            visind = range(bz)  # [0]
            grids['images'] = torchvision.utils.make_grid(
                images[visind]).detach().cpu()
            grids['landmarks_gt'] = torchvision.utils.make_grid(
                tensor_vis_landmarks(images[visind], landmarks[visind]))
            grids['landmarks2d'] = torchvision.utils.make_grid(
                tensor_vis_landmarks(images[visind], landmarks2d[visind]))
            grids['landmarks3d'] = torchvision.utils.make_grid(
                tensor_vis_landmarks(images[visind], landmarks3d[visind]))
            # grids['albedoimage'] = torchvision.utils.make_grid(
            #     (ops['albedo_images'])[visind].detach().cpu())
            # grids['render'] = torchvision.utils.make_grid(
            #     predicted_images[visind].detach().float().cpu())
            shape_images = self.render.render_shape(
                vertices, trans_vertices, images)
            grids['shape'] = torchvision.utils.make_grid(
                F.interpolate(shape_images[visind], (self.image_size[1], self.image_size[0]))).detach().float().cpu()

            # grids['tex'] = torchvision.utils.make_grid(F.interpolate(albedos[visind], [224, 224])).detach().cpu()
            grid = torch.cat(list(grids.values()), 1)
            grid_image = (grid.numpy().transpose(
                1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
            grid_image = np.minimum(np.maximum(
                grid_image, 0), 255).astype(np.uint8)

            cv2.imwrite('{}/batch_{}.jpg'.format(self.config.savefolder,
                        self.batch_idx), grid_image)

    def save_obj(self, params, save_path="./Results/test.obj"):
        self.setup_camera(is_perspective=False)
        vertices, _, _ = self.forward(shape_params=torch.from_numpy(params['shape']).to(self.device),
                                      expression_params=torch.from_numpy(
            params['exp']).to(self.device),
            pose_params=torch.from_numpy(params['pose']).to(self.device))
        trans_vertices = self.project_fun(vertices, self.cam, self.cam_t)
        trans_vertices[..., 1:] = - trans_vertices[..., 1:]

        # save the vertices of first face
        trans_vertices = trans_vertices[0]
        self.render.save_obj(filename=save_path, vertices=trans_vertices)

    def render_shape(self, params):
        self.setup_camera(is_perspective=False)
        vertices, _, _ = self.forward(shape_params=torch.from_numpy(params['shape']).to(self.device),
                                      expression_params=torch.from_numpy(
            params['exp']).to(self.device),
            pose_params=torch.from_numpy(params['pose']).to(self.device))
        trans_vertices = self.project_fun(vertices, self.cam, self.cam_t)
        trans_vertices[..., 1:] = - trans_vertices[..., 1:]
        shape_images = self.render.render_shape(vertices, trans_vertices,)
        shape_images = (shape_images.cpu().numpy().transpose(
            0, 2, 3, 1).copy() * 255)[:, :, :, [2, 1, 0]]
        return shape_images


class FLAMETex(nn.Module):
    """
    current FLAME texture are adapted from BFM Texture Model
    """

    def __init__(self, config):
        super(FLAMETex, self).__init__()
        tex_params = config.tex_params
        tex_space = np.load(config.tex_space_path)
        texture_mean = tex_space['mean'].reshape(1, -1)
        texture_basis = tex_space['tex_dir'].reshape(-1, 200)
        num_components = texture_basis.shape[1]
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(
            texture_basis[:, :tex_params]).float()[None, ...]
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)

    def forward(self, texcode):
        texture = self.texture_mean + \
            (self.texture_basis * texcode[:, None, :]).sum(-1)
        texture = texture.reshape(
            texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = F.interpolate(texture, [256, 256])
        texture = texture[:, [2, 1, 0], :, :]
        return texture
