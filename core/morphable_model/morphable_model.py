# coding=utf-8
'''
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2021-04-24 05:47:43
LastEditors: yanxinhao
Description: A basic class of 3DMM
'''
import numpy as np
import torch
from .BFM import BFM
from .Flame import FLAME


class MorphableModel(object):
    """docstring for morphable model

    Args:
        object ([type]): [description]
    """

    def __init__(self, config=None, model_type='BFM', device="cuda"):
        """constructor of MorphableModel

        Args:
            model_path (string): the path to model file
            model_type (string): the specific model_type like 'BFM' or 'FLAME'
        """
        super(MorphableModel, self).__init__()
        self.config = config
        self.device = device
        self.image_size = config.image_size
        if model_type == 'BFM':
            self.model = BFM(config.model_path)
        elif model_type == 'FLAME':
            self.model = FLAME(
                config, image_size=self.image_size).to(self.device)

        # initilize landmark detector
        self.initilization()

    def initilization(self):
        import sys
        sys.path.append("./core/detector/")
        import face_alignment
        face_detector = 'sfd'
        face_detector_kwargs = {
            "filter_threshold": 0.8
        }
        # Run the 3D face alignment on a test image, without CUDA.
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=True,
                                               face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)

    def generate_vertices(self, params):
        vertices = self.model(params)
        return vertices

    def write_obj(self, params, file_path="./Results/test.obj"):
        self.model.save_obj(params, file_path)

    def render_shape(self, params):
        images = self.model.render_shape(params)
        return images

    def fitting_image(self, image):
        """fit 3DMM in one image 

        Args:
            image (np.array): rgb image

        Returns:
            params: a dict of 3DMM parameters
        """
        landmarks = np.array(self.get_2d_landmarks(image))
        # fitting by landmarks
        landmarks[:, :, 0] = landmarks[:, :, 0] / float(image.shape[1]) * 2 - 1
        landmarks[:, :, 1] = landmarks[:, :, 1] / float(image.shape[0]) * 2 - 1
        params = self.model.optimize_by_landmark(landmarks=landmarks, images=torch.from_numpy(
            np.array([image.transpose(2, 0, 1) / 255.])))
        return params

    def get_2d_landmarks(self, image):
        """detect the faces in the image and get the landmarks of faces

        Args:
            image (np.array): rgb image
        """
        preds = self.fa.get_landmarks(image)
        return preds
