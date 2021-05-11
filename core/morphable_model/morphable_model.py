# coding=utf-8
'''
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2021-05-10 14:32:09
LastEditors: yanxinhao
Description: A basic class of 3DMM
'''
import os
import math
import cv2
import json
# from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
import numpy as np
import torch
from .BFM import BFM
from .Flame import FLAME
from .util import check_mkdir


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
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=self.device, flip_input=True,
                                               face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)

    def generate_vertices(self, params):
        vertices = self.model(params)
        return vertices

    def write_obj(self, params, file_path="./Results/test.obj"):
        self.model.save_obj(params, file_path)

    def render_shape(self, params):
        images = self.model.render_shape(params)
        return images

    def get_2d_landmarks(self, image):
        """detect the faces in the image and get the landmarks of faces

        Args:
            image (np.array): rgb image
        """
        preds = self.fa.get_landmarks(image)
        return preds

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

    def _get_dataloader(self, images=None, landmarks=None, batch_size=1):
        # step 1: get landmarks
        if landmarks is None:
            data_size = images.shape[0]
            images = torch.from_numpy(images.transpose(0, 3, 1, 2))
            landmarks = self.fa.get_landmarks_from_batch(image_batch=images)
            landmarks = torch.from_numpy(np.array(landmarks)).to(self.device)
            # preprocess
            landmarks[:, :, 0] = landmarks[:, :, 0] / \
                float(self.image_size[0]) * 2 - 1
            landmarks[:, :, 1] = landmarks[:, :, 1] / \
                float(self.image_size[1]) * 2 - 1
            images = images / 255.0
        else:
            data_size = landmarks.shape[0]
        # step 2: setup dataloader

        class Data(Dataset):
            """Dataset for camera calibration
               For different images/landmarks of the same person :
                    shape_params and camera_intrinsic should be the same;
                    pose,expression,cam_t,lighting and texture params are various
            Args:
                Dataset ([type]): [description]
            """

            def __init__(self, landmarks, images=None):
                """the landmarks dataset

                Args:
                    landmarks (tensor): B*68*2
                    images (tensor, optional): B*3*h*w. Defaults to None.
                """
                super(Data, self).__init__()
                self.landmarks = landmarks
                self.images = images

            def __len__(self):
                return self.landmarks.shape[0]

            def __getitem__(self, index):
                if self.images is None:
                    return self.landmarks[index], index, None
                else:
                    return self.landmarks[index], index, self.images[index]

        dataset = Data(landmarks=landmarks, images=images)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        return dataloader, data_size

    def camera_calib(self, images=None, landmarks=None, is_perspective=True):
        check_mkdir(self.config.savefolder)
        assert images is not None or landmarks is not None
        dataloader, data_size = self._get_dataloader(
            images=images, landmarks=landmarks)
        #  init camera and parameters,camera is 'self.model.cam'
        self.model.setup_camera(is_perspective=is_perspective)
        self.model.initialize_params(batch_size=data_size)
        self.model.camera_calib(dataloader, savefolder=self.config.savefolder)

    def fit_identity(self, camera_path, images=None, landmarks=None):
        check_mkdir(self.config.savefolder)
        assert images is not None or landmarks is not None
        dataloader, data_size = self._get_dataloader(
            images=images, landmarks=landmarks)
        #  init camera and parameters,camera is 'self.model.cam'
        self.model.setup_camera(camera_path=camera_path)
        self.model.initialize_params(batch_size=data_size)
        self.model.identity_fitting(
            dataloader, savefolder=self.config.savefolder)

    def fit_dir(self, shape_path, camera_path, dir_path, start_index=0, chunk=512, sort_fun=lambda name: int(name[2:6])):
        check_mkdir(self.config.savefolder)
        # step 1: get images and image_names
        image_names = os.listdir(dir_path)
        image_names.sort(key=sort_fun)

        class Data(Dataset):
            def __init__(self, images, image_names, detector, device):
                super(Data, self).__init__()
                self.images = images
                self.image_names = image_names
                self.detector = detector
                self.device = device

            def __len__(self):
                return self.images.shape[0]

            def __getitem__(self, index):
                image = self.images[index]
                _, h, w = image.shape
                # get landmarks
                # only get one person's landmarks
                landmarks = self.detector.get_landmarks_from_batch(image.unsqueeze(0))[
                    0]
                landmarks = torch.from_numpy(
                    np.array(landmarks)).to(self.device)
                # preprocess
                landmarks[:, 0] = landmarks[:, 0] / \
                    float(w) * 2 - 1
                landmarks[:, 1] = landmarks[:, 1] / \
                    float(h) * 2 - 1
                image = image / 255.0
                return image, landmarks, index, self.image_names[index]
        if start_index != 0:
            with open(os.path.join(self.config.savefolder, "data.json"), "r") as file:
                data = json.load(file)
        else:
            data = {}
        chuck_nums = math.ceil(len(image_names) / chunk)
        # load all images,minibatches to avoid OOM.
        for i in range(start_index, len(image_names), chunk):
            images = []
            names = image_names[i:i + chunk]
            print(
                f'------------fit_dir :total chuck nums:{chuck_nums}, processing No. {i+1}--------------')
            for image_name in names:
                image_path = os.path.join(dir_path, image_name)
                image = cv2.imread(image_path)
                image = cv2.resize(image, self.config.image_size)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
            images = torch.from_numpy(
                np.array(images).transpose(0, 3, 1, 2)).to(self.device)
            # step 2: load shape and cameras; init parameters of 3DMM
            bz = images.shape[0]
            self.model.initialize_params(batch_size=bz)
            self.model.load_shape(shape_path)
            self.model.setup_camera(camera_path=camera_path)
            dataset = Data(images, names, self.fa, device=self.device)
            dataloader = DataLoader(dataset, batch_size=1)
            results = self.model.fit(dataloader)
            data.update(results)
            with open(os.path.join(self.config.savefolder, "data.json"), "w") as file:
                json.dump(data, file, indent=4)
