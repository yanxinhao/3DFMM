# coding=utf-8
"""
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2021-06-02 02:45:20
LastEditors: yanxinhao
Description: 
Date: 2021-06-02 02:45:19
FilePath: /3DFMM/core/morphable_model/camera.py
"""
import torch
import json


class Camera(object):
    def __init__(self, camera_path=None, is_perspective=True, **kwargs):
        super(Camera, self).__init__()
        if camera_path is not None:
            self.load(camera_path)
        else:
            self.is_perspective = is_perspective
            if is_perspective:
                self.focal_length = kwargs.get("focal_length", None)
                self.principal_point = kwargs.get("principal_point", None)
            else:
                self.scale = kwargs.get("scale", None)

    def get_params(self):
        if self.is_perspective:
            return [self.focal_length, self.principal_point]
        else:
            return [self.scale]

    def __str__(self):
        if self.is_perspective:
            return f"perspective camera : \n focal length is {self.focal_length} ; \n principal_point is {self.principal_point}"
        else:
            return f"ortho camera : \n scale is {self.scale}"

    def save(self, save_path):
        data = {"is_perspective": self.is_perspective}
        if self.is_perspective:
            data["focal_length"] = self.focal_length.cpu().detach().numpy().tolist()
            data["principal_point"] = (
                self.principal_point.cpu().detach().numpy().tolist()
            )
        else:
            data["scale"] = self.scale.cpu().detach().numpy().tolist()
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, file_path):
        with open(file_path, "r") as f:
            camera = json.load(f)
        self.is_perspective = camera["is_perspective"]
        if self.is_perspective:
            self.focal_length = torch.tensor(camera["focal_length"])
            self.principal_point = torch.tensor(camera["principal_point"])
        else:
            self.scale = torch.tesnor(camera["scale"])
