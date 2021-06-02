# coding=utf-8
"""
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2021-06-02 02:49:02
LastEditors: yanxinhao
Description: 
Date: 2021-06-02 02:49:01
FilePath: /3DFMM/experiments/identity_fit.py
"""
import os
import cv2
import sys
import numpy as np

sys.path.append("./core")
from morphable_model import MorphableModel, dict2obj


def main():
    config_flame = {
        # FLAME
        # acquire it from FLAME project page
        "flame_model_path": "./Data/morphable_model/FLAME/generic_model.pkl",
        "flame_lmk_embedding_path": "./Data/morphable_model/FLAME/landmark_embedding.npy",
        # acquire it from FLAME project page
        "tex_space_path": "./Data/morphable_model/FLAME/FLAME_texture.npz",
        "camera_path": "./Results/dave_dvp/camera.json",
        "shape_params": 100,
        "expression_params": 50,
        "pose_params": 6,
        "tex_params": 50,
        "use_face_contour": True,
        "cropped_size": 512,
        "batch_size": 1,
        # 'image_size': (456, 352),
        "image_size": (512, 512),
        "e_lr": 0.005,
        "e_wd": 0.0001,
        "savefolder": "./Results/dave_dvp",
        "image_dir": "./Data/dave_dvp/train/",
        # weights of losses and reg terms
        "w_pho": 8,
        "w_lmks": 1,
        "w_shape_reg": 1e-4,
        "w_expr_reg": 1e-4,
        "w_pose_reg": 0,
    }
    config_flame = dict2obj(config_flame)
    w, h = config_flame.image_size
    flame = MorphableModel(config=config_flame, model_type="FLAME")

    image_dir = config_flame.image_dir
    images = []
    image_names = os.listdir(image_dir)
    image_names.sort(key=lambda name: int(name[2:6]))
    for i, image_name in enumerate(image_names):
        if i == 50:
            break
        image_path = image_dir + image_name
        image = cv2.imread(image_path)
        image = cv2.resize(image, config_flame.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    images = np.array(images)
    flame.fit_identity(camera_path=config_flame.camera_path, images=images)


if __name__ == "__main__":
    main()
