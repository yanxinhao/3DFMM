# coding=utf-8
"""
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
Date: 2021-09-24 13:24:46
LastEditTime: 2021-09-24 13:24:46
LastEditors: yanxinhao
FilePath: /3DFMM/experiments/camera_calib.py
Description: 
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
        "camera_params": 3,
        "shape_params": 100,
        "expression_params": 50,
        "pose_params": 6,
        "tex_params": 50,
        "use_face_contour": True,
        "cropped_size": 512,
        "batch_size": 1,
        # 'image_size': (456, 352),
        "image_size": (562, 762),
        "e_lr": 0.005,
        "e_wd": 0.0001,
        # "savefolder": "./Results/dave_dvp_principle0",
        # "image_dir": "./Data/dave_dvp/train/",
        "savefolder": "./tmp/",
        "image_dir": "/p300/WorkSpace/NeuralRendering/NDface/datasets/face_images/KDEF/AM28/",
        # weights of losses and reg terms
        "w_pho": 8,
        "w_lmks": 1,
        "w_shape_reg": 1e-4,
        "w_expr_reg": 1e-4,
        "w_pose_reg": 0,
    }
    config_flame = dict2obj(config_flame)
    w, h = config_flame.image_size
    print("loading flame model")
    flame = MorphableModel(config=config_flame, model_type="FLAME")

    # image_path = "./Data/dave_dvp/train/f_0121.png"
    image_dir = config_flame.image_dir
    images = []
    image_names = os.listdir(image_dir)
    # image_names.sort(key=lambda name: int(name[2:6]))
    for i, image_name in enumerate(image_names):
        if i == 50:
            break
        image_path = image_dir + image_name
        image = cv2.imread(image_path)
        image = cv2.resize(image, config_flame.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    images = np.array(images)
    flame.camera_calib(images=images, is_perspective=True)


if __name__ == "__main__":
    main()
