# coding=utf-8
"""
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2021-06-02 11:35:42
LastEditors: yanxinhao
FilePath: /3DFMM/experiments/render_shape.py
Date: 2021-06-02 11:35:41
Description: 
"""
import cv2
import os
import torch
import json
import sys
import numpy as np

sys.path.append("./core")
from morphable_model import MorphableModel, dict2obj


def main():
    # test flame
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
        "image_size": (512, 512),
        "savefolder": "./Results/dave_dvp/val",
        "file_path": "./Results/dave_dvp/val/data.json",
        "camera_path": "./Results/dave_dvp/camera.json",
    }
    config_flame = dict2obj(config_flame)
    flame = MorphableModel(config=config_flame, model_type="FLAME")

    with open(config_flame.file_path, "r") as f:
        params = json.load(f)

    for img_name, data in params.items():
        img_name = img_name.split(".")[0]
        data["pose"] = torch.tensor(data["pose"], dtype=torch.float32).unsqueeze(0)
        data["expression"] = torch.tensor(
            data["expression"], dtype=torch.float32
        ).unsqueeze(0)
        data["shape"] = torch.tensor(data["shape"], dtype=torch.float32).unsqueeze(0)
        data["camera_path"] = config_flame.camera_path
        data["cam_t"] = torch.tensor(data["cam_t"], dtype=torch.float32).unsqueeze(0)
        shape_images = flame.render_shape(**data)
        cv2.imwrite(
            os.path.join(config_flame.savefolder, f"{img_name}_shape.png"),
            shape_images[0],
        )

    images = flame.render_init_shape(is_perspective=False)
    cv2.imwrite(
        os.path.join(config_flame.savefolder, f"init_shape.png"),
        images[0],
    )
    # # adjust poses
    # radian = np.pi / 180.0
    # # global pose of face
    # poses = radian * np.array(
    #     [
    #         [[-90, 0, 0], [-45, 0, 0], [0, 0, 0], [45, 0, 0], [90, 0, 0]],
    #         [[0, -90, 0], [0, -45, 0], [0, 0, 0], [0, 45, 0], [0, 90, 0]],
    #         [[0, 0, -90], [0, 0, -45], [0, 0, 0], [0, 0, 45], [0, 0, 90]],
    #     ]
    # )
    # axis = ["x", "y", "z"]
    # params = params["f_0000.png"]
    # for i in range(poses.shape[0]):
    #     for j in range(poses.shape[1]):
    #         params["pose"] = np.array(params["pose"], dtype=np.float32)
    #         params["pose"][:3] = poses[i][j]
    #         params["expression"] = np.array(params["expression"], dtype=np.float32)
    #         params["shape"] = np.array(params["shape"], dtype=np.float32)
    #         params["camera_path"] = config_flame.camera_path
    #         params["cam_t"] = np.array(params["cam_t"], dtype=np.float32)
    #         shape_images = flame.render_shape(**params)
    #         cv2.imwrite(
    #             os.path.join(config_flame.savefolder, f"{axis[i]}_{j}.png"),
    #             shape_images[0],
    #         )


if __name__ == "__main__":
    main()
