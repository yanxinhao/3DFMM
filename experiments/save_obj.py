# coding=utf-8
"""
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2021-06-02 03:25:56
LastEditors: yanxinhao
FilePath: /3DFMM/experiments/save_obj.py
Date: 2021-06-02 03:25:56
Description: 
"""
import sys
import torch
import os
import numpy as np
import json

sys.path.append("./core")
from morphable_model import MorphableModel, dict2obj, check_mkdir


def main():
    config_flame = {
        # FLAME
        # acquire it from FLAME project page
        "flame_model_path": "./Data/morphable_model/FLAME/generic_model.pkl",
        "flame_lmk_embedding_path": "./Data/morphable_model/FLAME/landmark_embedding.npy",
        # acquire it from FLAME project page
        "tex_space_path": "./Data/morphable_model/FLAME/FLAME_texture.npz",
        "image_size": (512, 512),
        "shape_params": 100,
        "expression_params": 50,
        # data path
        "savefolder": "./Results/dave_dvp/val",
        "file_path": "./Results/dave_dvp/val/data.json",
    }
    config_flame = dict2obj(config_flame)
    flame = MorphableModel(config=config_flame, model_type="FLAME")

    with open(config_flame.file_path, "r") as f:
        params = json.load(f)
    check_mkdir(config_flame.savefolder)
    for img_name, data in params.items():
        img_name = img_name.split(".")[0]
        expression = torch.tensor(data["expression"], dtype=torch.float32).unsqueeze(0)
        shape = torch.tensor(data["shape"], dtype=torch.float32).unsqueeze(0)
        pose = torch.tensor(data["pose"], dtype=torch.float32).unsqueeze(0)
        save_path = os.path.join(config_flame.savefolder, img_name + ".obj")
        flame.write_obj(
            file_path=save_path, expression=expression, shape=shape, pose=pose
        )


if __name__ == "__main__":
    main()
