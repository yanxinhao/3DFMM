# coding=utf-8
"""
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2021-09-18 13:09:21
LastEditors: yanxinhao
FilePath: /3DFMM/tools/generate_json.py
Date: 2021-09-18 13:09:21
Description: 
"""
import os
import argparse
import json
import pickle

config_flame = {
    "flame_model_path": "./Data/morphable_model/FLAME/generic_model.pkl",
    "shape_params": 100,
    "expression_params": 50,
    "pose_params": 6,
    "tex_params": 50,
}


def main(args):
    json_dict = {}
    dim_shape = config_flame["shape_params"]
    dim_exp = config_flame["expression_params"]
    dim_pose = config_flame["pose_params"]
    dim_tex = config_flame["tex_params"]
    with open(config_flame["flame_model_path"], "rb") as f:
        ss = pickle.load(f, encoding="latin1")
    print(ss)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate json file")
    parser.add_argument(
        "-o", "--out_folder", type=str, default="./Data/face_specific_params/"
    )
    parser.add_argument("-fname", "--file_name", type=str, default="template")
    args = parser.parse_args()
    os.makedirs(args.out_folder, exist_ok=True)
    main(args)
