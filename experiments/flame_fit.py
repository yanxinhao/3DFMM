# coding=utf-8
'''
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2021-05-06 16:47:59
LastEditors: yanxinhao
Description: 
'''
import os
import cv2
import sys
import numpy as np
sys.path.append('./core')
from morphable_model import MorphableModel, dict2obj


def main():
    config_flame = {
        # FLAME
        # acquire it from FLAME project page
        'flame_model_path': './Data/morphable_model/FLAME/generic_model.pkl',
        'flame_lmk_embedding_path': './Data/morphable_model/FLAME/landmark_embedding.npy',
        # acquire it from FLAME project page
        'tex_space_path': './Data/morphable_model/FLAME/FLAME_texture.npz',
        'camera_path': "./Results/dave_dvp/camera.json",
        'shape_path': "./Results/dave_dvp/identity.json",
        'shape_params': 100,
        'expression_params': 50,
        'pose_params': 6,
        'tex_params': 50,
        'use_face_contour': True,

        # 'cropped_size': 256,
        'batch_size': 1,
        # 'image_size': (456, 352),
        'image_size': (256, 256),
        'e_lr': 0.005,
        'e_wd': 0.0001,
        'imagefolder': './Data/dave_dvp/test',
        'savefolder': './Results/dave_dvp/test',
        # weights of losses and reg terms
        'w_pho': 8,
        'w_lmks': 1,
        'w_shape_reg': 1e-4,
        'w_expr_reg': 1e-4,
        'w_pose_reg': 0,
    }
    config_flame = dict2obj(config_flame)
    w, h = config_flame.image_size
    flame = MorphableModel(config=config_flame, model_type='FLAME')

    flame.fit_dir(config_flame.shape_path,
                  config_flame.camera_path, config_flame.imagefolder)


if __name__ == '__main__':
    main()
