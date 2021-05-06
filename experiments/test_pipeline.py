# coding=utf-8
'''
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2021-05-04 15:33:16
LastEditors: yanxinhao
Description: test these modules in core folder
'''
import cv2
import sys
import numpy as np
sys.path.append('./core')
from morphable_model import MorphableModel, dict2obj


def main():
    # test bfm
    # config_bfm = {'model_path': './Data/morphable_model/BFM/BFM.mat',
    #               'image_size': (512, 512), }
    # config_bfm = dict2obj(config_bfm)
    # bfm = MorphableModel(config=config_bfm, model_type='BFM')

    # test flame
    config_flame = {
        # FLAME
        # acquire it from FLAME project page
        'flame_model_path': './Data/morphable_model/FLAME/generic_model.pkl',
        'flame_lmk_embedding_path': './Data/morphable_model/FLAME/landmark_embedding.npy',
        # acquire it from FLAME project page
        'tex_space_path': './Data/morphable_model/FLAME/FLAME_texture.npz',
        'camera_params': 3,
        'shape_params': 100,
        'expression_params': 50,
        'pose_params': 6,
        'tex_params': 50,
        'use_face_contour': True,

        'cropped_size': 256,
        'batch_size': 1,
        'image_size': (456, 352),
        # 'image_size': (256, 256),
        'e_lr': 0.005,
        'e_wd': 0.0001,
        'savefolder': './Results/',
        # weights of losses and reg terms
        'w_pho': 8,
        'w_lmks': 1,
        'w_shape_reg': 1e-4,
        'w_expr_reg': 1e-4,
        'w_pose_reg': 0,
    }
    config_flame = dict2obj(config_flame)
    flame = MorphableModel(config=config_flame, model_type='FLAME')

    # image_path = "./Data/dave_dvp/train/f_0121.png"
    image_path = "./Data/download.jpeg"
    image = cv2.imread(image_path)
    image = cv2.resize(image, config_flame.image_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    params = flame.fitting_image(image)
    np.save("./Results/test_params.npy", params)

    params = np.load("./Results/test_params.npy", allow_pickle=True).item()
    flame.write_obj(params)

    # adjust poses
    radian = np.pi / 180.0
    # global pose of face
    poses = radian * np.array([
        [[-90, 0, 0], [-45, 0, 0], [0, 0, 0], [45, 0, 0], [90, 0, 0]],
        [[0, -90, 0], [0, -45, 0], [0, 0, 0], [0, 45, 0], [0, 90, 0]],
        [[0, 0, -90], [0, 0, -45], [0, 0, 0], [0, 0, 45], [0, 0, 90]],
    ])
    axis = ['x', 'y', 'z']
    for i in range(poses.shape[0]):
        for j in range(poses.shape[1]):
            params['pose'][..., :3] = poses[i][j]
            shape_images = flame.render_shape(params)
            cv2.imwrite(
                f"./Results/image_shape_{axis[i]}_{j}.png", shape_images[0])


if __name__ == '__main__':
    main()
