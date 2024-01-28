#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    # 初始化方法
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path #将传入的 args 对象中的 model_path 属性赋值给 self.model_path，表示模型的路径。
        self.loaded_iter = None #初始化 self.loaded_iter 为 None，用于存储已加载的模型的迭代次数。
        self.gaussians = gaussians #将传入的高斯模型对象赋值给 self.gaussians 属性。

        if load_iteration: #可选参数，默认为 None。如果提供了值，它将被用作已加载模型的迭代次数。
            if load_iteration == -1: #如果没有提供 load_iteration，则将点云数据和相机信息保存到文件中。
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter)) #输出加载模型的迭代次数的信息。

        self.train_cameras = {}
        self.test_cameras = {}

        # 根据场景的类型（Colmap 或 Blender）加载相应的场景信息，存储在 scene_info 变量中。
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # 保存点云数据和相机信息：
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 随机排序相机
        if shuffle: #可选参数，默认为 True。如果设置为 True，则会对场景中的训练和测试相机进行随机排序。
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # 设置相机的范围：
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 加载训练和测试相机：
        for resolution_scale in resolution_scales: #可选参数，默认为 [1.0]。一个浮点数列表，用于指定训练和测试相机的分辨率缩放因子。
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # 加载或创建高斯模型
        if self.loaded_iter: #如果已加载模型，则调用 load_ply 方法加载点云数据。
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else: #否则，调用 create_from_pcd 方法根据场景信息中的点云数据创建高斯模型。
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):#该方法用于保存点云数据到文件，其参数 iteration 指定了迭代次数。
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):#返回训练相机的列表，可以根据指定的缩放因子 scale 获取相应分辨率的相机列表。
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):#返回测试相机的列表，可以根据指定的缩放因子 scale 获取相应分辨率的相机列表。
        return self.test_cameras[scale]