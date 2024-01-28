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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    # 构建渲染结果和ground truth保存路径
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    # 确保渲染结果和ground truth保存路径存在
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # 遍历所有视图进行渲染
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        # 调用 render 函数执行渲染，获取渲染结果
        rendering = render(view, gaussians, pipeline, background)["render"] #这里执行的就是上面解析过的render的代码了~

        # 获取视图的ground truth
        gt = view.original_image[0:3, :, :]

        # 保存渲染结果和ground truth为图像文件
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad(): # 禁用梯度计算，因为在渲染过程中不需要梯度信息
        gaussians = GaussianModel(dataset.sh_degree) # 创建一个 GaussianModel 对象，用于处理高斯模型
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)  # 创建一个 Scene 对象，用于处理场景的渲染

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0] # 根据数据集的背景设置，定义背景颜色
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # 将背景颜色转换为 PyTorch 张量，同时将其移到 GPU 上

        if not skip_train:  # 如果不跳过训练数据集的渲染
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background) # 调用 render_set 函数渲染训练数据集

        if not skip_test:  # 如果不跳过测试数据集的渲染
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background) # 调用 render_set 函数渲染测试数据集

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)