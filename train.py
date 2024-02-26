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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

#主函数
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0 #初始化迭代次数。
    tb_writer = prepare_output_and_logger(dataset)  #设置 TensorBoard 写入器和日志记录器。
    gaussians = GaussianModel(dataset.sh_degree) #（重点看，需要转跳）创建一个 GaussianModel 类的实例，输入一系列参数，其参数取自数据集。
    scene = Scene(dataset, gaussians) #（这个类的主要目的是处理场景的初始化、保存和获取相机信息等任务，）创建一个 Scene 类的实例，使用数据集和之前创建的 GaussianModel 实例作为参数。
    gaussians.training_setup(opt) #设置 GaussianModel 的训练参数。
    if checkpoint: #如果有提供检查点路径。
        (model_params, first_iter) = torch.load(checkpoint)#通过 torch.load(checkpoint) 加载检查点的模型参数和起始迭代次数。
        gaussians.restore(model_params, opt)#通过 gaussians.restore 恢复模型的状态。

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0] #设置背景颜色，根据数据集是否有白色背景来选择。
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") #将背景颜色转化为 PyTorch Tensor，并移到 GPU 上。

    # 创建两个 CUDA 事件，用于测量迭代时间。
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress") #创建一个 tqdm 进度条，用于显示训练进度。
    first_iter += 1
    # 接下来开始循环迭代
    for iteration in range(first_iter, opt.iterations + 1): #主要的训练循环开始。       
        if network_gui.conn == None: #检查 GUI 是否连接，如果连接则接收 GUI 发送的消息。
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record() #用于测量迭代时间。

        gaussians.update_learning_rate(iteration) #更新学习率。

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree() #每 1000 次迭代，增加球谐函数的阶数。

        # Pick a random Camera （随机选择一个训练相机。）
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render （渲染图像，计算损失（L1 loss 和 SSIM loss））
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) #计算渲染的图像与真实图像之间的loss
        loss.backward() #更新损失。loss反向传播

        iter_end.record() #用于测量迭代时间。

        with torch.no_grad(): #记录损失的指数移动平均值，并定期更新进度条。
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations): #如果达到保存迭代次数，保存场景。
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification（在一定的迭代次数内进行密集化处理。）
            if iteration < opt.densify_until_iter: #在达到指定的迭代次数之前执行以下操作。
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter]) #将每个像素位置上的最大半径记录在 max_radii2D 中。这是为了密集化时进行修剪（pruning）操作时的参考。
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter) #将与密集化相关的统计信息添加到 gaussians 模型中，包括视图空间点和可见性过滤器。

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0: #在指定的迭代次数之后，每隔一定的迭代间隔进行以下密集化操作。
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None #根据当前迭代次数设置密集化的阈值。如果当前迭代次数大于 opt.opacity_reset_interval，则设置 size_threshold 为 20，否则为 None。
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold) #执行密集化和修剪操作，其中包括梯度阈值、密集化阈值、相机范围和之前计算的 size_threshold。
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter): #在每隔一定迭代次数或在白色背景数据集上的指定迭代次数时，执行以下操作。
                    gaussians.reset_opacity() #重置模型中的某些参数，涉及到透明度的操作，具体实现可以在 reset_opacity 方法中找到。

            # Optimizer step（执行优化器的步骤，然后清零梯度。）
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # 如果达到检查点迭代次数，保存检查点。
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer: #将 L1 loss、总体 loss 和迭代时间写入 TensorBoard。
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 在指定的测试迭代次数，进行渲染并计算 L1 loss 和 PSNR。
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # 获取渲染结果和真实图像
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):  # 在 TensorBoard 中记录渲染结果和真实图像
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                     # 计算 L1 loss 和 PSNR
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                 # 计算平均 L1 loss 和 PSNR
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])  
                # 在控制台打印评估结果        
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                # 在 TensorBoard 中记录评估结果
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # 在 TensorBoard 中记录场景的不透明度直方图和总点数。
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()#使用 torch.cuda.empty_cache() 清理 GPU 内存。

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port) #这行代码初始化一个 GUI 服务器，使用 args.ip 和 args.port 作为参数。这可能是一个用于监视和控制训练过程的图形用户界面的一部分。
    torch.autograd.set_detect_anomaly(args.detect_anomaly) #这行代码设置 PyTorch 是否要检测梯度计算中的异常。
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    # 输入的参数包括：模型的参数（传入的为数据集的位置）、优化器的参数、其他pipeline的参数，测试迭代次数、保存迭代次数 、检查点迭代次数 、开始检查点 、调试起点

    # All done
    print("\nTraining complete.")
