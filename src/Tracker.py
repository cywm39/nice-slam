import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer


class Tracker(object):
    def __init__(self, cfg, args, slam
                 ):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        self.idx = slam.idx
        self.nice = slam.nice
        self.bound = slam.bound
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.shared_c = slam.shared_c
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list

        self.cam_lr = cfg['tracking']['lr']
        self.device = cfg['tracking']['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.seperate_LR = cfg['tracking']['seperate_LR']
        self.w_color_loss = cfg['tracking']['w_color_loss']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.handle_dynamic = cfg['tracking']['handle_dynamic']
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        # Tracker和Mapper中都加载一遍dataset可能是因为防止线程访问同一个资源导致冲突
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optimizer.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
        if self.nice:
            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(device)-det_rays_o)/det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

        #if 'grid_fine' in self.c:
            #print('grid_fine is in self.c.')
        #else:
            #print('grid_fine is not in self.c')
        ret = self.renderer.render_batch_ray(
            self.c, self.decoders, batch_rays_d, batch_rays_o,  self.device, stage='color',  gt_depth=batch_gt_depth)
        depth, uncertainty, color = ret

        # 和mapper里一样，这里的思路也是对pixels的采样完全随机，渲染完毕计算loss时加上一个mask，
        # 只不过这里的mask会包括对动态物体的过滤以及对depth=0的点的过滤两部分
        uncertainty = uncertainty.detach()
        if self.handle_dynamic:
            tmp = torch.abs(batch_gt_depth-depth)/torch.sqrt(uncertainty+1e-10)
            mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)
        else:
            mask = batch_gt_depth > 0

        loss = (torch.abs(batch_gt_depth-depth) /
                torch.sqrt(uncertainty+1e-10))[mask].sum()

        if self.use_color_in_tracking:
            color_loss = torch.abs(
                batch_gt_color - color)[mask].sum()
            loss += self.w_color_loss*color_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    def update_para_from_mapping(self, queue):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        # mapper每隔一定帧才优化一次，所以没优化的时候不用更新
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
            # Tracking线程中用到的grid不是self.shared_c而是self.c，self.shared_c来自slam.shared_c，
            # slam.shared_c被放到了显存上，所以是共享的，mapper中的修改tracker中也能实时看到，
            # 那么既然这样为什么tracker中不直接使用self.shared_c，而是每当mapping更新了地图的时候就copy一个副本下来用？
            # for key, val in self.shared_c.items():
            #     val = val.clone().to(self.device)
            #     self.c[key] = val
            received_mapper = False
            received_coarse_mappper = False
            while True:
                value = queue.get()
                if 'grid_middle' in value:
                    for key, val in value.items():
                        tmp = torch.from_numpy(val)
                        self.c[key] = tmp.to(self.device)
                        print("copied " + str(key) + " grid to Tracker.")
                elif 'grid_coarse' in value:
                    tmp = torch.from_numpy(value['grid_coarse'])
                    self.c['grid_coarse'] = tmp.to(self.device)
                    print("copied " + "grid_coarse" + " grid to Tracker.")
                elif value == "Mapper_end":
                    received_mapper = True
                elif value == "Coarse_mapper_end":
                    received_coarse_mappper = True
                if received_mapper and received_coarse_mappper:
                    break
            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self, queue, is_update_bound):
        device = self.device
        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)

        # tracker启动以后遍历所有帧然后结束
        for idx, gt_color, gt_depth, gt_c2w in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx[0]
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_c2w = gt_c2w[0]

            if False:
                print("---------------------------------------------change bound---------------------------------------------")
                values = [[-2.9000, 8.9400],
                        [-3.2000, 5.7600],
                        [-3.5000, 3.5400]]
                new_bound = torch.tensor(values, dtype=torch.float64)
                # new_bound.share_memory_()
                # self.slam.bound = new_bound.clone().cpu()
                self.bound[0][0] = 0
                if self.slam.nice:
                    self.slam.shared_decoders.bound = self.slam.bound
                    self.slam.shared_decoders.middle_decoder.bound = self.slam.bound
                    self.slam.shared_decoders.fine_decoder.bound = self.slam.bound
                    self.slam.shared_decoders.color_decoder.bound = self.slam.bound
                    if self.slam.coarse:
                        self.slam.shared_decoders.coarse_decoder.bound = self.slam.bound*self.slam.coarse_bound_enlarge

                # coarse_padding_tensor = torch.zeros([1, 32, 7, 8, 5]).normal_(mean=0, std=0.01)
                # coarse_padding_tensor = coarse_padding_tensor.to(self.cfg['mapping']['device'])
                # coarse_padding_tensor.share_memory_()
                # coarse_source_tensor = self.slam.shared_c['grid_coarse']
                # self.slam.shared_c['grid_coarse'] = torch.cat([coarse_source_tensor, coarse_padding_tensor], dim=-1)

                middle_padding_tensor = torch.zeros([1, 32, 21, 28, 3]).normal_(mean=0, std=0.01)
                middle_padding_tensor = middle_padding_tensor.to(self.cfg['mapping']['device'])
                middle_padding_tensor.share_memory_()
                middle_source_tensor = self.slam.shared_c['grid_middle']
                self.slam.shared_c['grid_middle'] = torch.cat([middle_source_tensor, middle_padding_tensor], dim=-1)

                fine_padding_tensor = torch.zeros([1, 32, 43, 56, 5]).normal_(mean=0, std=0.01)
                fine_padding_tensor = fine_padding_tensor.to(self.cfg['mapping']['device'])
                fine_padding_tensor.share_memory_()
                fine_source_tensor = self.slam.shared_c['grid_fine']
                self.slam.shared_c['grid_fine'] = torch.cat([fine_source_tensor, fine_padding_tensor], dim=-1)

                color_padding_tensor = torch.zeros([1, 32, 43, 56, 5]).normal_(mean=0, std=0.01)
                color_padding_tensor = color_padding_tensor.to(self.cfg['mapping']['device'])
                color_padding_tensor.share_memory_()
                color_source_tensor = self.slam.shared_c['grid_color']
                self.slam.shared_c['grid_color'] = torch.cat([color_source_tensor, color_padding_tensor], dim=-1)
                
                
                self.slam.renderer.bound = self.slam.bound
                self.slam.mesher.renderer = self.slam.renderer
                self.slam.mesher.bound = self.slam.bound
                self.slam.logger.shared_c = self.slam.shared_c
                self.slam.logger.shared_decoders = self.slam.shared_decoders
                self.slam.mapper.c = self.slam.shared_c
                self.slam.mapper.bound = self.slam.bound
                self.slam.mapper.logger = self.slam.logger
                self.slam.mapper.mesher = self.slam.mesher
                self.slam.mapper.renderer = self.slam.renderer
                self.slam.mapper.decoders = self.slam.shared_decoders

                self.slam.coarse_mapper.c = self.slam.shared_c
                self.slam.coarse_mapper.bound = self.slam.bound
                self.slam.coarse_mapper.logger = self.slam.logger
                self.slam.coarse_mapper.mesher = self.slam.mesher
                self.slam.coarse_mapper.renderer = self.slam.renderer
                self.slam.coarse_mapper.decoders = self.slam.shared_decoders

                self.bound = self.slam.bound
                self.mesher = self.slam.mesher
                self.shared_c = self.slam.shared_c
                self.renderer = self.slam.renderer
                self.shared_decoders = self.slam.shared_decoders
                

            # strict策略下，tracker每优化完every_frame个帧之后等mapper
            # 例如若every_frame是5，tracker跑完2 3 4 5帧以后，到6帧时才等mapper，
            # 因为mapper那边此时2 3 4帧都不做优化，5帧才做优化，
            # 所以tracker在2 3 4帧等mapper是没用的，那边在这几帧不会启动
            if self.sync_method == 'strict':
                # strictly mapping and then tracking
                # initiate mapping every self.every_frame frames
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                    while self.mapping_idx[0] != idx-1:
                        time.sleep(0.1)
                    pre_c2w = self.estimate_c2w_list[idx-1].to(device)
            elif self.sync_method == 'loose':
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                while self.mapping_idx[0] < idx-self.every_frame-self.every_frame//2:
                    time.sleep(0.1)
            elif self.sync_method == 'free':
                # pure parallel, if mesh/vis happens may cause inbalance
                pass

            self.update_para_from_mapping(queue)

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)
            # 第0帧，或者设置了不优化位姿直接用gt位姿时，c2w(估计的位姿)就设置成gt位姿
            if idx == 0 or self.gt_camera:
                c2w = gt_c2w
                if not self.no_vis_on_first_frame:
                    self.visualizer.vis(
                        idx, 0, gt_depth, gt_color, c2w, self.c, self.decoders)
            # 否则进入tracking主优化过程
            else:
                # 从4*4变换矩阵得到四元数
                gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                # 根据匀速运动模型从pre_c2w(上一帧估计的位姿)得到当前帧估计位姿的初始值
                if self.const_speed_assumption and idx-2 >= 0:
                    pre_c2w = pre_c2w.float()
                    delta = pre_c2w@self.estimate_c2w_list[idx-2].to(
                        device).float().inverse()
                    estimated_new_cam_c2w = delta@pre_c2w
                else:
                # 否则直接拿上一帧估计位姿当作当前帧位姿的初始值
                    estimated_new_cam_c2w = pre_c2w

                camera_tensor = get_tensor_from_camera(
                    estimated_new_cam_c2w.detach())
                if self.seperate_LR:
                    camera_tensor = camera_tensor.to(device).detach()
                    T = camera_tensor[-3:]
                    quad = camera_tensor[:4]
                    cam_para_list_quad = [quad]
                    quad = Variable(quad, requires_grad=True)
                    T = Variable(T, requires_grad=True)
                    camera_tensor = torch.cat([quad, T], 0)
                    cam_para_list_T = [T]
                    cam_para_list_quad = [quad]
                    optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr},
                                                         {'params': cam_para_list_quad, 'lr': self.cam_lr*0.2}])
                else:
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    cam_para_list = [camera_tensor]
                    optimizer_camera = torch.optim.Adam(
                        cam_para_list, lr=self.cam_lr)

                initial_loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device)-camera_tensor).mean().item()
                candidate_cam_tensor = None 
                current_min_loss = 10000000000.
                # 优化num_cam_iters次，挑出来loss最小的位姿当作当前帧估计位姿
                for cam_iter in range(self.num_cam_iters):
                    if self.seperate_LR:
                        camera_tensor = torch.cat([quad, T], 0).to(self.device)

                    self.visualizer.vis(
                        idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)

                    loss = self.optimize_cam_in_batch(
                        camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera)

                    if cam_iter == 0:
                        initial_loss = loss

                    loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(device)-camera_tensor).mean().item()
                    if self.verbose:
                        if cam_iter == self.num_cam_iters-1:
                            print(
                                f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                                f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_tensor = camera_tensor.clone().detach()
                # loss最小的位姿经过变换，得到4*4变换矩阵c2w，即当前帧位姿估计结果
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                    [1, 4])).type(torch.float32).to(self.device)
                c2w = get_camera_from_tensor(
                    candidate_cam_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)
            self.estimate_c2w_list[idx] = c2w.clone().cpu()
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            pre_c2w = c2w.clone()
            # NICE_SLAM.py中的idx更新完全由tracker进行
            self.idx[0] = idx
            if self.low_gpu_mem:
                torch.cuda.empty_cache()
        print("Tracker: End")
