import os
import torch
from gaussian_renderer import render
from scene import GaussianModel
from arguments import OptimizationParams
from utils.loss_utils import l1_loss, ssim
from utils.confidence_utils import projection_in_image, compute_confidence_wrapper, compute_confidence_sample_wrapper
from utils.debug_utils import save_deleted_gaussians_confidence, \
    save_added_gaussians, save_confidence, save_viewspace_gradient, \
   save_opacity, save_deleted_gaussians, save_densified_points
from confidence_filter import heap_sort


class DistributionControl:
    def __init__(self, gaussians: GaussianModel, all_cameras: dict, opt: OptimizationParams, pipe, dataset):
        self.gaussians = gaussians
        self.all_cameras = all_cameras
        self.opt = opt
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.bg = torch.rand(3, device="cuda") if opt.random_background else background
        self.pipe = pipe
        self.show_debug_info = False
        self.model_path = dataset.model_path
        self.last_add_gaussians = None
        self.viewspace_gradients = torch.zeros((gaussians.size, 2), dtype=torch.float, device="cuda")
        self.debug_point_cloud_path = None
        self.random_drop_flag = False
        self.iteration = None

    def accumulate_view_space_gradients(self, gt_img, render_img, viewspace_points, visibility):
        with torch.no_grad():
            self.gaussians.optimizer.zero_grad(set_to_none=True)
        Ll1 = l1_loss(render_img, gt_img)
        ssim_loss = ssim(render_img, gt_img)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim_loss)
        loss.backward()
        with torch.no_grad():
            grad = viewspace_points.grad[visibility, 2].clone()  # [N,]
            self.gaussians.optimizer.zero_grad(set_to_none=True)
        return grad

    def render_and_accumulate(
            self,
            get_vis_pair=True,
            get_view_space_gradient=True
    ):
        cur_gaussians_size = self.gaussians.size
        contribution_weight = None
        contribution_heap = None
        index_heap = None
        view_space_gradient = None
        view_space_gradient_weight = None
        if get_vis_pair:
            contribution_heap = torch.zeros((cur_gaussians_size, 2), dtype=torch.float32, device="cuda")
            index_heap = torch.zeros((cur_gaussians_size, 2), dtype=torch.int32, device="cuda") - 1
        if get_view_space_gradient:
            view_space_gradient = torch.zeros((cur_gaussians_size,), dtype=torch.float32, device="cuda")
            view_space_gradient_weight = torch.zeros((cur_gaussians_size,), dtype=torch.float32, device="cuda")

        for cam in self.all_cameras.values():
            render_pkg = render(cam, self.gaussians, self.pipe, self.bg, contrib_weight=True)
            # parse render package
            render_visibility = render_pkg["visibility_filter"]
            contrib_weight_cam = render_pkg["contrib_weight"]
            with torch.no_grad():
                if get_vis_pair:
                    # Accumulate contributions matrix
                    on_image_mask = projection_in_image(
                        self.gaussians.get_xyz, cam.original_intrinsics, cam.original_extrinsics,
                        cam.image_height, cam.image_width
                    )
                    contrib_mask = contrib_weight_cam > 0.0
                    vis_mask = render_visibility & on_image_mask & contrib_mask
                    contribution_heap[vis_mask], index_heap[vis_mask] = (
                        heap_sort(
                            contribution_heap[vis_mask],
                            index_heap[vis_mask],
                            contrib_weight_cam[vis_mask], cam.uid
                        )
                    )
                    del on_image_mask, contrib_mask, vis_mask
            if get_view_space_gradient:
                weight = contrib_weight_cam
                view_space_gradient[render_visibility] += (self.accumulate_view_space_gradients(
                    cam.original_image, render_pkg["render"], render_pkg["viewspace_points"], render_visibility
                ) * weight[render_visibility])
                view_space_gradient_weight[render_visibility] += weight[render_visibility]
                del weight
            del render_pkg, render_visibility, contrib_weight_cam

        if get_view_space_gradient:
            view_space_gradient = view_space_gradient.detach()
            view_space_gradient_weight = view_space_gradient_weight.detach()
            view_space_gradient = view_space_gradient / view_space_gradient_weight
            view_space_gradient[view_space_gradient.isnan()] = 0.0

        return {
            "contribution_weight": contribution_weight,
            "vis_pair": torch.flip(index_heap, dims=[1]) if get_vis_pair else None,
            "viewspace_gradient": view_space_gradient
        }

    def compute_dynamic_gradient_threshold(self, grads, grad_ratio=0.75):
        grad_mean = torch.mean(grads).item()
        grad_min = grads.min().item()
        grad_max = 3 * grad_mean
        grad_hist = torch.histc(grads, bins=1000, min=grad_min, max=grad_max)
        cdf = torch.cumsum(grad_hist.float(), dim=0)
        total_count = torch.sum(grad_hist)
        target_count = total_count * grad_ratio
        quantile_index = torch.nonzero(cdf >= target_count, as_tuple=False)[0]
        grad_threshold = grad_min + (quantile_index / 1000) * (grad_max - grad_min)

        self.random_drop_flag = False
        if grad_threshold < self.opt.densify_grad_threshold:
            grad_threshold = self.opt.densify_grad_threshold
            self.random_drop_flag = True

        return grad_threshold

    def densification(self, viewspace_gradient):
        # Compute dynamic gradient threshold
        grad_threshold = self.compute_dynamic_gradient_threshold(viewspace_gradient)
        densification_mask = viewspace_gradient >= grad_threshold
        generated_gaussians = self.gaussians.densify_rand_organized(densification_mask)
        if self.show_debug_info:
            save_densified_points(
                self.debug_point_cloud_path,
                self.gaussians.get_xyz[densification_mask],
                generated_gaussians["xyz"].reshape(self.gaussians.get_xyz[densification_mask].shape[0], 2, 3)
            )

        return generated_gaussians

    @torch.no_grad()
    def compute_confidence_with_vis_pair(self, vis_pair, xyz, cov=None):
        confidence_values = torch.zeros(xyz.shape[0], dtype=torch.float32, device="cuda")

        unique_pairs = torch.unique(vis_pair, dim=0)
        unique_pairs = unique_pairs.cpu().numpy().tolist()
        unique_pairs_dict = {}
        for pair in unique_pairs:
            unique_pairs_dict[tuple(pair)] = []
        vis_pair_list = vis_pair.cpu().numpy().tolist()
        for idx, pair in enumerate(vis_pair_list):
            unique_pairs_dict[tuple(pair)].append(idx)

        for pair, gaussians_id in unique_pairs_dict.items():
            if pair[0] == -1 or pair[1] == -1:
                continue
            cam1 = self.all_cameras[pair[0]]
            cam2 = self.all_cameras[pair[1]]
            selected_xyz = xyz[gaussians_id]  # [N'', 3]
            if cov is not None:
                confidence = compute_confidence_sample_wrapper(selected_xyz, cov, cam1, cam2)
            else:
                raise NotImplementedError("Confidence computation without covariance is not implemented")
            confidence_values[gaussians_id] = confidence
        return confidence_values

    def random_drop(self, drop_ratio):
        drop_mask = torch.rand(self.gaussians.size, device="cuda") < drop_ratio
        self.gaussians.prune_points(drop_mask)
        return drop_mask

    def control(self, iteration):
        debug_point_cloud_path = os.path.join(
            self.model_path,
            "point_cloud/iteration_{}_debug_point_cloud".format(iteration)
        )
        if self.show_debug_info:
            self.debug_point_cloud_path = debug_point_cloud_path
        self.iteration = iteration
        if iteration < self.opt.densify_until_iter:
            if iteration > self.opt.densify_from_iter and iteration % self.opt.control_interval == 0:
                # 1. get view space gradient, contribution weight, contribution matrix
                render_and_accumulate_pkg = self.render_and_accumulate(
                    get_vis_pair=False if iteration % 1000 == 0 else True
                )
                with torch.no_grad():
                    vis_pair = render_and_accumulate_pkg["vis_pair"]
                    viewspace_gradient = render_and_accumulate_pkg["viewspace_gradient"]

                    # 2. compute confidence with vis pair
                    if iteration % 1000 == 0:
                        prune_mask = torch.zeros(self.gaussians.size, dtype=torch.bool, device="cuda")
                    else:
                        confidence = self.compute_confidence_with_vis_pair(
                            vis_pair, self.gaussians.get_xyz, self.gaussians.get_covariance()
                        )
                        confidence_mask = confidence < self.opt.confidence_threshold
                        vis_mask = vis_pair == -1
                        one_view = torch.sum(vis_mask, dim=1) == 1
                        confidence_mask[one_view] = False
                        if self.show_debug_info:
                            save_confidence(
                                debug_point_cloud_path,
                                self.gaussians.get_xyz,
                                confidence,
                                vis_pair
                            )
                            save_opacity(
                                debug_point_cloud_path,
                                self.gaussians.get_xyz,
                                self.gaussians.get_opacity
                            )
                            save_deleted_gaussians_confidence(
                                debug_point_cloud_path,
                                self.gaussians.get_xyz,
                                confidence_mask
                            )
                        prune_mask = confidence_mask

                    if self.last_add_gaussians is not None:
                        prune_mask[self.last_add_gaussians] = False
                    if self.show_debug_info:
                        save_deleted_gaussians(
                            debug_point_cloud_path,
                            self.gaussians.get_xyz,
                            prune_mask
                        )
                    # 3. invalid Gaussians
                    invalid_mask = torch.isnan(self.gaussians.get_xyz) | torch.isinf(self.gaussians.get_xyz)  # [N, 3]
                    invalid_mask = torch.sum(invalid_mask, dim=1) > 0

                    prune_mask = prune_mask | invalid_mask
                    self.gaussians.prune_points(prune_mask)

                    # 4. densification
                    viewspace_gradient = viewspace_gradient[~prune_mask]
                    if self.show_debug_info:
                        save_viewspace_gradient(
                            debug_point_cloud_path,
                            self.gaussians.get_xyz,
                            viewspace_gradient
                        )
                    generated_gaussians = self.densification(viewspace_gradient)

                    # 5. add the generated gaussians
                    last_size = self.gaussians.size
                    self.gaussians.densify_points(generated_gaussians)
                    self.last_add_gaussians = torch.zeros(self.gaussians.size, dtype=torch.bool, device="cuda")
                    self.last_add_gaussians[last_size:] = True
                    if self.show_debug_info:
                        save_added_gaussians(
                            debug_point_cloud_path,
                            self.gaussians.get_xyz,
                            self.last_add_gaussians
                        )

                    # 6. according to the dynamic threshold, drop some gaussians
                    if self.random_drop_flag:
                        drop_mask = self.random_drop(0.05)
                        self.last_add_gaussians = self.last_add_gaussians[~drop_mask]

                    # 7. reset opacity and radius
                    if iteration % self.opt.opacity_reset_interval == 0:
                        self.gaussians.reset_opacity()
                    self.gaussians.reset_R()
                    torch.cuda.empty_cache()
