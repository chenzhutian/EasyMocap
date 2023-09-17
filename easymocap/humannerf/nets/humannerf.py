import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.network_util import MotionBasisComputer
from .mweight_vol_decoders import MotionWeightVolumeDecoder
from .pose_decoders import BodyPoseRefiner
from .non_rigid_motion_mlps import NonRigidMotionMLP
from .canonical_mlps import CanonicalMLP
from .embedders.hannw_fourier import get_embedder as get_non_rigid_embedder
from .embedders.fourier import get_embedder

# from configs import cfg

total_bones = 24
N_samples = 128      # number of samples for each ray in coarse ray matching

class Network(nn.Module):
    def __init__(self,
        mweight_volume, # args
        non_rigid_motion_mlp, # args
        canonical_mlp, # args
        pose_decoder, # args
        nerf, # args
    ):
        super(Network, self).__init__()

        print("------------------ GPU Configurations ------------------")
        self.gpu_args = {}
        n_gpus = torch.cuda.device_count()
        self.gpu_args['n_gpus'] = n_gpus
        if n_gpus > 0:
            all_gpus = list(range(n_gpus))
            self.gpu_args['primary_gpus'] = [0]
            if n_gpus > 1:
                self.gpu_args['secondary_gpus'] = [g for g in all_gpus \
                                        if g not in self.gpu_args['primary_gpus']]
            else:
                self.gpu_args['secondary_gpus'] = self.gpu_args['primary_gpus']

        print(f"Primary GPUs: {self.gpu_args['primary_gpus']}")
        print(f"Secondary GPUs: {self.gpu_args['secondary_gpus']}")
        print("--------------------------------------------------------")

        self.nerf_args = nerf

        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(total_bones=total_bones)

        # motion weight volume
        self.mweight_vol_decoder = MotionWeightVolumeDecoder(
            embedding_size=mweight_volume['embedding_size'],
            volume_size=mweight_volume['volume_size'],
            total_bones=total_bones
        )

        # non-rigid motion st positional encoding
        self.get_non_rigid_embedder = get_non_rigid_embedder

        # non-rigid motion MLP
        _, non_rigid_pos_embed_size = \
            self.get_non_rigid_embedder(non_rigid_motion_mlp['multires'], 
                                        non_rigid_motion_mlp['i_embed'],
                                        non_rigid_motion_mlp['kick_in_iter'],
                                        non_rigid_motion_mlp['full_band_iter']
                                        )

        self.non_rigid_mlp = NonRigidMotionMLP(
                pos_embed_size=non_rigid_pos_embed_size,
                condition_code_size=non_rigid_motion_mlp['condition_code_size'],
                mlp_width=non_rigid_motion_mlp['mlp_width'],
                mlp_depth=non_rigid_motion_mlp['mlp_depth'],
                skips=non_rigid_motion_mlp['skips'])

        self.non_rigid_motion_mlp_args = non_rigid_motion_mlp

        self.non_rigid_mlp = \
            nn.DataParallel(
                self.non_rigid_mlp,
                device_ids=self.gpu_args['secondary_gpus'],
                output_device=self.gpu_args['secondary_gpus'][0])

        # canonical positional encoding
        cnl_pos_embed_fn, cnl_pos_embed_size = \
            get_embedder(canonical_mlp['multires'], 
                         canonical_mlp['i_embed'])
        self.pos_embed_fn = cnl_pos_embed_fn

        # canonical mlp 
        skips = [4]
        self.cnl_mlp = CanonicalMLP(
                input_ch=cnl_pos_embed_size, 
                mlp_depth=canonical_mlp['mlp_depth'], 
                mlp_width=canonical_mlp['mlp_width'],
                skips=skips)

        self.cnl_mlp = \
            nn.DataParallel(
                self.cnl_mlp,
                device_ids=self.gpu_args['secondary_gpus'],
                output_device=self.gpu_args['primary_gpus'][0])

        # pose decoder MLP
        self.pose_decoder = BodyPoseRefiner(
                total_bones,
                embedding_size=pose_decoder['embedding_size'],
                mlp_width=pose_decoder['mlp_width'],
                mlp_depth=pose_decoder['mlp_depth'])
        self.pose_decoder_args = pose_decoder

    def deploy_mlps_to_secondary_gpus(self):
        self.cnl_mlp = self.cnl_mlp.to(self.gpu_args['secondary_gpus'][0])
        if self.non_rigid_mlp:
            self.non_rigid_mlp = self.non_rigid_mlp.to(self.gpu_args['secondary_gpus'][0])

        return self


    def _query_mlp(
            self,
            pos_xyz,
            pos_embed_fn, 
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input):

        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        chunk = self.nerf_args['netchunk_per_gpu']*len(self.gpu_args['secondary_gpus'])

        result = self._apply_mlp_kernals(
                        pos_flat=pos_flat,
                        pos_embed_fn=pos_embed_fn,
                        non_rigid_mlp_input=non_rigid_mlp_input,
                        non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                        chunk=chunk)

        output = {}

        raws_flat = result['raws']
        output['raws'] = torch.reshape(
                            raws_flat, 
                            list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]])

        return output


    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))


    def _apply_mlp_kernals(
            self, 
            pos_flat,
            pos_embed_fn,
            non_rigid_mlp_input,
            non_rigid_pos_embed_fn,
            chunk):
        raws = []

        # iterate ray samples by trunks
        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            total_elem = end - start

            xyz = pos_flat[start:end]

            if not cfg.ignore_non_rigid_motions:
                non_rigid_embed_xyz = non_rigid_pos_embed_fn(xyz)
                result = self.non_rigid_mlp(
                    pos_embed=non_rigid_embed_xyz,
                    pos_xyz=xyz,
                    condition_code=self._expand_input(non_rigid_mlp_input, total_elem)
                )
                xyz = result['xyz']

            xyz_embedded = pos_embed_fn(xyz)
            raws += [self.cnl_mlp(
                        pos_embed=xyz_embedded)]

        output = {}
        output['raws'] = torch.cat(raws, dim=0).to(self.gpu_args['primary_gpus'][0])

        return output


    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        chunk = self.nerf_args['chunk']
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self._render_rays(rays_flat[i:i+chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret


    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, bgcolor=None):
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        alpha = _raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
        alpha = alpha * raw_mask[:, :, 0]

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)

        rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.

        return rgb_map, acc_map, weights, depth_map


    @staticmethod
    def _sample_motion_fields(
            pts,
            motion_scale_Rs, 
            motion_Ts, 
            motion_weights_vol,
            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
            output_list):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3) # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1] 

        weights_list = []
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            pos = (pos - cnl_bbox_min_xyz[None, :]) \
                            * cnl_bbox_scale_xyz[None, :] - 1.0 
            weights = F.grid_sample(input=motion_weights[None, i:i+1, :, :, :], 
                                    grid=pos[None, None, None, :, :],           
                                    padding_mode='zeros', align_corners=True)
            weights = weights[0, 0, 0, 0, :, None] 
            weights_list.append(weights) 
        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        total_bases = backwarp_motion_weights.shape[-1]

        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, 
                                                dim=-1, keepdim=True)
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
                        torch.stack(weighted_motion_fields, dim=0), dim=0
                        ) / backwarp_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask = backwarp_motion_weights_sum

        x_skel = x_skel.reshape(orig_shape[:2]+[3])
        backwarp_motion_weights = \
            backwarp_motion_weights.reshape(orig_shape[:2]+[total_bases])
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2]+[1])

        results = {}
        
        if 'x_skel' in output_list: # [N_rays x N_samples, 3]
            results['x_skel'] = x_skel
        if 'fg_likelihood_mask' in output_list: # [N_rays x N_samples, 1]
            results['fg_likelihood_mask'] = fg_likelihood_mask
        
        return results


    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        return rays_o, rays_d, near, far


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals


    def _render_rays(
            self, 
            ray_batch, 
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz,
            pos_embed_fn,
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input=None,
            bgcolor=None,
            **_):
        
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far)
        if self.nerf_args['perturb'] > 0.:
            z_vals = self._stratified_sampling(z_vals)

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        
        mv_output = self._sample_motion_fields(
                            pts=pts,
                            motion_scale_Rs=motion_scale_Rs[0], 
                            motion_Ts=motion_Ts[0], 
                            motion_weights_vol=motion_weights_vol,
                            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                            output_list=['x_skel', 'fg_likelihood_mask'])
        pts_mask = mv_output['fg_likelihood_mask']
        cnl_pts = mv_output['x_skel']

        query_result = self._query_mlp(
                                pos_xyz=cnl_pts,
                                non_rigid_mlp_input=non_rigid_mlp_input,
                                pos_embed_fn=pos_embed_fn,
                                non_rigid_pos_embed_fn=non_rigid_pos_embed_fn)
        raw = query_result['raws']
        
        rgb_map, acc_map, _, depth_map = \
            self._raw2outputs(raw, pts_mask, z_vals, rays_d, bgcolor)

        return {'rgb' : rgb_map,  
                'alpha' : acc_map, 
                'depth': depth_map}


    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(
                                        dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts


    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        tmp_total_bones = total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, tmp_total_bones, 3, 3)

    
    def forward(self,
                rays, 
                dst_Rs, dst_Ts, cnl_gtfms,
                motion_weights_priors,
                dst_posevec=None,
                near=None, far=None,
                iter_val=1e7,
                **kwargs):

        dst_Rs=dst_Rs[None, ...]
        dst_Ts=dst_Ts[None, ...]
        dst_posevec=dst_posevec[None, ...]
        cnl_gtfms=cnl_gtfms[None, ...]
        motion_weights_priors=motion_weights_priors[None, ...]

        # correct body pose
        if iter_val >= self.pose_decoder_args.get('kick_in_iter', 0):
            pose_out = self.pose_decoder(dst_posevec)
            refined_Rs = pose_out['Rs']
            refined_Ts = pose_out.get('Ts', None)
            
            dst_Rs_no_root = dst_Rs[:, 1:, ...]
            dst_Rs_no_root = self._multiply_corrected_Rs(
                                        dst_Rs_no_root, 
                                        refined_Rs)
            dst_Rs = torch.cat(
                [dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)

            if refined_Ts is not None:
                dst_Ts = dst_Ts + refined_Ts

        non_rigid_pos_embed_fn, _ = \
            self.get_non_rigid_embedder(
                multires=self.non_rigid_motion_mlp_args['multires'],                         
                is_identity=self.non_rigid_motion_mlp_args['i_embed'],
                iter_val=iter_val,)

        if iter_val < self.non_rigid_motion_mlp_args['kick_in_iter']:
            # mask-out non_rigid_mlp_input 
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec

        kwargs.update({
            "pos_embed_fn": self.pos_embed_fn,
            "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            "non_rigid_mlp_input": non_rigid_mlp_input
        })

        motion_scale_Rs, motion_Ts = self._get_motion_base(
                                            dst_Rs=dst_Rs, 
                                            dst_Ts=dst_Ts, 
                                            cnl_gtfms=cnl_gtfms)
        motion_weights_vol = self.mweight_vol_decoder(
            motion_weights_priors=motion_weights_priors)
        motion_weights_vol=motion_weights_vol[0] # remove batch dimension

        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
            'motion_weights_vol': motion_weights_vol
        })

        rays_o, rays_d = rays
        rays_shape = rays_d.shape 

        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos, **kwargs)
        for k in all_ret:
            k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_shape)

        return all_ret
