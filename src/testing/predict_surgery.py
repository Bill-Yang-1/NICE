from __future__ import division
import torch
import torch.optim as optim
import os
import numpy as np
import trimesh
from tqdm import tqdm
from src.models.loss_functions import  compute_skull_loss_corresp_forward
from src import env_paths
from src.reconstruction.reconstruction import deform_mesh, get_logits_all,create_grid_points_from_bounds, mesh_from_logits



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SurgeryPredictAutoDecoder(object):
    def __init__(self,
                 decoder,
                 decoder_shape,
                 cfg,
                 device,
                 val_dataset,
                 output_fd,
                 ckpt=None):

        
        self.output_fd = output_fd
        os.makedirs(self.output_fd, exist_ok=True)
        self.decoder = decoder.to(device)
        if decoder_shape is not None:
            self.decoder_shape = decoder_shape.to(device)
        else:
            self.decoder_shape = None
        self.ckpt = ckpt
        self.latent_codes_shape_val = torch.nn.Embedding(len(val_dataset)*2, decoder_shape.lat_dim,
                                                   max_norm=1.0, sparse=True, device=device).float()
        
        self.latent_codes_shape_val.requires_grad = False
        self.latent_codes_val = torch.nn.Embedding(len(val_dataset), self.decoder.lat_dim_surgery,
                                                   max_norm=1.0, sparse=True, device=device).float()


        self.latent_codes_val_best = torch.nn.Embedding(len(val_dataset), self.decoder.lat_dim_surgery,
                                                   max_norm=1.0, sparse=True, device=device).float()
        
        # load shape ckpt
        ckpt_path = env_paths.CKPT_FULL_HEAD
        self.load_shape_ckpt(ckpt_path)
        
        surgery_ckpt_path = env_paths.CKPT_SURGERY
        self.load_surgery_ckpt(surgery_ckpt_path)
        torch.nn.init.normal_(
            self.latent_codes_val.weight.data,
            0.0,
            0.01
        )
        print('Number of Parameters in decoder: {}'.format(count_parameters(self.decoder)))
        self.cfg = cfg['training']
        self.device = device
        self.optimizer_encoder = optim.AdamW(params=list(decoder.parameters()),
                                             lr=self.cfg['lr'],
                                             weight_decay=self.cfg['weight_decay'])
        self.optimizer_lat_val = optim.SparseAdam(list(self.latent_codes_val.parameters()), lr=self.cfg['lr_lat'])
        self.lr = self.cfg['lr']
        self.lr_lat = self.cfg['lr_lat']
        self.val_dataset = val_dataset
        self.val_min = None
        self.val_data_loader = self.val_dataset.get_loader()
        self.min = [-0.5, -0.5, -0.5]
        self.max = [0.5, 0.5, 0.5] 
        self.res = 256
        self.grid_points = create_grid_points_from_bounds(self.min, self.max, self.res)
        self.grid_points = torch.from_numpy(self.grid_points).to(self.device, dtype=torch.float)
        self.grid_points = torch.reshape(self.grid_points, (1, len(self.grid_points), 3)).to(self.device)
        self.past_eval_steps = 0

        print('Done init trainer')

    def init_shape_state(self, ckpt, path):
        path = path + 'checkpoint_epoch_{}.tar'.format(ckpt)
        checkpoint = torch.load(path)
        self.decoder_shape.load_state_dict(checkpoint['decoder_state_dict'])
        self.latent_codes_shape.load_state_dict(checkpoint['latent_codes_state_dict'])
        print('Train shape space loaded with dims: ')
        print(self.latent_codes_shape.weight.shape)
        self.latent_codes_shape_val.load_state_dict(checkpoint['latent_codes_val_state_dict'])
        print('Loaded checkpoint from: {}'.format(path))

      
    def load_shape_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.decoder_shape.load_state_dict(checkpoint['decoder_state_dict'])
        print('Loaded checkpoint from: {}'.format(ckpt_path))
        
        
    def load_surgery_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print('Loaded checkpoint from: {}'.format(ckpt_path))


    def reduce_lr(self, epoch):
        if epoch > 0 and self.cfg['lr_decay_interval'] is not None and epoch % self.cfg['lr_decay_interval'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval'])
            lr = self.cfg['lr'] * self.cfg['lr_decay_factor']**decay_steps
            print('Reducting LR to {}'.format(lr))
            for param_group in self.optimizer_encoder.param_groups:
                param_group["lr"] = lr

        if epoch > 0 and self.cfg['lr_decay_interval_lat'] is not None and epoch % self.cfg['lr_decay_interval_lat'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval_lat'])
            lr = self.cfg['lr_lat'] * self.cfg['lr_decay_factor_lat']**decay_steps
            print('Reducting LR for latent codes to {}'.format(lr))
            for param_group in self.optimizer_lat_val.param_groups:
                param_group["lr"] = lr



    def test_model(self, epochs):
        print('Starting to test model.')
        pbar = tqdm(range(epochs), desc="Fitting Latent Codes")
        for epoch in pbar:
            self.reduce_lr(epoch)
            val_loss_dict = self.compute_skull_val_loss(epoch)
            current_lat_val_lr = self.optimizer_lat_val.param_groups[0]['lr']
            mxl_loss = val_loss_dict['mxl_loss_corresp']
            mdb_loss = val_loss_dict['mdb_loss_corresp']
            
            if self.val_min is None:
                self.val_min = val_loss_dict['loss']
            if val_loss_dict['loss'] < self.val_min:
                self.val_min = val_loss_dict['loss']
                self.latent_codes_val_best = self.latent_codes_val
            
            pbar.set_description(f'Epoch: {epoch}|lr = {current_lat_val_lr:.2e}|mxl = {mxl_loss:.4e}|mdb = {mdb_loss:.4e}')
        self.fit_val_dataset(epoch)
        

    def compute_skull_val_loss(self, epoch):
        self.decoder.eval()

        sum_val_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
        sum_val_loss_dict.update({'loss': 0.0})

        c = 0
        for val_batch in self.val_data_loader:

            self.optimizer_lat_val.zero_grad()
            l_dict = compute_skull_loss_corresp_forward(val_batch, self.decoder, self.decoder_shape, self.latent_codes_val,
                                  self.latent_codes_shape_val, self.device, None)

            for k in l_dict.keys():
                sum_val_loss_dict[k] += l_dict[k].item()
            val_loss = 0.0
            for key in l_dict.keys():
                val_loss += self.cfg['lambdas'][key] * l_dict[key]
            val_loss.backward()
            self.optimizer_lat_val.step()

            sum_val_loss_dict['loss'] += val_loss.item()
            c = c + 1

        for k in sum_val_loss_dict.keys():
            sum_val_loss_dict[k] /= c
        return sum_val_loss_dict



    def construct_rec(self, encoding_expr, encoding_shape, mesh, epoch=-1, landmarks=None,mode='sft'):

        if mesh is None:
            # reconstruct neutral geometry from implicit repr.
            encoding_shape = encoding_shape.unsqueeze(0)
            encoding_expr = encoding_expr.unsqueeze(0)
            logits = get_logits_all(decoder=self.decoder_shape,
                                encoding=encoding_shape,
                                grid_points=self.grid_points.clone(),
                                nbatch_points=25000,
                                mode = mode,
                                )
            mesh = mesh_from_logits(logits, self.min, self.max, self.res)

        deformed_mesh = deform_mesh(mesh, self.decoder, encoding_expr, landmarks, lat_rep_shape=encoding_shape,mode=mode)

        return mesh, deformed_mesh


    def fit_val_dataset(self, epoch):
        self.decoder.eval()
        exp_dir = os.path.join(self.output_fd, 'val_recon')

        os.makedirs(exp_dir, exist_ok=True)
        
        d_set = self.val_dataset
        lat_codes = self.latent_codes_val
        lat_codes_shape = self.latent_codes_shape_val
        
        for idx in range(len(self.val_dataset)):
            iden_val = self.val_dataset.subjects_id[idx]

            encoding_expr = lat_codes(
                    torch.from_numpy(np.array([[idx]])).to(self.device)).squeeze().unsqueeze(0)


            
            encoding_shape = torch.from_numpy(self.val_dataset.get_shape_latent_code(idx)).to(self.device).unsqueeze(0)
            
            
            print(f'Reconstruction idx = {idx} | patient id = {iden_val}')
            

            val_dataset_path = self.val_dataset.datapath
            for m in ['sft', 'mxl', 'mdb']:
                m_gt_pre = trimesh.load(os.path.join(val_dataset_path, iden_val, f'ori_{m}_pre.obj'), process=False)
                
                m_gt_post = trimesh.load(os.path.join(val_dataset_path, iden_val, f'ori_{m}_post.obj'), process=False)
               
                if self.decoder_shape.sft_mlp_pos is not None:
                    gt_landmarks = self.decoder_shape.sft_mlp_pos(encoding_shape[..., :self.decoder_shape.lat_dim_glob]).view(
                        encoding_expr.shape[0], -1, 3)
                    gt_landmarks += self.decoder.sft_landmarks.squeeze(0)
                    gt_landmarks_sft = self.decoder_shape.sft_mlp_pos(encoding_shape[..., :self.decoder_shape.lat_dim_glob]).view(
                        encoding_expr.shape[0], -1, 3)
                    gt_landmarks_sft += self.decoder.sft_landmarks.squeeze(0)
                    gt_landmarks_mxl = self.decoder_shape.mxl_mlp_pos(encoding_shape[..., :self.decoder_shape.lat_dim_glob]).view(
                        encoding_expr.shape[0], -1, 3)
                    gt_landmarks_mxl += self.decoder.mxl_landmarks.squeeze(0)
                    gt_landmarks_mdb = self.decoder_shape.mdb_mlp_pos(encoding_shape[..., :self.decoder_shape.lat_dim_glob]).view(
                        encoding_expr.shape[0], -1, 3)
                    gt_landmarks_mdb += self.decoder.mdb_landmarks.squeeze(0)
                    gt_landmarks = torch.cat([gt_landmarks_sft, gt_landmarks_mxl, gt_landmarks_mdb], dim=1)
                
                
                # do Marching cubes and deform results
                trim, trim_deformed = self.construct_rec(encoding_expr,
                                                            encoding_shape,
                                                            None,
                                                            epoch=epoch,
                                                            landmarks=gt_landmarks,
                                                            mode=m)
                
                trim_reg, trim_deformed_reg = self.construct_rec(encoding_expr,
                                                                    encoding_shape,
                                                                    m_gt_pre,
                                                                    epoch=epoch,
                                                                    landmarks=gt_landmarks,
                                                                    mode=m)
                
                trim.export(exp_dir + f'/{iden_val}_{m}_gt_pre.ply')
                m_gt_post.export(exp_dir + f'/{iden_val}_{m}_gt_post.ply')
                trim_deformed.export(exp_dir + f'/{iden_val}_{m}_pred_post_marching_cube.ply')

                trim_reg.export(exp_dir + f'/{iden_val}_{m}_gt_pre_reg.ply')
                trim_deformed_reg.export(exp_dir + f'/{iden_val}_{m}_pred_post_reg.ply')


