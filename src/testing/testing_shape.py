from __future__ import division
import torch
import torch.optim as optim
import os
import numpy as np
import math
from tqdm import tqdm
from src.models.loss_functions import  compute_all_loss
from src.reconstruction.reconstruction import create_grid_points_from_bounds, mesh_from_logits,get_logits, get_logits_all
from src import env_paths

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TesterAutoDecoder(object):

    def __init__(self, decoder, cfg, device, val_dataset, output_fd):
        
        # load checkpoint for decoder
        self.ckpt_path = env_paths.CKPT_FULL_HEAD
        checkpoint = torch.load(self.ckpt_path, map_location=device)#, weights_only=True)
        decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=True)
        
        
        self.output_fd = output_fd
        self.decoder = decoder

        self.latent_codes_val = torch.nn.Embedding(len(val_dataset), decoder.lat_dim,
                                               max_norm=1.0, sparse=True, device=device).float()

        torch.nn.init.normal_(
            self.latent_codes_val.weight.data,
            0.0,
            0.1 / math.sqrt(decoder.lat_dim),
        )

        print('Number of Parameters in decoder: {}'.format(count_parameters(self.decoder)))
        self.cfg = cfg['training']
        self.device = device
        self.decoder = self.decoder.to(self.device)
        self.optimizer_lat_val = optim.SparseAdam(list(self.latent_codes_val.parameters()), lr=self.cfg['lr_lat'])
        self.lr = self.cfg['lr']
        self.lr_lat = self.cfg['lr_lat']
        self.current_lr_lat = self.lr_lat

        self.val_dataset = val_dataset
        self.val_min = None
        self.val_data_loader = self.val_dataset.get_loader()

        config = self.log_dict(cfg)

        print('Big Box')
        self.min = [-0.5, -0.5, -0.5]
        self.max = [0.5, 0.5, 0.5]
        self.res = 256
        self.grid_points = create_grid_points_from_bounds(self.min, self.max, self.res)
        self.grid_points = torch.from_numpy(self.grid_points).to(self.device, dtype=torch.float)
        self.grid_points = torch.reshape(self.grid_points, (1, len(self.grid_points), 3)).to(self.device)
        self.log_steps = 0
        
        
        self.sft_lmks = self.decoder.sft_landmarks.squeeze().cpu().numpy()
        print(self.sft_lmks.shape)
        self.sft_min_box = self.sft_lmks.min(axis=0) - 0.05
        self.sft_max_box = self.sft_lmks.max(axis=0) + 0.05
        print(self.sft_min_box)
        print(self.sft_max_box)
        
        self.sft_grid_points = create_grid_points_from_bounds(self.sft_min_box, self.sft_max_box, self.res)
        self.sft_grid_points = torch.from_numpy(self.sft_grid_points).to(self.device, dtype=torch.float)
        self.sft_grid_points = torch.reshape(self.sft_grid_points, (1, len(self.sft_grid_points), 3)).to(self.device)
        
        
        
        self.mxl_lmks = self.decoder.mxl_landmarks.squeeze().cpu().numpy()
        print(self.mxl_lmks.shape)
        self.mxl_min_box = self.mxl_lmks.min(axis=0) - 0.05
        self.mxl_max_box = self.mxl_lmks.max(axis=0) + 0.05
        print(self.mxl_min_box)
        print(self.mxl_max_box)
        
        self.mxl_grid_points = create_grid_points_from_bounds(self.mxl_min_box, self.mxl_max_box, self.res)
        self.mxl_grid_points = torch.from_numpy(self.mxl_grid_points).to(self.device, dtype=torch.float)
        self.mxl_grid_points = torch.reshape(self.mxl_grid_points, (1, len(self.mxl_grid_points), 3)).to(self.device)
        
        
        self.mdb_lmks = self.decoder.mdb_landmarks.squeeze().cpu().numpy()
        print(self.mdb_lmks.shape)
        self.mdb_min_box = self.mdb_lmks.min(axis=0) - 0.05
        self.mdb_max_box = self.mdb_lmks.max(axis=0) + 0.05
        print(self.mdb_min_box)
        print(self.mdb_max_box)
        
        self.mdb_grid_points = create_grid_points_from_bounds(self.mdb_min_box, self.mdb_max_box, self.res)
        self.mdb_grid_points = torch.from_numpy(self.mdb_grid_points).to(self.device, dtype=torch.float)
        self.mdb_grid_points = torch.reshape(self.mdb_grid_points, (1, len(self.mdb_grid_points), 3)).to(self.device)
        

    def reduce_lr(self, epoch):
        if epoch > 0 and self.cfg['lr_decay_interval_lat'] is not None and epoch % self.cfg['lr_decay_interval_lat'] == 0:
            decay_steps = int(epoch/self.cfg['lr_decay_interval_lat'])
            lr = self.cfg['lr_lat'] * self.cfg['lr_decay_factor_lat']**decay_steps
            self.current_lr_lat = lr
            print('Reducting LR for latent codes to {}'.format(lr))
            for param_group in self.optimizer_lat_val.param_groups:
                param_group["lr"] = lr

    def test_model(self, epochs):
        latent_code_dict = dict()
        pbar = tqdm(range(epochs), desc="Fitting Latent Codes")
        for epoch in pbar:
            self.reduce_lr(epoch)
            val_loss_dict, latent_code_dict = self.compute_val_loss(epoch)
            sft_surf_sdf = val_loss_dict['sft_surf_sdf'] # on surface loss
            mdb_surf_sdf = val_loss_dict['mdb_surf_sdf'] # on surface loss
            
            mxl_surf_sdf = val_loss_dict['mxl_surf_sdf'] # on surface loss
            mxl_normals = val_loss_dict['mxl_normals'] # on surface normal loss
            mxl_landmarks = val_loss_dict['mxl_landmarks'] # mxl lmk loss
            total_loss = val_loss_dict['loss']
            
            pbar.set_description(f'Epoch: {epoch}|lr = {self.current_lr_lat}|total loss = {total_loss:.6f}|sft_surf_sdf = {sft_surf_sdf:.6f}|mxl_surf_sdf = {mxl_surf_sdf:.6f}|mdb_surf_sdf = {mdb_surf_sdf:.6f}')
        self.fit_val_dataset()






    def compute_val_loss(self, epoch):
        self.decoder.eval()
        latent_code_dict = dict()
        sum_val_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
        sum_val_loss_dict.update({'loss': 0.0})

        c = 0
        for val_batch in self.val_data_loader:
            self.optimizer_lat_val.zero_grad()
            
            l_dict, latent_code_dict= compute_all_loss(val_batch, self.decoder, self.latent_codes_val, self.device)
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
        return sum_val_loss_dict, latent_code_dict


    def log_dict(self, cfg):
        return cfg
    
    def fit_val_dataset(self, epoch = None):
        self.decoder.eval()
        exp_dir = self.output_fd
        os.makedirs(exp_dir, exist_ok=True)
        
        
            
        for idx in range(len(self.val_dataset)):
            iden_val = self.val_dataset.subjects_id[idx]
            cur_fd = os.path.join(exp_dir, iden_val)
            os.makedirs(cur_fd, exist_ok=True)
            encoding = self.latent_codes_val(
                    torch.from_numpy(np.array([[idx]])).to(self.device)).squeeze().unsqueeze(0)
            print(f'Reconstruction idx = {idx} | patient id = {iden_val}')
            latent_code = encoding.cpu().detach().numpy().squeeze(0)
            np.save(os.path.join(cur_fd, 'shape_latent_code.npy'), latent_code)
            
            
            logits_val_sft = get_logits_all(self.decoder,
                                          encoding,
                                          self.grid_points.clone(),
                                          nbatch_points=25000,
                                          mode = 'sft'
                                          )
            
            trim_val_sft = mesh_from_logits(logits_val_sft, self.min, self.max, self.res)
            trim_val_sft.export(cur_fd + f'/sft.ply')
            
            logits_val_mxl = get_logits_all(self.decoder,
                                          encoding,
                                          self.grid_points.clone(),
                                          nbatch_points=25000,
                                          mode = 'mxl'
                                          )
            
            trim_val_mxl = mesh_from_logits(logits_val_mxl, self.min, self.max, self.res)
            trim_val_mxl.export(cur_fd + f'/mxl.ply')
            
            logits_val_mdb = get_logits_all(self.decoder,
                                          encoding,
                                          self.grid_points.clone(),
                                          nbatch_points=25000,
                                          mode = 'mdb'
                                          )
            
            trim_val_mdb = mesh_from_logits(logits_val_mdb, self.min, self.max, self.res)
            trim_val_mdb.export(cur_fd + f'/mdb.ply')
            
            

            
            
        

