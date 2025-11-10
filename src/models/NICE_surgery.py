import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class DeepSDF(nn.Module):
    def __init__(
            self,
            lat_dim,
            hidden_dim,
            nlayers=8,
            geometric_init=True,
            radius_init=1,
            beta=100,
            out_dim=1,
            num_freq_bands=None,
            input_dim=3,
    ):
        super().__init__()
        if num_freq_bands is None:
            d_in_spatial = input_dim
        else:
            d_in_spatial = input_dim*(2*num_freq_bands+1)
        d_in = lat_dim + d_in_spatial
        self.lat_dim = lat_dim
        self.input_dim = input_dim
        print(d_in)
        print(hidden_dim)
        dims = [hidden_dim] * nlayers
        dims = [d_in] + dims + [out_dim]

        self.num_layers = len(dims)
        self.skip_in = [nlayers//2]
        self.num_freq_bands = num_freq_bands
        if num_freq_bands is not None:
            fun = lambda x: 2 ** x
            self.freq_bands = fun(torch.arange(num_freq_bands))

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            # if true preform preform geometric initialization
            if geometric_init:

                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        else:
            self.activation = nn.ReLU()

    def forward(self, xyz, lat_rep, landmarks=None):

        if self.num_freq_bands is not None:
            pos_embeds = [xyz]
            for freq in self.freq_bands:
                pos_embeds.append(torch.sin(xyz* freq))
                pos_embeds.append(torch.cos(xyz * freq))

            pos_embed = torch.cat(pos_embeds, dim=-1)
            inp = torch.cat([pos_embed, lat_rep], dim=-1)
        else:
            inp = torch.cat([xyz, lat_rep], dim=-1)
        x = inp

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, inp], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x, None


def sample_point_feature(q, p, fea, var=0.1**2, background=False):
    # q: B x M x 3
    # p: B x N x 3
    # fea: B x N x c_dim
    # p, fea = c

    dist = -((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3) + 10e-6) ** 2
    if background:
        dist_const = torch.ones_like(dist[:, :, :1])* (-0.2)
        dist = torch.cat([dist, dist_const], dim=-1)

    weight = (dist / var).exp()  # Guassian kernel

    # weight normalization
    weight = weight / (weight.sum(dim=2).unsqueeze(-1) + 1e-6)
    #c_out = weight @ fea  # B x M x c_dim
    c_out = (weight.unsqueeze(-1) * fea).sum(dim=2)
    return c_out


class NICE_surgery(nn.Module):
    def __init__(
            self,
            mode,
            lat_dim_surgery, 
            lat_dim_id,  
            lat_dim_glob_shape, 
            
            
            sft_lat_dim_loc,
            sft_n_loc, # number of facial landmark points
            sft_landmarks, # average landmark positions
            sft_hidden_dim,
            sft_n_layers,
            
            mxl_lat_dim_loc,
            mxl_n_loc, # number of facial landmark points
            mxl_landmarks, # average landmark positions
            mxl_hidden_dim,
            mxl_n_layers,
            
            mdb_lat_dim_loc,
            mdb_n_loc, # number of facial landmark points
            mdb_landmarks, # average landmark positions
            mdb_hidden_dim,
            mdb_n_layers,
            
            
            
            out_dim=1,
            input_dim=3,
    ):
        super().__init__()
        self.mode = mode
        self.lat_dim_glob_shape = lat_dim_glob_shape  
        self.lat_dim_surgery = lat_dim_surgery            

        self.input_dim = input_dim               
        self.out_dim = out_dim + 1                 
        
        
        self.sft_lat_dim_loc = sft_lat_dim_loc   
        self.mxl_lat_dim_loc = mxl_lat_dim_loc   
        self.mdb_lat_dim_loc = mdb_lat_dim_loc   
        self.sft_num_kps = sft_n_loc            
        self.mxl_num_kps = mxl_n_loc              
        self.mdb_num_kps = mdb_n_loc              
        sft_hidden_dim = sft_hidden_dim  
        mxl_hidden_dim = mxl_hidden_dim  
        mdb_hidden_dim = mdb_hidden_dim  
        
        self.sft_landmarks = sft_landmarks
        self.mxl_landmarks = mxl_landmarks
        self.mdb_landmarks = mdb_landmarks

        if self.mode == 'compress': # F_ex conditioned on z_ex and a projection of z_id and landmarks to lower dimensions
            self.lat_dim = lat_dim_surgery + lat_dim_id   
            
                
            self.sft_compressor = nn.Sequential(
            nn.Linear((self.sft_lat_dim_loc + 3) * (self.sft_num_kps) + self.sft_lat_dim_loc + lat_dim_glob_shape, 512),
                        nn.ReLU(),
                        nn.Linear(512, lat_dim_id),
                    )
            
            self.mxl_compressor = nn.Sequential(
            nn.Linear((self.mxl_lat_dim_loc + 3) * (self.mxl_num_kps) + self.mxl_lat_dim_loc + lat_dim_glob_shape, 512),
                        nn.ReLU(),
                        nn.Linear(512, lat_dim_id),
                    )
            
            self.mdb_compressor = nn.Sequential(
            nn.Linear((self.mdb_lat_dim_loc + 3) * (self.mdb_num_kps) + self.mdb_lat_dim_loc + lat_dim_glob_shape, 512),
                        nn.ReLU(),
                        nn.Linear(512, lat_dim_id),
                    )
            
            

        elif self.mode == 'GNN':
            self.lat_dim = lat_dim_surgery * 2
        else:
            raise ValueError('Unknown mode!')


        print('creating DeepSDF with...')
        print('lat dim', self.lat_dim)

        self.sft_defDeepSDF = DeepSDF(lat_dim=self.lat_dim, 
                                  hidden_dim=sft_hidden_dim, 
                                  nlayers=sft_n_layers,
                                  geometric_init=False,
                                  out_dim=out_dim,
                                  input_dim=input_dim).float()
        
        self.mxl_defDeepSDF = DeepSDF(lat_dim=self.lat_dim, 
                                  hidden_dim=mxl_hidden_dim, 
                                  nlayers=mxl_n_layers,
                                  geometric_init=False,
                                  out_dim=out_dim,
                                  input_dim=input_dim).float()
        
        self.mdb_defDeepSDF = DeepSDF(lat_dim=self.lat_dim,
                                  hidden_dim=mdb_hidden_dim, 
                                  nlayers=mdb_n_layers,
                                  geometric_init=False,
                                  out_dim=out_dim,
                                  input_dim=input_dim).float()



    def forward(self,
                xyz : torch.tensor,
                lat_rep : torch.tensor,
                landmarks : Optional[torch.tensor],
                surgery_mode='sft') -> (torch.tensor, torch.tensor):
        '''
         xyz: B x N x 3 : queried 3D coordinates
         lat: B x N x lat_dim : latent code, concatenation of [z_id, z_ex]
         landmarks: B x N x n_kps x 3 : facial landmark positions in case F_id uses such

         returns: offsets that model the deformation for each queried points.
           Remaining features are returned separately if there are any
        '''


        if len(xyz.shape) < 3:
            xyz = xyz.unsqueeze(0)

        B, N, _ = xyz.shape
        if self.mode == 'compress':
            global_start, global_end = 0, self.lat_dim_glob_shape
            sft_start, sft_end = global_end, global_end + self.sft_lat_dim_loc * (self.sft_num_kps + 1)
            mxl_start, mxl_end = sft_end, sft_end + self.mxl_lat_dim_loc * (self.mxl_num_kps + 1)
            mdb_start, mdb_end = mxl_end, mxl_end + self.mdb_lat_dim_loc * (self.mdb_num_kps + 1)
            surgery_start = mdb_end
            


            
            if surgery_mode == 'sft':
                if not landmarks.shape[1] == N:
                    if len(landmarks.shape) != 4:
                        landmarks = landmarks.unsqueeze(1).repeat(1, N, 1, 1)
                    else:
                        landmarks = landmarks[:, 0, :, :].unsqueeze(1).repeat(1, N, 1, 1)
                concat = torch.cat([lat_rep[..., :global_end], lat_rep[..., sft_start:sft_end], landmarks.reshape(B, N, -1)], dim=-1)
                compressed = self.sft_compressor(concat[:, 0, :]).unsqueeze(1).repeat(1, N, 1)
                if self.training:
                    compressed += torch.randn(compressed.shape, device=compressed.device) / 200
                cond = torch.cat([compressed, lat_rep[..., surgery_start:]], dim=-1)
                pred = self.sft_defDeepSDF(xyz, cond)[0]
                return pred[..., :3], pred[..., -1:]
            
            if surgery_mode == 'mxl':
                if not landmarks.shape[1] == N:
                    if len(landmarks.shape) != 4:
                        landmarks = landmarks.unsqueeze(1).repeat(1, N, 1, 1)
                    else:
                        landmarks = landmarks[:, 0, :, :].unsqueeze(1).repeat(1, N, 1, 1)
                concat = torch.cat([lat_rep[..., :global_end], lat_rep[..., mxl_start:mxl_end], landmarks.reshape(B, N, -1)], dim=-1)
                compressed = self.mxl_compressor(concat[:, 0, :]).unsqueeze(1).repeat(1, N, 1)
                if self.training: 
                    compressed += torch.randn(compressed.shape, device=compressed.device) / 200
                cond = torch.cat([compressed, lat_rep[..., surgery_start:]], dim=-1)
                pred = self.mxl_defDeepSDF(xyz, cond)[0]
                return pred[..., :3], pred[..., -1:]
            
            
            if surgery_mode == 'mdb':
                if not landmarks.shape[1] == N:
                    if len(landmarks.shape) != 4:
                        landmarks = landmarks.unsqueeze(1).repeat(1, N, 1, 1)
                    else:
                        landmarks = landmarks[:, 0, :, :].unsqueeze(1).repeat(1, N, 1, 1)
                concat = torch.cat([lat_rep[..., :global_end], lat_rep[..., mdb_start:mdb_end], landmarks.reshape(B, N, -1)], dim=-1)
                compressed = self.mdb_compressor(concat[:, 0, :]).unsqueeze(1).repeat(1, N, 1)
                if self.training:
                    compressed += torch.randn(compressed.shape, device=compressed.device) / 200
                cond = torch.cat([compressed, lat_rep[..., surgery_start:]], dim=-1)
                pred = self.mdb_defDeepSDF(xyz, cond)[0]
                return pred[..., :3], pred[..., -1:]
                
            
            

