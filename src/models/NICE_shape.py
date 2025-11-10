
import torch
import torch.nn as nn
import numpy as np
import math



class EnsembledLinear(nn.Module):
    def __init__(self, ensemble_size, n_symm, in_features, out_features, bias=True):
        super().__init__()
        self.ensemble_size = ensemble_size 
        self.n_symm = n_symm
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.Tensor(ensemble_size - self.n_symm, out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(ensemble_size - self.n_symm, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for e in range(self.ensemble_size - self.n_symm):
            torch.nn.init.kaiming_uniform_(self.weight[e, ...], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[e, ...])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.bias[e, ...], -bound, bound)

    def forward(self, input):
        W = torch.cat([
            self.weight[:self.n_symm, ...].repeat_interleave(2, dim=0), self.weight[self.n_symm:, ...]],
                      dim=0)

        # perform matrix multiplication
        output = torch.bmm(W, input.permute(0, 2, 1)).permute(0, 2, 1) # A x B x D_out

        if self.bias is not None:
            b = torch.cat(
                [self.bias[:self.n_symm, ...].repeat_interleave(2, dim=0), self.bias[self.n_symm:, ...]],
                    dim=0)
            output += b.unsqueeze(1)
        return output


class EnsembledDeepSDF(nn.Module):
    '''
    Execute multiple DeepSDF networks in parallel
    '''
    def __init__(
            self,
            ensemble_size,
            n_symm,
            lat_dim,
            hidden_dim,
            nlayers,
            out_dim=1,
            input_dim=3,
    ):
        super().__init__()
        d_in = input_dim+lat_dim 

        self.ensemble_size = ensemble_size 
        self.n_symm = n_symm  
        self.lat_dim = lat_dim 
        self.input_dim = input_dim 

        dims = [hidden_dim] * nlayers  
        dims = [d_in] + dims + [out_dim] 

        self.num_layers = len(dims)
        self.skip_in = [nlayers//2] 
        

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - d_in
                in_dim = dims[layer]
            else:
                out_dim = dims[layer + 1]
                in_dim = dims[layer]

            lin = EnsembledLinear(self.ensemble_size, self.n_symm, in_dim, out_dim)
            setattr(self, "lin" + str(layer), lin)
        

        self.activation = nn.Softplus(beta=100)

    def forward(self, xyz, lat_rep):
        # xyz: A x B x nPoints x 3
        # lat_rep: A x B x nPoints x nFeats
        A, B, nP, _ = xyz.shape

        # A: landmarks + 1, B: batch, nP: N
        inp = torch.cat([xyz, lat_rep], dim=-1)

        # merge batch and point dimension
        inp = inp.reshape(A, B*nP, -1) # A x (B*nP) x (3+nFeats)
        x = inp

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, inp], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

        # un-merge dimensions
        x = x.reshape(A, B, nP, -1)

        return x


def sample_point_feature(q, p, fea, var=0.1**2, background=False):
    # q: B x n_points x 3
    # p: B x n_kps x 3
    # fea: B x n_points x n_kps x channel_dim


    # distance betweeen each query point to the point cloud
    dist = -((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3) + 10e-6) ** 2 


    

    # add "background" mlp that doesn't have landmark point
    if background:
        dist_const = torch.ones_like(dist[:, :, :1])* (-0.2)
        dist = torch.cat([dist, dist_const], dim=-1)

    weight = (dist / var).exp()  # Guassian kernel

    # weight normalization
    weight = weight / (weight.sum(dim=2).unsqueeze(-1) + 1e-6)

    c_out = (weight.unsqueeze(-1) * fea).sum(dim=2) # B x n_points x channel_dim
    return c_out


class NICE_shape(nn.Module):
    def __init__(
            self,
            lat_dim_glob : int,  # lat_dim_glob is fixed as global feature for sft, mxl, mdb
            sft_lat_dim_loc : int,
            sft_n_loc : int, # number of facial landmark points
            sft_n_symm_pairs: int, 
            sft_landmarks : torch.tensor, # average landmark positions
            sft_hidden_dim : int,
            sft_n_layers : int,
            
            mxl_lat_dim_loc : int,
            mxl_n_loc : int, # number of maxilla landmark points
            mxl_n_symm_pairs: int, 
            mxl_landmarks : torch.tensor, # average landmark positions
            mxl_hidden_dim : int,
            mxl_n_layers : int,
            
            mdb_lat_dim_loc : int,
            mdb_n_loc : int, # number of mandible landmark points
            mdb_n_symm_pairs: int, 
            mdb_landmarks : torch.tensor, # average landmark positions
            mdb_hidden_dim : int,
            mdb_n_layers : int,
            
            sft_pos_mlp_dim : int=256, # hidden dim of MLP_pos
            mxl_pos_mlp_dim : int=256, # hidden dim of MLP_pos
            mdb_pos_mlp_dim : int=256, # hidden dim of MLP_pos
            
            out_dim : int=1, # dimensionality of the modeled neural field
            input_dim : int=3, # (input) domain of the modeled neural field
    ):
        super().__init__()
        # self.lat_dim_glob is fixed as global feature for sft, mxl, mdb
        # the latent code is written as [Zglobal, Zsft, Zmxl, Zmdb]
        self.lat_dim_glob = lat_dim_glob  
        self.input_dim = input_dim
        self.out_dim = out_dim
        
        self.sft_lat_dim_loc = sft_lat_dim_loc    
        self.mxl_lat_dim_loc = mxl_lat_dim_loc    
        self.mdb_lat_dim_loc = mdb_lat_dim_loc    
        
        self.lat_dim = lat_dim_glob + (sft_n_loc+1) * sft_lat_dim_loc \
            + (mxl_n_loc+1) * mxl_lat_dim_loc  + (mdb_n_loc+1) * mdb_lat_dim_loc

        

        self.sft_pos_mlp_dim = sft_pos_mlp_dim    
        self.mxl_pos_mlp_dim = mxl_pos_mlp_dim    
        self.mdb_pos_mlp_dim = mdb_pos_mlp_dim    

        self.sft_num_kps = sft_n_loc             
        self.mxl_num_kps = mxl_n_loc             
        self.mdb_num_kps = mdb_n_loc           
        
        self.sft_num_symm_pairs = sft_n_symm_pairs 
        self.mxl_num_symm_pairs = mxl_n_symm_pairs  
        self.mdb_num_symm_pairs = mdb_n_symm_pairs  

        sft_lat_dim_part = lat_dim_glob + self.sft_lat_dim_loc 
        mxl_lat_dim_part = lat_dim_glob + self.mxl_lat_dim_loc 
        mdb_lat_dim_part = lat_dim_glob + self.mdb_lat_dim_loc 
        
        sft_hidden_dim = sft_hidden_dim 
        mxl_hidden_dim = mxl_hidden_dim 
        mdb_hidden_dim = mdb_hidden_dim 

        self.sft_ensembled_deep_sdf = EnsembledDeepSDF(ensemble_size=self.sft_num_kps+1,
                                                   n_symm=self.sft_num_symm_pairs,
                                                   lat_dim=sft_lat_dim_part,
                                                   hidden_dim=sft_hidden_dim,
                                                   nlayers=sft_n_layers,
                                                   out_dim=out_dim,
                                                   input_dim=input_dim,
                                                   ).float()
        
        self.mxl_ensembled_deep_sdf = EnsembledDeepSDF(ensemble_size=self.mxl_num_kps+1,
                                                   n_symm=self.mxl_num_symm_pairs,
                                                   lat_dim=mxl_lat_dim_part,
                                                   hidden_dim=mxl_hidden_dim,
                                                   nlayers=mxl_n_layers,
                                                   out_dim=out_dim,
                                                   input_dim=input_dim,
                                                   ).float()
        
        self.mdb_ensembled_deep_sdf = EnsembledDeepSDF(ensemble_size=self.mdb_num_kps+1,
                                                   n_symm=self.mdb_num_symm_pairs,
                                                   lat_dim=mdb_lat_dim_part,
                                                   hidden_dim=mdb_hidden_dim,
                                                   nlayers=mdb_n_layers,
                                                   out_dim=out_dim,
                                                   input_dim=input_dim,
                                                   ).float()
        

        self.sft_landmarks = sft_landmarks
        self.mxl_landmarks = mxl_landmarks
        self.mdb_landmarks = mdb_landmarks

        self.sft_mlp_pos = nn.Sequential(
            nn.Linear(self.lat_dim_glob, self.sft_pos_mlp_dim),
            nn.ReLU(),
            nn.Linear(self.sft_pos_mlp_dim, self.sft_pos_mlp_dim),   
            nn.ReLU(),
            nn.Linear(self.sft_pos_mlp_dim, self.sft_num_kps * 3)   
        )
        
        self.mxl_mlp_pos = nn.Sequential(
            nn.Linear(self.lat_dim_glob, self.mxl_pos_mlp_dim), 
            nn.ReLU(),
            nn.Linear(self.mxl_pos_mlp_dim, self.mxl_pos_mlp_dim), 
            nn.ReLU(),
            nn.Linear(self.mxl_pos_mlp_dim, self.mxl_num_kps * 3)   
        )
        
        self.mdb_mlp_pos = nn.Sequential(
            nn.Linear(self.lat_dim_glob, self.mdb_pos_mlp_dim), 
            nn.ReLU(),
            nn.Linear(self.mdb_pos_mlp_dim, self.mdb_pos_mlp_dim),  
            nn.ReLU(),
            nn.Linear(self.mdb_pos_mlp_dim, self.mdb_num_kps * 3)   
        )


    def forward(self,
                xyz :torch.tensor,
                lat_rep : torch.tensor,
                # sft_landmarks_gt : Optional[torch.tensor]
                mode : str,
                ) ->(torch.tensor, torch.tensor):
        '''
        xyz: B x N x 3 : queried 3D coordinates
        lat_rep: B x N x self.lat_dim
        returns: predictd sdf values, and predicted facial landmark positions
        '''

        if len(xyz.shape) < 3:
            xyz = xyz.unsqueeze(0)

        B, N, _ = xyz.shape
        if lat_rep.shape[1] == 1:
            lat_rep = lat_rep.repeat(1, N, 1)

        assert self.lat_dim == lat_rep.shape[-1], 'lat dim {}, lat_rep {}'.format(self.lat_dim, lat_rep.shape)

        if mode == 'sft':
            sft_landmarks = self.sft_mlp_pos(lat_rep[:, 0, :self.lat_dim_glob]).view(B, self.sft_num_kps, 3) # B x n_kps x 3
            sft_landmarks += self.sft_landmarks.squeeze(0)

            if len(sft_landmarks.shape) < 4:
                sft_landmarks = sft_landmarks.unsqueeze(1).repeat(1, N, 1, 1) # B x N x n_kps x 3
            else:
                sft_landmarks = sft_landmarks.repeat(1, N, 1, 1)
 
            

            # represent xyz in all local coordinate systems
            # for the very last landmark there is no local coordinate system, it uses the global one instead
            sft_coords = xyz.unsqueeze(2) - torch.cat([sft_landmarks,
                                                torch.zeros_like(sft_landmarks[:, :, :1, :])], dim=2)  # B x N x nkps x 3


            sft_coords[:, :, 1:2*self.sft_num_symm_pairs:2, 0] *= -1

            # prepare latent codes
            t1 = lat_rep[:, :, :self.lat_dim_glob].unsqueeze(2).repeat(1, 1, self.sft_num_kps+1, 1)
            t2 = lat_rep[:, :, self.lat_dim_glob:self.lat_dim_glob+(self.sft_num_kps+1)*self.sft_lat_dim_loc].reshape(B, -1, self.sft_num_kps+1, self.sft_lat_dim_loc)

            if t2.shape[1] != N:
                t2 = t2.repeat(1, N, 1, 1)
                t1 = t1.repeat(1, N, 1, 1)
            cond = torch.cat([t1, t2], dim=-1) 

            sft_coords = sft_coords.permute(2, 0, 1, 3) 
            cond = cond.permute(2, 0, 1, 3) 

            sdf_pred = self.sft_ensembled_deep_sdf(sft_coords, cond)
            



            if not self.training:
                sdf_pred[:, :, -1, 0] = 1

            sdf_pred = sdf_pred.permute(1, 2, 0, 3) # B x N x nkps x 1
            # blend predictions
            pred = sample_point_feature(xyz[..., :3], sft_landmarks[:, 0, :, :3], sdf_pred, background=True, var=0.1**2)


            return pred, sft_landmarks[:, 0, :, :], cond
        
        elif mode == 'mxl':
            mxl_landmarks = self.mxl_mlp_pos(lat_rep[:, 0, :self.lat_dim_glob]).view(B, self.mxl_num_kps, 3) # B x n_kps x 3
            mxl_landmarks += self.mxl_landmarks.squeeze(0)
            if len(mxl_landmarks.shape) < 4:
                mxl_landmarks = mxl_landmarks.unsqueeze(1).repeat(1, N, 1, 1) # B x N x n_kps x 3
            else:
                mxl_landmarks = mxl_landmarks.repeat(1, N, 1, 1)

            

            # represent xyz in all local coordinate systems
            # for the very last landmark there is no local coordinate system, it uses the global one instead
            mxl_coords = xyz.unsqueeze(2) - torch.cat([mxl_landmarks,
                                                torch.zeros_like(mxl_landmarks[:, :, :1, :])], dim=2)  # B x N x nkps x 3



            mxl_coords[:, :, 1:2*self.mxl_num_symm_pairs:2, 0] *= -1

            # prepare latent codes

            t1 = lat_rep[:, :, :self.lat_dim_glob].unsqueeze(2).repeat(1, 1, self.mxl_num_kps+1, 1)
            t2 = lat_rep[:, :, self.lat_dim_glob+(self.sft_num_kps+1)*self.sft_lat_dim_loc:self.lat_dim_glob+(self.sft_num_kps+1)*self.sft_lat_dim_loc+\
                (self.mxl_num_kps+1)*self.mxl_lat_dim_loc].reshape(B, -1, self.mxl_num_kps+1, self.mxl_lat_dim_loc)

            if t2.shape[1] != N:
                t2 = t2.repeat(1, N, 1, 1)
                t1 = t1.repeat(1, N, 1, 1)
            cond = torch.cat([t1, t2], dim=-1) # B x N x nkps x (dim_glob + dim_loc)

            mxl_coords = mxl_coords.permute(2, 0, 1, 3) # nkps x B x N x 3
            cond = cond.permute(2, 0, 1, 3) # nkps x B x N x (dim_glob + dim_loc)

            sdf_pred = self.mxl_ensembled_deep_sdf(mxl_coords, cond)
            


            if not self.training:
                sdf_pred[:, :, -1, 0] = 1

            sdf_pred = sdf_pred.permute(1, 2, 0, 3) # B x N x nkps x 1
            # blend predictions
            pred = sample_point_feature(xyz[..., :3], mxl_landmarks[:, 0, :, :3], sdf_pred, background=True, var=0.1**2)


            
            return pred, mxl_landmarks[:, 0, :, :], cond
        
        elif mode == 'mdb':
            mdb_landmarks = self.mdb_mlp_pos(lat_rep[:, 0, :self.lat_dim_glob]).view(B, self.mdb_num_kps, 3) # B x n_kps x 3
            mdb_landmarks += self.mdb_landmarks.squeeze(0)


            if len(mdb_landmarks.shape) < 4:
                mdb_landmarks = mdb_landmarks.unsqueeze(1).repeat(1, N, 1, 1) # B x N x n_kps x 3
            else:
                mdb_landmarks = mdb_landmarks.repeat(1, N, 1, 1)


            # represent xyz in all local coordinate systems
            # for the very last landmark there is no local coordinate system, it uses the global one instead
            mdb_coords = xyz.unsqueeze(2) - torch.cat([mdb_landmarks,
                                                torch.zeros_like(mdb_landmarks[:, :, :1, :])], dim=2)  # B x N x nkps x 3


            mdb_coords[:, :, 1:2*self.mdb_num_symm_pairs:2, 0] *= -1

            # prepare latent codes

            t1 = lat_rep[:, :, :self.lat_dim_glob].unsqueeze(2).repeat(1, 1, self.mdb_num_kps+1, 1)
            t2 = lat_rep[:, :, self.lat_dim_glob+(self.sft_num_kps+1)*self.sft_lat_dim_loc+(self.mxl_num_kps+1)*self.mxl_lat_dim_loc:].reshape(B, -1, self.mdb_num_kps+1, self.mdb_lat_dim_loc)
            if t2.shape[1] != N:
                t2 = t2.repeat(1, N, 1, 1)
                t1 = t1.repeat(1, N, 1, 1)
            cond = torch.cat([t1, t2], dim=-1) # B x N x nkps x (dim_glob + dim_loc)

            mdb_coords = mdb_coords.permute(2, 0, 1, 3) # nkps x B x N x 3
            cond = cond.permute(2, 0, 1, 3) # nkps x B x N x (dim_glob + dim_loc)

            sdf_pred = self.mdb_ensembled_deep_sdf(mdb_coords, cond)
            

            if not self.training:
                sdf_pred[:, :, -1, 0] = 1 

            sdf_pred = sdf_pred.permute(1, 2, 0, 3) # B x N x nkps x 1
            # blend predictions
            pred = sample_point_feature(xyz[..., :3], mdb_landmarks[:, 0, :, :3], sdf_pred, background=True, var=0.1**2)
            return pred, mdb_landmarks[:, 0, :, :], cond