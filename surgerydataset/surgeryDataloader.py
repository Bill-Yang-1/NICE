import os
import torch
import random
from torch.utils.data import Dataset
import torch
import numpy as np
import os
import trimesh

sft_lmk_index = r'NICE/template/sft_lmk.txt'
mxl_lmk_index = r'NICE/template/mxl_lmk.txt'
mdb_lmk_index = r'NICE/template/mdb_lmk.txt'

def read_txt_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(int(line.strip()))
    return np.array(data)

sft_lmk = read_txt_file(sft_lmk_index)
mxl_lmk = read_txt_file(mxl_lmk_index)
mdb_lmk = read_txt_file(mdb_lmk_index)

def get_sft_lmk(datapath, subject_id, pre_post):
    sft_obj_path = os.path.join(datapath, subject_id, f'ori_sft_{pre_post}.obj')
    sft_v = trimesh.load(sft_obj_path, process=False).vertices
    sft_lmk_v = sft_v[sft_lmk]
    return np.array(sft_lmk_v)

def get_mxl_lmk(datapath, subject_id, pre_post):
    mxl_obj_path = os.path.join(datapath, subject_id, f'ori_mxl_{pre_post}.obj')
    mxl_v = trimesh.load(mxl_obj_path, process=False).vertices
    mxl_lmk_v = mxl_v[mxl_lmk]
    return np.array(mxl_lmk_v)

def get_mdb_lmk(datapath, subject_id, pre_post):
    mdb_obj_path = os.path.join(datapath, subject_id, f'ori_mdb_{pre_post}.obj')
    mdb_v = trimesh.load(mdb_obj_path, process=False).vertices
    mdb_lmk_v = mdb_v[mdb_lmk]
    return np.array(mdb_lmk_v)


class SurgeryDataLoader(Dataset):
    def __init__(self,
                 datapath : str,
                 n_supervision_points : int,
                 batch_size : int,
                 sft_lm_inds : np.ndarray,
                 mxl_lm_inds : np.ndarray,
                 mdb_lm_inds : np.ndarray,
                 ):
        

        self.sft_lm_inds = sft_lm_inds
        self.mxl_lm_inds = mxl_lm_inds
        self.mdb_lm_inds = mdb_lm_inds
        self.datapath = datapath
        
        self.batch_size = batch_size
        
        self.subjects_id = [dir for dir in os.listdir(self.datapath)]
        
        self.n_supervision_points = n_supervision_points
        
        if self.sft_lm_inds is not None:
            self.gt_sft_lmks = {}
            for i, iden in enumerate(self.subjects_id):
                sft_lmk = get_sft_lmk(self.datapath, iden, 'pre')
                self.gt_sft_lmks[iden] = sft_lmk
        else:
            self.gt_sft_lmks = np.zeros([39, 3])

        if self.mxl_lm_inds is not None:
            self.gt_mxl_lmks = {}
            for i, iden in enumerate(self.subjects_id):
                mxl_lmk = get_mxl_lmk(self.datapath, iden, 'pre')
                self.gt_mxl_lmks[iden] = mxl_lmk
        else:
            self.gt_mxl_lmks = np.zeros([43, 3])

        if self.mdb_lm_inds is not None:
            self.gt_mdb_lmks = {}
            for i, iden in enumerate(self.subjects_id):
                mdb_lmk = get_mdb_lmk(self.datapath, iden, 'pre')
                self.gt_mdb_lmks[iden] = mdb_lmk
        else:
            self.gt_mdb_lmks = np.zeros([16, 3])
            
            
        
    def __len__(self):
        return len(self.subjects_id)
    
    
    def __getitem__(self, idx):
        iden = self.subjects_id[idx]
        if self.sft_lm_inds is not None:
            gt_sft_lmk = self.gt_sft_lmks[iden]
        if self.mxl_lm_inds is not None:
            gt_mxl_lmk = self.gt_mxl_lmks[iden]
        if self.mdb_lm_inds is not None:
            gt_mdb_lmk = self.gt_mdb_lmks[iden]
            
            
        # below is for soft    
        sft_rnd_file = 0#np.random.randint(0, env_paths.NUM_SPLITS_EXPR)
        
        sft_point_corresp = np.load(os.path.join(self.datapath, iden, f'sft_corresp_{sft_rnd_file}.npy'))

        sft_valid = np.logical_not( np.any(np.isnan(sft_point_corresp), axis=-1))
        sft_point_corresp = sft_point_corresp[sft_valid, :].astype(np.float32)

        
        sft_point_corresp_normals = np.load(os.path.join(self.datapath, iden, f'sft_corresp_normals_{sft_rnd_file}.npy'))

        sft_normals_valid = np.logical_not( np.any(np.isnan(sft_point_corresp_normals), axis=-1))
        sft_point_corresp_normals = sft_point_corresp_normals[sft_normals_valid, :].astype(np.float32)

        sft_sup_idx = 0#np.random.randint(0, sft_point_corresp.shape[0], self.n_supervision_points)
        sft_sup_points_pre = sft_point_corresp[sft_sup_idx, :3]
        sft_sup_points_post = sft_point_corresp[sft_sup_idx, 3:]
        sft_sup_points_normals_pre = sft_point_corresp_normals[sft_sup_idx, :3]
        sft_sup_points_normals_post = sft_point_corresp_normals[sft_sup_idx, 3:]
        
        

        mxl_rnd_file = 0#np.random.randint(0, env_paths.NUM_SPLITS_EXPR)
        
        mxl_point_corresp = np.load(os.path.join(self.datapath, iden, f'mxl_corresp_{mxl_rnd_file}.npy'))

        mxl_valid = np.logical_not( np.any(np.isnan(mxl_point_corresp), axis=-1))
        mxl_point_corresp = mxl_point_corresp[mxl_valid, :].astype(np.float32)

        
        mxl_point_corresp_normals = np.load(os.path.join(self.datapath, iden, f'mxl_corresp_normals_{mxl_rnd_file}.npy'))

        mxl_normals_valid = np.logical_not( np.any(np.isnan(mxl_point_corresp_normals), axis=-1))
        mxl_point_corresp_normals = mxl_point_corresp_normals[mxl_normals_valid, :].astype(np.float32)

        
        mxl_sup_idx = 0#np.random.randint(0, mxl_point_corresp.shape[0], self.n_supervision_points)
        mxl_sup_points_pre = mxl_point_corresp[mxl_sup_idx, :3]
        mxl_sup_points_post = mxl_point_corresp[mxl_sup_idx, 3:]
        mxl_sup_points_normals_pre = mxl_point_corresp_normals[mxl_sup_idx, :3]
        mxl_sup_points_normals_post = mxl_point_corresp_normals[mxl_sup_idx, 3:]
        
        
        
        # below is for mandible   
        mdb_rnd_file = 0#np.random.randint(0, env_paths.NUM_SPLITS_EXPR)
        
        mdb_point_corresp = np.load(os.path.join(self.datapath, iden, f'mdb_corresp_{mdb_rnd_file}.npy'))

        mdb_valid = np.logical_not( np.any(np.isnan(mdb_point_corresp), axis=-1))
        mdb_point_corresp = mdb_point_corresp[mdb_valid, :].astype(np.float32)

        
        mdb_point_corresp_normals = np.load(os.path.join(self.datapath, iden, f'mdb_corresp_normals_{mdb_rnd_file}.npy'))

        mdb_normals_valid = np.logical_not( np.any(np.isnan(mdb_point_corresp_normals), axis=-1))
        mdb_point_corresp_normals = mdb_point_corresp_normals[mdb_normals_valid, :].astype(np.float32)

        
        mdb_sup_idx = 0#np.random.randint(0, mdb_point_corresp.shape[0], self.n_supervision_points)
        mdb_sup_points_pre = mdb_point_corresp[mdb_sup_idx, :3]
        mdb_sup_points_post = mdb_point_corresp[mdb_sup_idx, 3:]
        mdb_sup_points_normals_pre = mdb_point_corresp_normals[mdb_sup_idx, :3]
        mdb_sup_points_normals_post = mdb_point_corresp_normals[mdb_sup_idx, 3:]
        
        
        # pre shape latent code
        shape_latent_code = np.load(os.path.join(self.datapath, iden, f'shape_latent_code.npy'))
        
        
        return {'sft_points_pre': sft_sup_points_pre,
                'sft_points_post': sft_sup_points_post,
                'sft_points_normals_pre': sft_sup_points_normals_pre,
                'sft_points_normals_post': sft_sup_points_normals_post,
                
                'mxl_points_pre': mxl_sup_points_pre,
                'mxl_points_post': mxl_sup_points_post,
                'mxl_points_normals_pre': mxl_sup_points_normals_pre,
                'mxl_points_normals_post': mxl_sup_points_normals_post,
                
                'mdb_points_pre': mdb_sup_points_pre,
                'mdb_points_post': mdb_sup_points_post,
                'mdb_points_normals_pre': mdb_sup_points_normals_pre,
                'mdb_points_normals_post': mdb_sup_points_normals_post,
                
                
                'idx': np.array([idx]),
                'shape_latent_code': shape_latent_code,
                

                'gt_sft_lmk': np.array(gt_sft_lmk),
                'gt_mxl_lmk': np.array(gt_mxl_lmk),
                'gt_mdb_lmk': np.array(gt_mdb_lmk),
                
                }

        
        
    def get_shape_latent_code(self, idx):
        iden = self.subjects_id[idx]
        shape_latent_code = np.load(os.path.join(self.datapath, iden, f'shape_latent_code.npy'))
        return shape_latent_code
        
        
    def get_loader(self, shuffle=True):
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=8, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn,
            pin_memory=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
        
