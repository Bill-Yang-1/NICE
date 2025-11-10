import os
import torch
from torch.utils.data import Dataset
import torch
import numpy as np
import os
import trimesh
from typing import  Optional
import traceback

sft_lmk_index = r'NICE/template/sft_lmk.txt'
mxl_lmk_index = r'NICE/template/mxl_lmk.txt'
mdb_lmk_index = r'NICE/template/mdb_lmk.txt'

def uniform_ball(n_points, rad=1.0):
    angle1 = np.random.uniform(-1, 1, n_points)
    angle2 = np.random.uniform(0, 1, n_points)
    radius = np.random.uniform(0, rad, n_points)

    r = radius ** (1/3)
    theta = np.arccos(angle1) 
    phi = 2 * np.pi * angle2
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.stack([x, y, z], axis=-1)

def read_txt_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(int(line.strip()))
    return np.array(data)

sft_lmk = read_txt_file(sft_lmk_index)
mxl_lmk = read_txt_file(mxl_lmk_index)
mdb_lmk = read_txt_file(mdb_lmk_index)

def get_sft_lmk(datapath, subject_id):
    sft_obj_path = os.path.join(datapath, subject_id, 'softtissue.obj')
    sft_v = trimesh.load(sft_obj_path, process=False).vertices
    sft_lmk_v = sft_v[sft_lmk]
    return np.array(sft_lmk_v)

def get_mxl_lmk(datapath, subject_id):
    mxl_obj_path = os.path.join(datapath, subject_id, 'maxilla.obj')
    mxl_v = trimesh.load(mxl_obj_path, process=False).vertices
    mxl_lmk_v = mxl_v[mxl_lmk]
    return np.array(mxl_lmk_v)

def get_mdb_lmk(datapath, subject_id):
    mdb_obj_path = os.path.join(datapath, subject_id, 'mandible.obj')
    mdb_v = trimesh.load(mdb_obj_path, process=False).vertices
    mdb_lmk_v = mdb_v[mdb_lmk]
    return np.array(mdb_lmk_v)
    


class DataLoader(Dataset):
    def __init__(self,
                 datapath : str,
                 n_supervision_points_face : int,
                 n_supervision_points_non_face : int,
                 batch_size : int,
                 sigma_near : float,
                 sft_lm_inds : np.ndarray,
                 mxl_lm_inds : np.ndarray,
                 mdb_lm_inds : np.ndarray,):
        
        self.sft_lm_inds = sft_lm_inds
        self.mxl_lm_inds = mxl_lm_inds
        self.mdb_lm_inds = mdb_lm_inds
        self.datapath = datapath
        self.batch_size = batch_size
        self.n_supervision_points_face = n_supervision_points_face
        self.n_supervision_points_non_face = n_supervision_points_non_face
        self.sigma_near = sigma_near
        
        self.subjects_id = [dir for dir in os.listdir(self.datapath)]
        

        if self.sft_lm_inds is not None:
            self.gt_sft_lmks = {}
            for i, iden in enumerate(self.subjects_id):
                sft_lmk = get_sft_lmk(self.datapath, iden)
                self.gt_sft_lmks[iden] = sft_lmk
        else:
            self.gt_sft_lmks = np.zeros([39, 3])

        if self.mxl_lm_inds is not None:
            self.gt_mxl_lmks = {}
            for i, iden in enumerate(self.subjects_id):
                mxl_lmk = get_mxl_lmk(self.datapath, iden)
                self.gt_mxl_lmks[iden] = mxl_lmk
        else:
            self.gt_mxl_lmks = np.zeros([43, 3])

        if self.mdb_lm_inds is not None:
            self.gt_mdb_lmks = {}
            for i, iden in enumerate(self.subjects_id):
                mdb_lmk = get_mdb_lmk(self.datapath, iden)
                self.gt_mdb_lmks[iden] = mdb_lmk
        else:
            self.gt_mdb_lmks = np.zeros([16, 3])

    def __len__(self):
        return len(self.subjects_id)
    
    
    def get_sft_on_face_npy(self,
                    subject_id: str,
                    rnd_file: Optional[int] = None) -> str:
        if rnd_file is None:
            rnd_file = 0 #np.random.randint(0, env_paths.NUM_SPLITS)
        return os.path.join(self.datapath, subject_id, f'sft_{rnd_file}_face.npy')


    def get_sft_non_face_npy(self,
                    subject_id: str,
                    rnd_file: Optional[int] = None) -> str:
        if rnd_file is None:
            rnd_file = 0 #np.random.randint(0, env_paths.NUM_SPLITS)
        return os.path.join(self.datapath, subject_id, f'sft_{rnd_file}_non_face.npy')
    

    def get_mxl_on_face_npy(self,
                    subject_id: str,
                    rnd_file: Optional[int] = None) -> str:
        if rnd_file is None:
            rnd_file = 0 #np.random.randint(0, env_paths.NUM_SPLITS)
        return os.path.join(self.datapath, subject_id, f'mxl_{rnd_file}_face.npy')


    def get_mxl_non_face_npy(self,
                    subject_id: str,
                    rnd_file: Optional[int] = None) -> str:
        if rnd_file is None:
            rnd_file = 0 #np.random.randint(0, env_paths.NUM_SPLITS)
        return os.path.join(self.datapath, subject_id, f'mxl_{rnd_file}_non_face.npy')
    

    def get_mdb_on_face_npy(self,
                    subject_id: str,
                    rnd_file: Optional[int] = None) -> str:
        if rnd_file is None:
            rnd_file = 0 #np.random.randint(0, env_paths.NUM_SPLITS)
        return os.path.join(self.datapath, subject_id, f'mdb_{rnd_file}_face.npy')

    
    
    def __getitem__(self, idx):
        iden = self.subjects_id[idx]
        if self.sft_lm_inds is not None:
            gt_sft_lmk = self.gt_sft_lmks[iden]
        if self.mxl_lm_inds is not None:
            gt_mxl_lmk = self.gt_mxl_lmks[iden]
        if self.mdb_lm_inds is not None:
            gt_mdb_lmk = self.gt_mdb_lmks[iden]

        
        
            
        try:
            # on face means: the front part of the entire facial softtissue
            # non face means: the back part of the entire facial softtissue
            sft_on_face = np.load(self.get_sft_on_face_npy(iden))
            sft_on_face_points = sft_on_face[:, :3]
            sft_on_face_normals = sft_on_face[:, 3:6]
            sft_non_face = np.load(self.get_sft_non_face_npy(iden))
            sft_non_face_points = sft_non_face[:, :3]
            sft_non_face_normals = sft_non_face[:, 3:6]

            mxl_on_face = np.load(self.get_mxl_on_face_npy(iden))
            mxl_on_face_points = mxl_on_face[:, :3]
            mxl_on_face_normals = mxl_on_face[:, 3:6]
            mxl_non_face = np.load(self.get_mxl_non_face_npy(iden))
            mxl_non_face_points = mxl_non_face[:, :3]
            mxl_non_face_normals = mxl_non_face[:, 3:6]
            
            mdb_on_face = np.load(self.get_mdb_on_face_npy(iden))
            mdb_on_face_points = mdb_on_face[:, :3]
            mdb_on_face_normals = mdb_on_face[:, 3:6]



            # subsample points for supervision
            sup_sft_idx = np.random.randint(0, sft_on_face_points.shape[0], self.n_supervision_points_face)
            sup_sft_on_face_points = sft_on_face_points[sup_sft_idx, :]
            sup_sft_on_face_normals = sft_on_face_normals[sup_sft_idx, :]
            sup_sft_idx_non = np.random.randint(0, sft_non_face.shape[0], self.n_supervision_points_non_face//5)
            sup_sft_non_face_points = sft_non_face_points[sup_sft_idx_non, :]
            sup_sft_non_face_normals = sft_non_face_normals[sup_sft_idx_non, :]

            sup_mxl_idx = np.random.randint(0, mxl_on_face_points.shape[0], self.n_supervision_points_face)
            sup_mxl_on_face_points = mxl_on_face_points[sup_mxl_idx, :]
            sup_mxl_on_face_normals = mxl_on_face_normals[sup_mxl_idx, :]
            sup_mxl_idx_non = np.random.randint(0, mxl_non_face.shape[0], self.n_supervision_points_non_face//5)
            sup_mxl_non_face_points = mxl_non_face_points[sup_mxl_idx_non, :]
            sup_mxl_non_face_normals = mxl_non_face_normals[sup_mxl_idx_non, :]
            
            sup_mdb_idx = np.random.randint(0, mdb_on_face_points.shape[0], self.n_supervision_points_face)
            sup_mdb_on_face_points = mdb_on_face_points[sup_mdb_idx, :]
            sup_mdb_on_face_normals = mdb_on_face_normals[sup_mdb_idx, :]
            
            
            
        except Exception as e:
            print('SUBJECT: {}'.format(iden))
            print(traceback.format_exc())
            return self.__getitem__(np.random.randint(0, self.__len__()))

        
        # sample points for grad-constraint
        sup_sft_grad_far = uniform_ball(self.n_supervision_points_face // 8, rad=0.5)
        sup_sft_grad_near = np.concatenate([sup_sft_on_face_points, sup_sft_non_face_points], axis=0) + \
                        np.random.randn(sup_sft_on_face_points.shape[0]+sup_sft_non_face_points.shape[0], 3) * self.sigma_near 

        sup_mxl_grad_far = uniform_ball(self.n_supervision_points_face // 2, rad=0.5)
        sup_mxl_grad_near = np.concatenate([sup_mxl_on_face_points, sup_mxl_non_face_points], axis=0) + \
                        np.random.randn(sup_mxl_on_face_points.shape[0]+sup_mxl_non_face_points.shape[0], 3) * self.sigma_near 

        sup_mdb_grad_far = uniform_ball(self.n_supervision_points_face // 8, rad=0.5)
        sup_mdb_grad_near = sup_mdb_on_face_points + np.random.randn(sup_mdb_on_face_points.shape[0], 3) * self.sigma_near


        ret_dict = {
                    'idx': np.array([idx]),
                    'sup_sft_on_face_points': sup_sft_on_face_points,
                    'sup_sft_on_face_normals': sup_sft_on_face_normals,
                    'sup_sft_grad_far': sup_sft_grad_far,
                    'sup_sft_grad_near': sup_sft_grad_near,
                    'sup_sft_non_face_points': sup_sft_non_face_points,
                    'sup_sft_non_face_normals': sup_sft_non_face_normals,
                    
                    'sup_mxl_on_face_points': sup_mxl_on_face_points,
                    'sup_mxl_on_face_normals': sup_mxl_on_face_normals,
                    'sup_mxl_grad_far': sup_mxl_grad_far,
                    'sup_mxl_grad_near': sup_mxl_grad_near,
                    'sup_mxl_non_face_points': sup_mxl_non_face_points,
                    'sup_mxl_non_face_normals': sup_mxl_non_face_normals,

                    'sup_mdb_on_face_points': sup_mdb_on_face_points,
                    'sup_mdb_on_face_normals': sup_mdb_on_face_normals,
                    'sup_mdb_grad_far': sup_mdb_grad_far,
                    'sup_mdb_grad_near': sup_mdb_grad_near,
                    }
        
        
        if not self.sft_lm_inds is None:
            ret_dict.update({'gt_sft_lmk': np.array(gt_sft_lmk)})
        if not self.mxl_lm_inds is None:
            ret_dict.update({'gt_mxl_lmk': np.array(gt_mxl_lmk)})
        if not self.sft_lm_inds is None:
            ret_dict.update({'gt_mdb_lmk': np.array(gt_mdb_lmk)})

        
        return ret_dict
    
    def get_loader(self, shuffle=False):
        #random.seed(0)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=8, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn,
            pin_memory=True)
        
    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
        
        
        
