import os
import sys
import argparse
import torch
import json, os, yaml
import torch
import numpy as np
sys.path.append(r'NICE')
from src import env_paths
from surgerydataset.surgeryDataloader import SurgeryDataLoader
from src.models.NICE_surgery import NICE_surgery
from src.training.training_surgery import SurgeryTrainerAutoDecoder

from src.models.NICE_shape import NICE_shape

parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-cfg_file', type=str)
parser.add_argument('-ckpt', type=int)
parser.add_argument('-mode', required=True, type=str)


try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

assert args.cfg_file is not None
CFG = yaml.safe_load(open(args.cfg_file, 'r'))

CFG['ex_decoder']['mode'] = args.mode

exp_dir = env_paths.EXPERIMENT_DIR + '/{}/'.format(args.exp_name)
fname = exp_dir + 'configs.yaml'
if not os.path.exists(exp_dir):
    print('Creating checkpoint dir: ' + exp_dir)
    os.makedirs(exp_dir)
    with open(fname, 'w') as yaml_file:
        yaml.safe_dump(CFG, yaml_file, default_flow_style=False)
else:
    with open(fname, 'r') as f:
        print('Loading config file from: ' + fname)
        CFG = yaml.safe_load(f)

print(json.dumps(CFG, sort_keys=True, indent=4))

device = torch.device("cuda")


sft_lm_inds = np.load(r'NICE/template/sft_lmk.npy')
sft_mean_lmk_coords = torch.from_numpy(np.load(r'NICE/template/sft_train_mean_lmk_coords.npy')).float().unsqueeze(0).unsqueeze(0).to(device)
mxl_lm_inds = np.load(r'NICE/template/mxl_lmk.npy')
mxl_mean_lmk_coords = torch.from_numpy(np.load(r'NICE/template/mxl_train_mean_lmk_coords.npy')).float().unsqueeze(0).unsqueeze(0).to(device)
mdb_lm_inds = np.load(r'NICE/template/mdb_lmk.npy')
mdb_mean_lmk_coords = torch.from_numpy(np.load(r'NICE/template/mdb_train_mean_lmk_coords.npy')).float().unsqueeze(0).unsqueeze(0).to(device)

TRAIN_FOLDER = r'NICE/surgerydataset/train_surgery_data'
TEST_FOLDER = r'NICE/surgerydataset/test_surgery_data'
VAL_FOLDER = r'NICE/surgerydataset/val_surgery_data'

train_dataset = SurgeryDataLoader(datapath=TRAIN_FOLDER,
                                            n_supervision_points=CFG['training']['npoints_decoder'],
                                            batch_size=CFG['training']['batch_size'],
                                            sft_lm_inds=sft_lm_inds,
                                            mxl_lm_inds=mxl_lm_inds,
                                            mdb_lm_inds=mdb_lm_inds
                                        )


val_dataset = SurgeryDataLoader(datapath=VAL_FOLDER,
                                            n_supervision_points=CFG['training']['npoints_decoder'],
                                            batch_size=CFG['training']['batch_size'],
                                            sft_lm_inds=sft_lm_inds,
                                            mxl_lm_inds=mxl_lm_inds,
                                            mdb_lm_inds=mdb_lm_inds
                                        )

print('Lens of datasets right after creation')
print(len(train_dataset))
print(len(val_dataset))


decoder_surgery = NICE_surgery(
    mode=CFG['ex_decoder']['mode'],
    lat_dim_surgery=CFG['ex_decoder']['decoder_lat_dim_surgery'],
    lat_dim_id=CFG['ex_decoder']['decoder_lat_dim_id'],
    lat_dim_glob_shape=CFG['id_decoder']['decoder_lat_dim_glob'],
    
    sft_lat_dim_loc=CFG['id_decoder']['decoder_sft_lat_dim_loc'],
    sft_n_loc=CFG['id_decoder']['decoder_sft_nloc'],
    sft_landmarks=sft_mean_lmk_coords,
    sft_hidden_dim=CFG['ex_decoder']['decoder_sft_hidden_dim'],
    sft_n_layers=CFG['ex_decoder']['decoder_sft_nlayers'],
    
    mxl_lat_dim_loc=CFG['id_decoder']['decoder_mxl_lat_dim_loc'],
    mxl_n_loc=CFG['id_decoder']['decoder_mxl_nloc'],
    mxl_landmarks=mxl_mean_lmk_coords,
    mxl_hidden_dim=CFG['ex_decoder']['decoder_mxl_hidden_dim'],
    mxl_n_layers=CFG['ex_decoder']['decoder_mxl_nlayers'],
    
    mdb_lat_dim_loc=CFG['id_decoder']['decoder_mdb_lat_dim_loc'],
    mdb_n_loc=CFG['id_decoder']['decoder_mdb_nloc'],
    mdb_landmarks=mdb_mean_lmk_coords,
    mdb_hidden_dim=CFG['ex_decoder']['decoder_mdb_hidden_dim'],
    mdb_n_layers=CFG['ex_decoder']['decoder_mdb_nlayers'],
    
    out_dim=3,
    input_dim=3,
)



decoder_shape = NICE_shape(
    lat_dim_glob=CFG['id_decoder']['decoder_lat_dim_glob'],
    
    sft_lat_dim_loc=CFG['id_decoder']['decoder_sft_lat_dim_loc'],
    sft_hidden_dim=CFG['id_decoder']['decoder_sft_hidden_dim'],
    sft_n_loc=CFG['id_decoder']['decoder_sft_nloc'],
    sft_n_symm_pairs=CFG['id_decoder']['decoder_sft_nsymm_pairs'],
    sft_landmarks=sft_mean_lmk_coords,
    sft_n_layers=CFG['id_decoder']['decoder_sft_nlayers'],
    
    mxl_lat_dim_loc=CFG['id_decoder']['decoder_mxl_lat_dim_loc'],
    mxl_hidden_dim=CFG['id_decoder']['decoder_mxl_hidden_dim'],
    mxl_n_loc=CFG['id_decoder']['decoder_mxl_nloc'],
    mxl_n_symm_pairs=CFG['id_decoder']['decoder_mxl_nsymm_pairs'],
    mxl_landmarks=mxl_mean_lmk_coords,
    mxl_n_layers=CFG['id_decoder']['decoder_mxl_nlayers'],
    
    mdb_lat_dim_loc=CFG['id_decoder']['decoder_mdb_lat_dim_loc'],
    mdb_hidden_dim=CFG['id_decoder']['decoder_mdb_hidden_dim'],
    mdb_n_loc=CFG['id_decoder']['decoder_mdb_nloc'],
    mdb_n_symm_pairs=CFG['id_decoder']['decoder_mdb_nsymm_pairs'],
    mdb_landmarks=mdb_mean_lmk_coords,
    mdb_n_layers=CFG['id_decoder']['decoder_mdb_nlayers'],
    
    out_dim=1,
)



trainer = SurgeryTrainerAutoDecoder(decoder_surgery,
                                      decoder_shape,
                                      CFG,
                                      device,
                                      train_dataset,
                                      val_dataset,
                                      args.exp_name,
                                      ckpt=args.ckpt)
trainer.train_model(9000)

