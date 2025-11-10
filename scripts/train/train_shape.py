import os
import argparse
import torch
import json, os, yaml
import torch
import numpy as np
import sys
sys.path.append(r'NICE')
from dataset.dataloader import DataLoader
from src.models.NICE_shape import NICE_shape
from src.training.trainging_shape import TrainerFullHeadAutoDecoder
from src import env_paths
parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-cfg_file', type=str)
parser.add_argument('-closed', required=False, action='store_true')
parser.set_defaults(closed=False)
parser.add_argument('-local', required=False, action='store_true')
parser.set_defaults(local=False)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]
    
    
assert args.cfg_file is not None
CFG = yaml.safe_load(open(args.cfg_file, 'r'))


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

if args.local:

    sft_lm_inds = np.load(r'NICE/template/sft_lmk.npy')
    sft_mean_lmk_coords = torch.from_numpy(np.load(r'NICE/template/sft_train_mean_lmk_coords.npy')).float().unsqueeze(0).unsqueeze(0).to(device)
    mxl_lm_inds = np.load(r'NICE/template/mxl_lmk.npy')
    mxl_mean_lmk_coords = torch.from_numpy(np.load(r'NICE/template/mxl_train_mean_lmk_coords.npy')).float().unsqueeze(0).unsqueeze(0).to(device)
    mdb_lm_inds = np.load(r'NICE/template/mdb_lmk.npy')
    mdb_mean_lmk_coords = torch.from_numpy(np.load(r'NICE/template/mdb_train_mean_lmk_coords.npy')).float().unsqueeze(0).unsqueeze(0).to(device)

else:
    sft_lm_inds = None
    sft_mean_lmk_coords = None
    mxl_lm_inds = None
    mxl_mean_lmk_coords = None
    mdb_lm_inds = None
    mdb_mean_lmk_coords = None

TRAIN_FOLDER = r'NICE/dataset/train_data'
TEST_FOLDER = r'NICE/dataset/test_data'
VAL_FOLDER = r'NICE/dataset/val_data'

train_dataset = DataLoader(datapath=TRAIN_FOLDER,
                            n_supervision_points_face=CFG['training']['npoints_decoder'],
                            n_supervision_points_non_face=CFG['training']['npoints_decoder_non'],
                            batch_size=CFG['training']['batch_size'],
                            sigma_near=CFG['training']['sigma_near'],
                            sft_lm_inds=sft_lm_inds,
                            mxl_lm_inds=mxl_lm_inds,
                            mdb_lm_inds=mdb_lm_inds)

val_dataset = DataLoader(datapath=VAL_FOLDER,
                            n_supervision_points_face=CFG['training']['npoints_decoder'],
                            n_supervision_points_non_face=CFG['training']['npoints_decoder_non'],
                            batch_size=CFG['training']['batch_size'],
                            sigma_near=CFG['training']['sigma_near'],
                            sft_lm_inds=sft_lm_inds,
                            mxl_lm_inds=mxl_lm_inds,
                            mdb_lm_inds=mdb_lm_inds)

train_data_loader = train_dataset.get_loader()

print('Done creating datasets!')

print('Length of Train Dataset: {}'.format(len(train_dataset)))
print('Length of Val Dataset: {}'.format(len(val_dataset)))



decoder = NICE_shape(
    lat_dim_glob=CFG['decoder']['decoder_lat_dim_glob'],
    
    sft_lat_dim_loc=CFG['decoder']['decoder_sft_lat_dim_loc'],
    sft_hidden_dim=CFG['decoder']['decoder_sft_hidden_dim'],
    sft_n_loc=CFG['decoder']['decoder_sft_nloc'],
    sft_n_symm_pairs=CFG['decoder']['decoder_sft_nsymm_pairs'],
    sft_landmarks=sft_mean_lmk_coords,
    sft_n_layers=CFG['decoder']['decoder_sft_nlayers'],
    
    mxl_lat_dim_loc=CFG['decoder']['decoder_mxl_lat_dim_loc'],
    mxl_hidden_dim=CFG['decoder']['decoder_mxl_hidden_dim'],
    mxl_n_loc=CFG['decoder']['decoder_mxl_nloc'],
    mxl_n_symm_pairs=CFG['decoder']['decoder_mxl_nsymm_pairs'],
    mxl_landmarks=mxl_mean_lmk_coords,
    mxl_n_layers=CFG['decoder']['decoder_mxl_nlayers'],
    
    mdb_lat_dim_loc=CFG['decoder']['decoder_mdb_lat_dim_loc'],
    mdb_hidden_dim=CFG['decoder']['decoder_mdb_hidden_dim'],
    mdb_n_loc=CFG['decoder']['decoder_mdb_nloc'],
    mdb_n_symm_pairs=CFG['decoder']['decoder_mdb_nsymm_pairs'],
    mdb_landmarks=mdb_mean_lmk_coords,
    mdb_n_layers=CFG['decoder']['decoder_mdb_nlayers'],
    
    out_dim=1,
)

    

decoder = decoder.to(device)

trainer = TrainerFullHeadAutoDecoder(decoder, CFG, device, train_dataset, val_dataset, args.exp_name)


if 'nepochs' in CFG['training']:
    trainer.train_model(CFG['training']['nepochs'])
else:
    trainer.train_model(30001)

