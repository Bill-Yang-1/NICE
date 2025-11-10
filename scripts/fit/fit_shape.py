import sys
import argparse
import torch
import json, yaml
import torch
import numpy as np
sys.path.append(r'NICE')
from dataset.dataloader import DataLoader
from src.models.NICE_shape import NICE_shape
from src.testing.testing_shape import TesterAutoDecoder
parser = argparse.ArgumentParser(description='Test Model')


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

print(json.dumps(CFG, sort_keys=True, indent=4))

device = torch.device("cuda")

output_fd = r'PATH2OUTPUT'
TEST_FOLDER =  r'NICE/dataset/test_data'

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


test_dataset = DataLoader(datapath=TEST_FOLDER,
                            n_supervision_points_face=CFG['training']['npoints_decoder'],
                            n_supervision_points_non_face=CFG['training']['npoints_decoder_non'],
                            batch_size=CFG['training']['batch_size'],
                            sigma_near=CFG['training']['sigma_near'],
                            sft_lm_inds=sft_lm_inds,
                            mxl_lm_inds=mxl_lm_inds,
                            mdb_lm_inds=mdb_lm_inds)


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
 
tester = TesterAutoDecoder(decoder, CFG, device, test_dataset, output_fd)

tester.test_model(241)

