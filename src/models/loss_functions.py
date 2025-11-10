from src.models.diff_operators  import gradient
import torch

def compute_sft_loss(batch, decoder, latent_codes, device):
    if 'path' in batch:
        del batch['path']

    batch_cuda_nice = {k: v.to(device).float() for (k, v) in zip(batch.keys(), batch.values())}

    idx = batch.get('idx').to(device)
    glob_cond = latent_codes(idx)
    loss_dict = actual_compute_sft_loss(batch_cuda_nice, decoder, glob_cond)

    return loss_dict


def actual_compute_sft_loss(batch_cuda, decoder, glob_cond):

    if hasattr(decoder, 'landmarks'):
        sft_lmks_preds = batch_cuda['gt_sft_lmk']
    else:
        sft_lmks_preds = None


    # prep
    sup_sft_surface = batch_cuda['sup_sft_on_face_points'].clone().detach().requires_grad_() # points on face surf
    sup_sft_surface_outer = batch_cuda['sup_sft_non_face_points'].clone().detach().requires_grad_() # points on non-face surf
    sup_sft_grad_far = batch_cuda['sup_sft_grad_far'].clone().detach().requires_grad_() # points in unifrm ball
    sup_sft_grad_near = batch_cuda['sup_sft_grad_near'].clone().detach().requires_grad_() # points near/off surface



    # model computations
    pred_sft_surface, landmarks = decoder(sup_sft_surface, glob_cond.repeat(1, sup_sft_surface.shape[1], 1), sft_lmks_preds)
    pred_sft_surface_outer, landmarks = decoder(sup_sft_surface_outer, glob_cond.repeat(1, sup_sft_surface_outer.shape[1], 1),
                                          sft_lmks_preds)
    pred_sft_surface_near, landmarks = decoder(sup_sft_grad_near, glob_cond.repeat(1, sup_sft_grad_near.shape[1], 1), sft_lmks_preds)


    pred_sft_surface_far = decoder(sup_sft_grad_far, glob_cond.repeat(1, sup_sft_grad_far.shape[1], 1), sft_lmks_preds)[0]


    # normal computation
    gradient_sft_surface = gradient(pred_sft_surface, sup_sft_surface)
    gradient_sft_surface_outer = gradient(pred_sft_surface_outer, sup_sft_surface_outer)
    gradient_sft_surface_far = gradient(pred_sft_surface_far, sup_sft_grad_far)
    gradient_sft_surface_near = gradient(pred_sft_surface_near, sup_sft_grad_near)



    # computation of losses for geometry
    sft_surf_sdf_loss = torch.abs(pred_sft_surface).squeeze()
    sft_surf_sdf_loss_outer = torch.abs(pred_sft_surface_outer).squeeze()

    sft_surf_normal_loss = (gradient_sft_surface - batch_cuda['sup_sft_on_face_normals']).norm(2, dim=-1)
    sft_surf_normal_loss_outer = torch.clamp((gradient_sft_surface_outer - batch_cuda['sup_sft_non_face_normals']).norm(2, dim=-1),
                                         None, 0.75) / 2

    sft_surf_grad_loss = torch.abs(gradient_sft_surface.norm(dim=-1) - 1)
    sft_surf_grad_loss_outer = torch.abs(gradient_sft_surface_outer.norm(dim=-1) - 1)

    sft_space_sdf_loss = torch.exp(-1e1 * torch.abs(pred_sft_surface_far))
    sft_space_grad_loss_far = torch.abs(gradient_sft_surface_far.norm(dim=-1) - 1)
    sft_space_grad_loss_near = torch.abs(gradient_sft_surface_near.norm(dim=-1) - 1)

    grad_loss = torch.cat([sft_surf_grad_loss, sft_surf_grad_loss_outer, sft_space_grad_loss_far, sft_space_grad_loss_near], dim=-1)


    lat_mag = torch.norm(glob_cond, dim=-1) ** 2
    glob_cond = glob_cond.squeeze(1)

    if hasattr(decoder, 'lat_dim_glob'):
    # symm lats
        if decoder.num_symm_pairs > 0:
            loc_lats_symm = glob_cond[:, 
                decoder.lat_dim_glob : decoder.lat_dim_glob + 2 * decoder.num_symm_pairs * decoder.lat_dim_loc
            ].view(
                glob_cond.shape[0], decoder.num_symm_pairs * 2, decoder.lat_dim_loc
            )
            symm_dist = torch.norm(
                loc_lats_symm[:, ::2, :] - loc_lats_symm[:, 1::2, :], dim=-1
            ).mean()
        else:
            symm_dist = torch.tensor(0.0, device=glob_cond.device)

        # middle lats
        loc_lats_middle = glob_cond[:, 
            decoder.lat_dim_glob + 2 * decoder.num_symm_pairs * decoder.lat_dim_loc : -decoder.lat_dim_loc
        ].view(
            glob_cond.shape[0], decoder.num_kps - decoder.num_symm_pairs * 2, decoder.lat_dim_loc
        )

        if loc_lats_middle.shape[1] % 2 == 0:
            middle_dist = torch.norm(
                loc_lats_middle[:, ::2, :] - loc_lats_middle[:, 1::2, :], dim=-1
            ).mean()
        else:
            middle_dist = torch.norm(
                loc_lats_middle[:, :-1:2, :] - loc_lats_middle[:, 1::2, :], dim=-1
            ).mean()
    else:
        symm_dist = None
        middle_dist = None


    if landmarks is not None:
        sft_loss_landmarks = (landmarks - batch_cuda['gt_sft_lmk']).square().mean()

        ret_dict =  {'sft_surf_sdf': torch.mean(torch.cat([sft_surf_sdf_loss, sft_surf_sdf_loss_outer], dim=-1)),
                'sft_normals': torch.mean(
                    torch.cat([sft_surf_normal_loss.squeeze(), sft_surf_normal_loss_outer.squeeze()], dim=-1)),
                'sft_space_sdf': torch.mean(sft_space_sdf_loss),
                'grad': torch.mean(grad_loss),
                'lat_reg': lat_mag.mean(),
                'sft_landmarks': sft_loss_landmarks,
                'symm_dist': symm_dist,
                'middle_dist': middle_dist, }
        return ret_dict
    else:
        ret_dict =  {'sft_surf_sdf': torch.mean(torch.cat([sft_surf_sdf_loss, sft_surf_sdf_loss_outer], dim=-1)),
                'sft_normals': torch.mean(
                    torch.cat([sft_surf_normal_loss.squeeze(), sft_surf_normal_loss_outer.squeeze()], dim=-1)),
                'sft_space_sdf': torch.mean(sft_space_sdf_loss),
                'grad': torch.mean(grad_loss),
                'lat_reg': lat_mag.mean()}
        return ret_dict


def compute_all_loss(batch, decoder, latent_codes, device):
    if 'path' in batch:
        del batch['path']

    batch_cuda_nice = {k: v.to(device).float() for (k, v) in zip(batch.keys(), batch.values())}

    idx = batch.get('idx').to(device)
    glob_cond = latent_codes(idx)
    loss_dict, latent_code_dict = actual_compute_all_loss(batch_cuda_nice, decoder, glob_cond)

    return loss_dict, latent_code_dict



def actual_compute_all_loss(batch_cuda, decoder, glob_cond):

    # ----------------------------------------- below is for sft ---------------------------------------
    # prep
    sup_sft_surface = batch_cuda['sup_sft_on_face_points'].clone().detach().requires_grad_() # points on face surf
    sup_sft_surface_outer = batch_cuda['sup_sft_non_face_points'].clone().detach().requires_grad_() # points on non-face surf
    sup_sft_grad_far = batch_cuda['sup_sft_grad_far'].clone().detach().requires_grad_() # points in unifrm ball
    sup_sft_grad_near = batch_cuda['sup_sft_grad_near'].clone().detach().requires_grad_() # points near/off surface



    # model computations
    pred_sft_surface, sft_landmarks, sft_latent_code = decoder(sup_sft_surface, glob_cond.repeat(1, sup_sft_surface.shape[1], 1), 'sft')
    pred_sft_surface_outer, sft_landmarks, _ = decoder(sup_sft_surface_outer, glob_cond.repeat(1, sup_sft_surface_outer.shape[1], 1),
                                          'sft')
    pred_sft_surface_near, sft_landmarks, _ = decoder(sup_sft_grad_near, glob_cond.repeat(1, sup_sft_grad_near.shape[1], 1), 'sft')


    pred_sft_surface_far = decoder(sup_sft_grad_far, glob_cond.repeat(1, sup_sft_grad_far.shape[1], 1), 'sft')[0]


    # normal computation
    gradient_sft_surface = gradient(pred_sft_surface, sup_sft_surface)
    gradient_sft_surface_outer = gradient(pred_sft_surface_outer, sup_sft_surface_outer)
    gradient_sft_surface_far = gradient(pred_sft_surface_far, sup_sft_grad_far)
    gradient_sft_surface_near = gradient(pred_sft_surface_near, sup_sft_grad_near)



    # computation of losses for geometry
    sft_surf_sdf_loss = torch.abs(pred_sft_surface).squeeze()
    sft_surf_sdf_loss_outer = torch.abs(pred_sft_surface_outer).squeeze()

    sft_surf_normal_loss = (gradient_sft_surface - batch_cuda['sup_sft_on_face_normals']).norm(2, dim=-1)
    sft_surf_normal_loss_outer = torch.clamp((gradient_sft_surface_outer - batch_cuda['sup_sft_non_face_normals']).norm(2, dim=-1),
                                         None, 0.75) / 2

    sft_surf_grad_loss = torch.abs(gradient_sft_surface.norm(dim=-1) - 1)
    sft_surf_grad_loss_outer = torch.abs(gradient_sft_surface_outer.norm(dim=-1) - 1)

    sft_space_sdf_loss = torch.exp(-1e1 * torch.abs(pred_sft_surface_far))
    sft_space_grad_loss_far = torch.abs(gradient_sft_surface_far.norm(dim=-1) - 1)
    sft_space_grad_loss_near = torch.abs(gradient_sft_surface_near.norm(dim=-1) - 1)

    sft_grad_loss = torch.cat([sft_surf_grad_loss, sft_surf_grad_loss_outer, sft_space_grad_loss_far, sft_space_grad_loss_near], dim=-1)


    # ----------------------------------------- below is for mxl ---------------------------------------
    sup_mxl_surface = batch_cuda['sup_mxl_on_face_points'].clone().detach().requires_grad_() # points on face surf
    sup_mxl_surface_outer = batch_cuda['sup_mxl_non_face_points'].clone().detach().requires_grad_() # points on non-face surf
    sup_mxl_grad_far = batch_cuda['sup_mxl_grad_far'].clone().detach().requires_grad_() # points in unifrm ball
    sup_mxl_grad_near = batch_cuda['sup_mxl_grad_near'].clone().detach().requires_grad_() # points near/off surface

    # model computations
    pred_mxl_surface, mxl_landmarks, mxl_latent_code = decoder(sup_mxl_surface, glob_cond.repeat(1, sup_mxl_surface.shape[1], 1), 'mxl')
    pred_mxl_surface_outer, mxl_landmarks, _ = decoder(sup_mxl_surface_outer, glob_cond.repeat(1, sup_mxl_surface_outer.shape[1], 1),
                                          'mxl')
    pred_mxl_surface_near, mxl_landmarks, _ = decoder(sup_mxl_grad_near, glob_cond.repeat(1, sup_mxl_grad_near.shape[1], 1), 'mxl')


    pred_mxl_surface_far = decoder(sup_mxl_grad_far, glob_cond.repeat(1, sup_mxl_grad_far.shape[1], 1), 'mxl')[0]

    # normal computation
    gradient_mxl_surface = gradient(pred_mxl_surface, sup_mxl_surface)
    gradient_mxl_surface_outer = gradient(pred_mxl_surface_outer, sup_mxl_surface_outer)
    gradient_mxl_surface_far = gradient(pred_mxl_surface_far, sup_mxl_grad_far)
    gradient_mxl_surface_near = gradient(pred_mxl_surface_near, sup_mxl_grad_near)


    # computation of losses for geometry
    mxl_surf_sdf_loss = torch.abs(pred_mxl_surface).squeeze()
    mxl_surf_sdf_loss_outer = torch.abs(pred_mxl_surface_outer).squeeze()

    mxl_surf_normal_loss = (gradient_mxl_surface - batch_cuda['sup_mxl_on_face_normals']).norm(2, dim=-1)
    mxl_surf_normal_loss_outer = torch.clamp((gradient_mxl_surface_outer - batch_cuda['sup_mxl_non_face_normals']).norm(2, dim=-1),
                                         None, 0.75) / 2

    mxl_surf_grad_loss = torch.abs(gradient_mxl_surface.norm(dim=-1) - 1)
    mxl_surf_grad_loss_outer = torch.abs(gradient_mxl_surface_outer.norm(dim=-1) - 1)

    mxl_space_sdf_loss = torch.exp(-1e1 * torch.abs(pred_mxl_surface_far))
    mxl_space_grad_loss_far = torch.abs(gradient_mxl_surface_far.norm(dim=-1) - 1)
    mxl_space_grad_loss_near = torch.abs(gradient_mxl_surface_near.norm(dim=-1) - 1)

    mxl_grad_loss = torch.cat([mxl_surf_grad_loss, mxl_surf_grad_loss_outer, mxl_space_grad_loss_far, mxl_space_grad_loss_near], dim=-1)


    # ----------------------------------------- below is for mdb ---------------------------------------
    sup_mdb_surface = batch_cuda['sup_mdb_on_face_points'].clone().detach().requires_grad_() # points on face surf
    sup_mdb_grad_far = batch_cuda['sup_mdb_grad_far'].clone().detach().requires_grad_() # points in unifrm ball
    sup_mdb_grad_near = batch_cuda['sup_mdb_grad_near'].clone().detach().requires_grad_() # points near/off surface


    # model computations
    pred_mdb_surface, mdb_landmarks, mdb_latent_code = decoder(sup_mdb_surface, glob_cond.repeat(1, sup_mdb_surface.shape[1], 1), 'mdb')
    pred_mdb_surface_near, mdb_landmarks, _ = decoder(sup_mdb_grad_near, glob_cond.repeat(1, sup_mdb_grad_near.shape[1], 1), 'mdb')
    pred_mdb_surface_far = decoder(sup_mdb_grad_far, glob_cond.repeat(1, sup_mdb_grad_far.shape[1], 1), 'mdb')[0]


    # normal computation
    gradient_mdb_surface = gradient(pred_mdb_surface, sup_mdb_surface)
    gradient_mdb_surface_far = gradient(pred_mdb_surface_far, sup_mdb_grad_far)
    gradient_mdb_surface_near = gradient(pred_mdb_surface_near, sup_mdb_grad_near)


    # computation of losses for geometry
    mdb_surf_sdf_loss = torch.abs(pred_mdb_surface).squeeze()

    mdb_surf_normal_loss = (gradient_mdb_surface - batch_cuda['sup_mdb_on_face_normals']).norm(2, dim=-1)

    mdb_surf_grad_loss = torch.abs(gradient_mdb_surface.norm(dim=-1) - 1)

    mdb_space_sdf_loss = torch.exp(-1e1 * torch.abs(pred_mdb_surface_far))
    mdb_space_grad_loss_far = torch.abs(gradient_mdb_surface_far.norm(dim=-1) - 1)
    mdb_space_grad_loss_near = torch.abs(gradient_mdb_surface_near.norm(dim=-1) - 1)

    mdb_grad_loss = torch.cat([mdb_surf_grad_loss, mdb_space_grad_loss_far, mdb_space_grad_loss_near], dim=-1)



    # ----------------------------------------- below is for global ---------------------------------------
    lat_mag = torch.norm(glob_cond, dim=-1) ** 2
    glob_cond = glob_cond.squeeze(1)
    
    
    # -------------------------------- below is for sft/mxl/mdb landmarks loss ---------------------------------------
    sft_loss_landmarks = (sft_landmarks - batch_cuda['gt_sft_lmk']).square().mean()
    mxl_loss_landmarks = (mxl_landmarks - batch_cuda['gt_mxl_lmk']).square().mean()
    mdb_loss_landmarks = (mdb_landmarks - batch_cuda['gt_mdb_lmk']).square().mean()

    sft_latent_code = sft_latent_code[:, :, 0, :]  
    sft_latent_code = sft_latent_code.permute(1, 0, 2)
    mxl_latent_code = mxl_latent_code[:, :, 0, :]  
    mxl_latent_code = mxl_latent_code.permute(1, 0, 2)
    mdb_latent_code = mdb_latent_code[:, :, 0, :]  
    mdb_latent_code = mdb_latent_code.permute(1, 0, 2)
    

    latent_code_dict = {
        'sft_latent_code': sft_latent_code,
        'mxl_latent_code': mxl_latent_code,
        'mdb_latent_code': mdb_latent_code,
    }
    

    ret_dict =  {
            'lat_reg': lat_mag.mean(),
            
            'sft_surf_sdf': torch.mean(torch.cat([sft_surf_sdf_loss, sft_surf_sdf_loss_outer], dim=-1)),
            'sft_normals': torch.mean(
                torch.cat([sft_surf_normal_loss.squeeze(), sft_surf_normal_loss_outer.squeeze()], dim=-1)),
            'sft_space_sdf': torch.mean(sft_space_sdf_loss),
            'sft_grad': torch.mean(sft_grad_loss),
            'sft_landmarks': sft_loss_landmarks,
            
            
            'mxl_surf_sdf': torch.mean(torch.cat([mxl_surf_sdf_loss, mxl_surf_sdf_loss_outer], dim=-1)),
            'mxl_normals': torch.mean(
                torch.cat([mxl_surf_normal_loss.squeeze(), mxl_surf_normal_loss_outer.squeeze()], dim=-1)),
            'mxl_space_sdf': torch.mean(mxl_space_sdf_loss),
            'mxl_grad': torch.mean(mxl_grad_loss),
            'mxl_landmarks': mxl_loss_landmarks,
            
            
            'mdb_surf_sdf': torch.mean(mdb_surf_sdf_loss),
            'mdb_normals': torch.mean(mdb_surf_normal_loss),
            'mdb_space_sdf': torch.mean(mdb_space_sdf_loss),
            'mdb_grad': torch.mean(mdb_grad_loss),
            'mdb_landmarks': mdb_loss_landmarks,
            
            
            
            }
    return ret_dict, latent_code_dict
    



def compute_all_loss_corresp_forward(batch, decoder, decoder_shape, latent_codes, latent_codes_shape, device, epoch=-1, exp_path=None):

    if 'path' in batch:
        del batch['path']
    batch_cuda = {k: v.to(device).float() for (k, v) in zip(batch.keys(), batch.values())}
    
    
    glob_cond_shape = batch['shape_latent_code'].to(device)
    glob_cond_shape = glob_cond_shape.unsqueeze(1)

    glob_cond_pose = latent_codes(batch['idx'].to(device))


    lat_mag = torch.norm(glob_cond_pose, dim=-1)**2

    gt_sft_landmarks = batch_cuda['gt_sft_lmk']
    gt_mxl_landmarks = batch_cuda['gt_mxl_lmk']
    gt_mdb_landmarks = batch_cuda['gt_mdb_lmk']

    # below is for sft
    sft_glob_cond = torch.cat([glob_cond_shape, glob_cond_pose], dim=-1)
    sft_points_pre = batch_cuda['sft_points_pre'].clone().detach().requires_grad_()
    sft_cond = sft_glob_cond.repeat(1, sft_points_pre.shape[1], 1)

    sft_delta, _ = decoder(sft_points_pre, sft_cond, gt_sft_landmarks, 'sft')
    sft_pred_post = sft_points_pre + sft_delta.squeeze()
    sft_points_post = batch_cuda['sft_points_post']
    
    sft_loss_corresp = (sft_pred_post - sft_points_post[:, :, :3])**2
    
    # enforce deformation field to be zero elsewhere
    sft_samps = (torch.rand(sft_cond.shape[0], 100, 3, device=sft_cond.device, dtype=sft_cond.dtype) -0.5)*2.5

    sft_samps_delta, _ = decoder(sft_samps, sft_cond[:, :100, :], gt_sft_landmarks, 'sft')
    
    sft_loss_reg_zero = (sft_samps_delta**2).mean()
    # -----------------------------------------------------
    
    # below is for mxl
    mxl_glob_cond = torch.cat([glob_cond_shape, glob_cond_pose], dim=-1)
    mxl_points_pre = batch_cuda['mxl_points_pre'].clone().detach().requires_grad_()
    mxl_cond = mxl_glob_cond.repeat(1, mxl_points_pre.shape[1], 1)
    mxl_delta, _ = decoder(mxl_points_pre, mxl_cond, gt_mxl_landmarks, 'mxl')
    mxl_pred_post = mxl_points_pre + mxl_delta.squeeze()
    mxl_points_post = batch_cuda['mxl_points_post']
    
    mxl_loss_corresp = (mxl_pred_post - mxl_points_post[:, :, :3])**2#.abs()
    
    # enforce deformation field to be zero elsewhere
    mxl_samps = (torch.rand(mxl_cond.shape[0], 100, 3, device=mxl_cond.device, dtype=mxl_cond.dtype) -0.5)*2.5

    mxl_samps_delta, _ = decoder(mxl_samps, mxl_cond[:, :100, :], gt_mxl_landmarks, 'mxl')
    
    mxl_loss_reg_zero = (mxl_samps_delta**2).mean()
    
    # -----------------------------------------------------
    
    # below is for mdb
    mdb_glob_cond = torch.cat([glob_cond_shape, glob_cond_pose], dim=-1)
    mdb_points_pre = batch_cuda['mdb_points_pre'].clone().detach().requires_grad_()
    mdb_cond = mdb_glob_cond.repeat(1, mdb_points_pre.shape[1], 1)
    mdb_delta, _ = decoder(mdb_points_pre, mdb_cond, gt_mdb_landmarks, 'mdb')
    mdb_pred_post = mdb_points_pre + mdb_delta.squeeze()
    mdb_points_post = batch_cuda['mdb_points_post']
    
    mdb_loss_corresp = (mdb_pred_post - mdb_points_post[:, :, :3])**2
    
    # enforce deformation field to be zero elsewhere
    mdb_samps = (torch.rand(mdb_cond.shape[0], 100, 3, device=mdb_cond.device, dtype=mdb_cond.dtype) -0.5)*2.5

    mdb_samps_delta, _ = decoder(mdb_samps, mdb_cond[:, :100, :], gt_mdb_landmarks, 'mdb')
    
    mdb_loss_reg_zero = (mdb_samps_delta**2).mean()
    

    


    return { 'sft_loss_corresp': sft_loss_corresp.mean(),
            'mxl_loss_corresp': mxl_loss_corresp.mean(),
            'mdb_loss_corresp': mdb_loss_corresp.mean(), 

            'lat_reg': lat_mag.mean(),
            'sft_loss_reg_zero': sft_loss_reg_zero,
            'mxl_loss_reg_zero': mxl_loss_reg_zero,
            'mdb_loss_reg_zero': mdb_loss_reg_zero,
            }



def compute_skull_loss_corresp_forward(batch, decoder, decoder_shape, latent_codes, latent_codes_shape, device, epoch=-1, exp_path=None):

    if 'path' in batch:
        del batch['path']
    batch_cuda = {k: v.to(device).float() for (k, v) in zip(batch.keys(), batch.values())}
    
    
    glob_cond_shape = batch['shape_latent_code'].to(device)
    glob_cond_shape = glob_cond_shape.unsqueeze(1)
    
    glob_cond_pose = latent_codes(batch['idx'].to(device))

    lat_mag = torch.norm(glob_cond_pose, dim=-1)**2
        
        
    gt_mxl_landmarks = batch_cuda['gt_mxl_lmk']
    gt_mdb_landmarks = batch_cuda['gt_mdb_lmk']
    # -----------------------------------------------------
    
    # below is for mxl
    mxl_glob_cond = torch.cat([glob_cond_shape, glob_cond_pose], dim=-1)
    mxl_points_pre = batch_cuda['mxl_points_pre'].clone().detach().requires_grad_()
    mxl_cond = mxl_glob_cond.repeat(1, mxl_points_pre.shape[1], 1)
    mxl_delta, _ = decoder(mxl_points_pre, mxl_cond, gt_mxl_landmarks, 'mxl')
    mxl_pred_plan = mxl_points_pre + mxl_delta.squeeze()
    mxl_points_post = batch_cuda['mxl_points_post']
    
    mxl_loss_corresp = (mxl_pred_plan - mxl_points_post[:, :, :3])**2
    
    # enforce deformation field to be zero elsewhere
    mxl_samps = (torch.rand(mxl_cond.shape[0], 100, 3, device=mxl_cond.device, dtype=mxl_cond.dtype) -0.5)*2.5

    mxl_samps_delta, _ = decoder(mxl_samps, mxl_cond[:, :100, :], gt_mxl_landmarks, 'mxl')
    
    mxl_loss_reg_zero = (mxl_samps_delta**2).mean()
    
    # -----------------------------------------------------
    
    # below is for mdb
    mdb_glob_cond = torch.cat([glob_cond_shape, glob_cond_pose], dim=-1)
    mdb_points_pre = batch_cuda['mdb_points_pre'].clone().detach().requires_grad_()
    mdb_cond = mdb_glob_cond.repeat(1, mdb_points_pre.shape[1], 1)
    mdb_delta, _ = decoder(mdb_points_pre, mdb_cond, gt_mdb_landmarks, 'mdb')
    mdb_pred_plan = mdb_points_pre + mdb_delta.squeeze()
    mdb_points_plan = batch_cuda['mdb_points_post']
    
    mdb_loss_corresp = (mdb_pred_plan - mdb_points_plan[:, :, :3])**2
    
    # enforce deformation field to be zero elsewhere
    mdb_samps = (torch.rand(mdb_cond.shape[0], 100, 3, device=mdb_cond.device, dtype=mdb_cond.dtype) -0.5)*2.5

    mdb_samps_delta, _ = decoder(mdb_samps, mdb_cond[:, :100, :], gt_mdb_landmarks, 'mdb')
    
    mdb_loss_reg_zero = (mdb_samps_delta**2).mean()
    

    


    return { 
            'mxl_loss_corresp': mxl_loss_corresp.mean(),
            'mdb_loss_corresp': mdb_loss_corresp.mean(),

            'lat_reg': lat_mag.mean(),
            'mxl_loss_reg_zero': mxl_loss_reg_zero,
            'mdb_loss_reg_zero': mdb_loss_reg_zero,
            }
