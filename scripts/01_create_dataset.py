import os
import sys
sys.path.insert(0, '/'.join(os.getcwd().split('/')[:-1]))

from qpi_deep_cwfs import utils
from qpi_deep_cwfs import hRAFT

from optical_volume import visualization
from optical_volume import torch_cwfs as TCWFS
from optical_volume import utils as ov_utils

import numpy as np
from tifffile import tifffile
from matplotlib import pyplot as plt

import torch
import torchvision
import torchvision.transforms.functional as F

import json
import argparse
from datetime import datetime
from typing import Dict, Any
from torch import Tensor
import csv


def norm(img, scale):
    return np.array(img*scale, dtype=np.uint16)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--exp_name", type=str, help='data/exp_name', required=True)
    parser.add_argument("--data_params", type=str, help='path to config file: configs/data_params', required=True)
    
    args = parser.parse_args()
    
    date = f'{datetime.today()}'.split()[0]
    EXP_NAME = f'{date}_{args.exp_name}'
    
    cwfs_path = f'../configs/{args.data_params}.json'
    exp_path = f'../data/{EXP_NAME}'
    
    assert os.path.isdir(exp_path) == False, "Experiment already exists."
    os.makedirs(exp_path)
    os.makedirs(f'{exp_path}/amplitude') # img_ids.tiff
    os.makedirs(f'{exp_path}/reference') # img_ids.tiff
    os.makedirs(f'{exp_path}/specimen') # img_ids.tiff
    os.makedirs(f'{exp_path}/gt_gradients') # img_ids_0.tiff, img_ids_1.tiff
    
    with open(f'{exp_path}/dataset_details.csv','w') as f:
        w = csv.writer(f)
    
    if os.path.exists(cwfs_path):
        with open(cwfs_path, 'r') as file:
            cwfs_params = json.load(file)
        
        with open(os.path.join(exp_path, 'data_creation_cwfs_params.json'), 'w') as file:
            json.dump(cwfs_params, file)
    else:
        print(f'{cwfs_path} does not exist.')
        raise
    
    SYS_PARAMS = cwfs_params['system_params']
    SPEC_PARAMS = cwfs_params['specimen_params']
    OPT_PARAMS = cwfs_params['optics_params']
    PM_PARAMS = cwfs_params['phase_mask_params']
    DATA_SAV_PARAMS = cwfs_params['data_saving_params']
    
    torch.manual_seed(DATA_SAV_PARAMS['SEED'])
    
    ## CWFS SETUP
    # Grid and propagation parameters setup
    WL = SYS_PARAMS['WL']
    SPATIAL_RES = SYS_PARAMS['SPATIAL_RES'] # dx, dy, dz
    GRID_SHAPE = SYS_PARAMS['GRID_SHAPE'] # x=0->, y=0->, z=0->
    N_BG = SYS_PARAMS['N_BG'] # immersion medium RI
    DIGITAL_PX_SIZE = torch.tensor(SYS_PARAMS['DIGITAL_PX_SIZE'])
    PAD = SYS_PARAMS['PAD']
    PAD_MODE = SYS_PARAMS['PAD_MODE']
    spatial_support = [SPATIAL_RES[i]*GRID_SHAPE[i] for i in range(3)]

    # specimen setup: ellipsoids and cubes
    NUM_ELEMENTS_ELLIPS = SPEC_PARAMS['NUM_ELEMENTS_ELLIPS']
    CENTER_SIGMA_ELLIPS = torch.tensor(SPEC_PARAMS['CENTER_SIGMA_ELLIPS'])
    RADII_RANGE_ELLIPS = torch.tensor(SPEC_PARAMS['RADII_RANGE_ELLIPS'])
    RI_SIGMA_ELLIPS = SPEC_PARAMS['RI_SIGMA_ELLIPS']
    NUM_ELEMENTS_CUBES = SPEC_PARAMS['NUM_ELEMENTS_CUBES']
    CENTER_SIGMA_CUBES = torch.tensor(SPEC_PARAMS['CENTER_SIGMA_CUBES'])
    RADII_RANGE_CUBES = torch.tensor(SPEC_PARAMS['RADII_RANGE_CUBES'])
    RI_SIGMA_CUBES = SPEC_PARAMS['RI_SIGMA_CUBES']
    GRAD_MEDIAN_KERNEL = SPEC_PARAMS['GRAD_MEDIAN_KERNEL']
    radii_min = SPATIAL_RES[0]*4 # the smallest aerial pixel in the image space

    # optics setup
    MAG = OPT_PARAMS['MAG']
    PSC = torch.tensor(OPT_PARAMS["PSC"])
    FOCUS_PLANE_VAR = OPT_PARAMS['FOCUS_PLANE_VAR_DIST']
    NA = OPT_PARAMS['NA_RANGE']

    # phase mask setup
    PM_PROB = PM_PARAMS['PM_PROB']
    PM_HEIGHT = PM_PARAMS['PM_HEIGHT_RANGE']
    PM_SIDE_LEN = torch.tensor(PM_PARAMS['PM_SIDE_LEN'])
    PM_RI = PM_PARAMS['PM_RI']
    PM_SMOOTHING = PM_PARAMS['PM_SMOOTHING_RANGE']
    DIST_M_IM = PM_PARAMS['DIST_M_IM_RANGE']
    
    # data saving setup
    DATA_GRAD_BIAS = DATA_SAV_PARAMS['GRAD_DATA_BIAS']
    DATA_GRAD_SCALE = DATA_SAV_PARAMS['GRAD_DATA_SCALE']
    DATA_SPECKLE_SCALE = DATA_SAV_PARAMS['SPECKLE_DATA_SCALE']
    DATA_AMP_SCALE = DATA_SAV_PARAMS['AMP_DATA_SCALE']
    DATASET_SIZE = DATA_SAV_PARAMS['TOTAL_DATA']
    
    
    cwfs_obj = TCWFS.CodedWFSForwardModel(WL, GRID_SHAPE, SPATIAL_RES, N_BG, digital_px_size=DIGITAL_PX_SIZE, padding=PAD, pad_mode=PAD_MODE)
    cwfs_obj.eff_mag_operator(MAG)

    PM_grid_shape = [int(GRID_SHAPE[0]*MAG/cwfs_obj.im_to_ob_space_scale), int(GRID_SHAPE[1]*MAG/cwfs_obj.im_to_ob_space_scale)]
    phase_mask_obj = TCWFS.PhaseMask(PM_SIDE_LEN, cwfs_obj.im_space_res, PM_grid_shape)

    reszd_nx_factor = int(MAG/cwfs_obj.im_to_ob_space_scale)
    lens = ov_utils.ObjImgMap(WL, [x/reszd_nx_factor for x in SPATIAL_RES[:2]], [x*reszd_nx_factor for x in GRID_SHAPE[:2]])

    # 4x4 pixel sum and total size /2 x /2 = /4
    conv2d = torch.nn.Conv2d(1, 1, cwfs_obj.sum_size.int().item(), cwfs_obj.sum_size.int().item(), bias=False)
    conv2d.weight = torch.nn.Parameter(torch.ones([cwfs_obj.sum_size.int().item(), cwfs_obj.sum_size.int().item()], dtype=torch.complex64).unsqueeze(0).unsqueeze(0))
    conv2d.weight.requires_grad = False
    
    for it1 in range(DATASET_SIZE):
        # if it1 % 1000 == 0:
        print(f'Sample: {it1+1} / {DATASET_SIZE}')

        pm_height = torch.tensor(utils.sample_uniform(PM_HEIGHT))
        pm_smoothing = utils.sample_uniform(PM_SMOOTHING)
        focus_plan_var = torch.tensor(utils.sample_sigma(FOCUS_PLANE_VAR))
        dist_pm_im = torch.tensor(utils.sample_uniform(DIST_M_IM))
        na_sampled = utils.sample_uniform(NA)
        
        # generate phase mask
        phase_mask_obj.create_height_map(pm_height, PM_PROB)
        phase_mask = phase_mask_obj.forward(PM_RI, WL, pm_smoothing, padding=0)

        cwfs_obj.reset_grid()
        
        if NUM_ELEMENTS_ELLIPS > 0:
            num_elements_ellips = torch.randint(low=1, high=NUM_ELEMENTS_ELLIPS+1, size=(1,)).item()
            centers_ellips = torch.randn([num_elements_ellips, 3])*CENTER_SIGMA_ELLIPS + torch.tensor(cwfs_obj.xyz_sup)/2
            radii_ellips = torch.rand([num_elements_ellips, 3])*RADII_RANGE_ELLIPS + radii_min
            RIs_ellips = torch.randn(num_elements_ellips)*RI_SIGMA_ELLIPS + N_BG
            cwfs_obj.add_ellipsoids(centers_ellips, radii_ellips, RIs_ellips, random_rotation=True, softness=1e-16)
            
        if NUM_ELEMENTS_CUBES > 0:
            num_elements_cubes = torch.randint(low=1, high=NUM_ELEMENTS_CUBES+1, size=(1,)).item()
            centers_cubes = torch.randn([num_elements_cubes, 3])*CENTER_SIGMA_CUBES + torch.tensor(cwfs_obj.xyz_sup)/2
            lengths_cubes = torch.rand([num_elements_cubes, 3])*RADII_RANGE_CUBES + radii_min
            RIs_cubes = torch.randn(num_elements_cubes)*RI_SIGMA_CUBES + N_BG
            cwfs_obj.add_cubes(centers_cubes, lengths_cubes, RIs_cubes, random_rotation=True, softness=1e-16)
        
        cwfs_obj.PSC_approximator(const_sigma=PSC[0], defocus_sigma=PSC[1], defocus_dist=focus_plan_var*1e6)
        
        # propagate throught the grid and back prop to its center
        _ = cwfs_obj.wavefield_focus()
        lens.low_pass_filter(na_sampled)

        # reference, object fields and ground truth
        ref, obj, gt = cwfs_obj.forward(lens, dist_pm_im, phase_mask, focus_plane_var=focus_plan_var, gradient_median_kernel_size=GRAD_MEDIAN_KERNEL)
        
        # intensities sliced to keep center 500x500 from 700x700
        # start with larger to remove 'refelct' padding artifacts from boundary regions
        img_ref = ref[100:-100, 100:-100].abs()**2
        img_obj = obj[100:-100, 100:-100].abs()**2
        
        # saving amplitude
        
        amp_pm_plane = conv2d(cwfs_obj.field_mask_plane.unsqueeze(0)).squeeze().abs()[100:-100, 100:-100]
        tifffile.imwrite(f'{exp_path}/amplitude/{it1}.tiff', norm(amp_pm_plane, DATA_AMP_SCALE))
        
        # saving speckle imgs
        # divide by ref max and scale by 2e4 and save as int16
            
        obj_to_save = img_obj.numpy()
        ref_to_save = img_ref.numpy()

        tifffile.imwrite(f'{exp_path}/specimen/{it1}.tiff', norm(obj_to_save/ref_to_save.max(), DATA_SPECKLE_SCALE))
        tifffile.imwrite(f'{exp_path}/reference/{it1}.tiff', norm(ref_to_save/ref_to_save.max(), DATA_SPECKLE_SCALE))
        
        # saving ground turth gradient  imgs
        tifffile.imwrite(f'{exp_path}/gt_gradients/{it1}_0.tiff', norm(gt[0][100:-100, 100:-100] + DATA_GRAD_BIAS, DATA_GRAD_SCALE))
        tifffile.imwrite(f'{exp_path}/gt_gradients/{it1}_1.tiff', norm(gt[1][100:-100, 100:-100] + DATA_GRAD_BIAS, DATA_GRAD_SCALE))
        
        
        # save details in csv
        data_dict = {
            'img_id': it1,
            'phase_mask_height': pm_height.item()*1e6,
            'phase_mask_smoothing': pm_smoothing,
            'dist_pm_im': dist_pm_im.item()*1e6,
            'focus_plan_var': focus_plan_var.item()*1e6,
            'NA': na_sampled,
            'num_ellipsoids': num_elements_ellips if 'num_elements_ellips' in locals() else 0,
            'num_cubes': num_elements_cubes if 'num_elements_cubes' in locals() else 0
        }

        with open(f'{exp_path}/dataset_details.csv','a') as f:
            w = csv.writer(f)
            if it1 == 0:
                w.writerow(data_dict.keys())
            w.writerow(data_dict.values())