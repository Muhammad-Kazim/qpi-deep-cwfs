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
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.functional as F
from torchvision.utils import flow_to_image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from scipy.ndimage import gaussian_filter
import json

import argparse
from datetime import datetime
from typing import Dict, Any

from torch import Tensor
from py_cwfs_alg import cws_module
import csv


if __name__=='__main__':
    """
    Inference on SynthELlips test set using ckpt and cwfs add method based on json file in configs/infer....json
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_params_path", type=str, help='configs/infer_params_path.json', required=True)
    args = parser.parse_args()

    infer_params_path = f'../configs/{args.infer_params_path}.json'

    if os.path.exists(infer_params_path):
        with open(infer_params_path, 'r') as file:
            infer_params = json.load(file)
    else:
        print(f'{infer_params_path} does not exist.')
        raise

    date = f'{datetime.today()}'.split()[0]

    EXP_NAME = '_'.join([date, infer_params['general']['exp_name']])

    exp_path = f'../infer/{EXP_NAME}'
    assert os.path.isdir(exp_path) == False, "Experiment already exists."
    os.makedirs(exp_path)

    # with open(os.path.join(exp_path, f'infer_2025-12-05_synthEllipsV0_params.json'), 'w') as file:
    with open(os.path.join(exp_path, f'{args.infer_params_path}.json'), 'w') as file:
        json.dump(infer_params, file)

    DATA_PARAMS = infer_params['data_params']
    PX_SIZE = DATA_PARAMS["DIGITAL_PX_SIZE"]
    DATA_SUBSET_PARAMS = infer_params['dataset_subset']

    DATASET_PATH = DATA_PARAMS['DATASET_PATH']
    assert os.path.exists(f'{DATASET_PATH}/dataset_details.csv'), f"dataset_details.csv does not exist at {DATASET_PATH}"
    assert os.path.exists(f'{DATASET_PATH}/data_creation_cwfs_params.json'), f"data_creation_cwfs_params.json does not exist at {DATASET_PATH}"

    torch.manual_seed(DATA_PARAMS['SEED'])
    generator1 = torch.Generator().manual_seed(DATA_PARAMS['SEED'])

    NETWORK_PARAMS = infer_params['network_params']

    # initializations
    if NETWORK_PARAMS['DEVICE'] == 'cuda':
        assert torch.cuda.is_available(), "CUDA not available. Switch to CPU."
    device = NETWORK_PARAMS['DEVICE'] 

    print(device)

    CKPT_PATH = NETWORK_PARAMS['CKPT_PATH'] if NETWORK_PARAMS['CKPT_LOAD'] else None
    model = hRAFT.init_model_RAFT(raft_large(progress=False), device=device, checkpoint=CKPT_PATH)

    transforms = hRAFT.OpticalFlowTransformRAFT()

    val_loss = torch.nn.MSELoss()  # reduction = 'mean'

    VAL_SPLIT = int(DATA_PARAMS['VAL_SPLIT'])
    TEST_SPLIT = int(DATA_PARAMS['TEST_SPLIT'])
    ellips_dataset = utils.SynthEllipsDataset(f'{DATASET_PATH}/dataset_details.csv', 
                                                f'{DATASET_PATH}/data_creation_cwfs_params.json', 
                                                f'{DATASET_PATH}', subset=DATA_SUBSET_PARAMS)

    _, _, test_set = torch.utils.data.random_split(ellips_dataset, [len(ellips_dataset)- VAL_SPLIT - TEST_SPLIT, VAL_SPLIT, TEST_SPLIT], generator=generator1)
    print(f'Splitting dataset: Test-{len(test_set)}')

    MINI_BATCH = NETWORK_PARAMS['MINI_BATCH']
    test_dataloader = DataLoader(test_set, batch_size=MINI_BATCH, shuffle=False, num_workers=0)
    
    CWFS_PARAMS = infer_params['cwfs_params']

    CWFS_PRIOR = CWFS_PARAMS['prior']
    CWFS_ITER = CWFS_PARAMS['iter']
    CWFS_SCALING = CWFS_PARAMS['data_scaling']
    CWFS_TOL = CWFS_PARAMS['tolerance']

    wc_reconstructor = cws_module.CWS()
    
    test_dataloader = DataLoader(test_set, batch_size=MINI_BATCH, shuffle=False, num_workers=0)

    model.eval()
    with torch.no_grad():
        # calculate test loss
        for i_batch, (sample_batched, gt_batched, idx) in enumerate(test_dataloader):
            idx = int(idx[0])
            with open(f'{DATASET_PATH}/dataset_details.csv', 'r') as fp:
                specimen_details = fp.readlines()[idx + 1]
            
            dist_pm_im = float(specimen_details.split(',')[3])*1e-6
                
            ref_imgs = []
            obj_imgs = []
            gt_flows = []
            
            # for it2 in range(MINI_BATCH):
            for it2 in range(len(sample_batched[1][0])):

                ref_imgs.append(sample_batched[1][0][it2])
                obj_imgs.append(sample_batched[1][1][it2])
                gt_flows.append(torch.stack([gt_batched[0][it2], gt_batched[1][it2]], dim=0))

            gt_opd = -1e6*PX_SIZE**2/dist_pm_im*utils.integrate_flow_field(np.stack([gt_flows[0][1], gt_flows[0][0]], axis=2))
            
            if NETWORK_PARAMS['network']:
                img1_batch, img2_batch = hRAFT.preprocess(ref_imgs, obj_imgs, transforms)
                list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
                pred_flows = F.resize(list_of_flows[-1].detach().cpu(), size=gt_batched[0][0].shape)
                nn_opd = -1e6*PX_SIZE**2/dist_pm_im*utils.integrate_flow_field(np.stack([pred_flows[0][1], pred_flows[0][0]], axis=2))
                loss_nn = val_loss(torch.tensor(gt_opd), torch.tensor(nn_opd)).item()
            
            if CWFS_PARAMS['cwfs']:
                _, _, _ = wc_reconstructor.run((ref_imgs[0] + CWFS_SCALING[0]).numpy()*CWFS_SCALING[1], (obj_imgs[0] + CWFS_SCALING[0]).numpy()*CWFS_SCALING[1], 
                            prior=CWFS_PRIOR, iter=CWFS_ITER, verbose=False, tol=CWFS_TOL)
                _, admm_opd = wc_reconstructor.get_field(pixel_size=PX_SIZE, z=dist_pm_im, RI=2.)
                loss_admm = val_loss(torch.tensor(gt_opd), 1e6*torch.tensor(admm_opd)).item()
                
                
            fig, axs = plt.subplots(1, 2, figsize=(18, 4), width_ratios=[1, 3.25])

            vmax, vmin = gt_opd.max(), gt_opd.min()
            frame = np.hstack([gt_opd, nn_opd, 1e6*admm_opd])

            axs[0].imshow(obj_imgs[0], cmap='gray')
            cm1 = axs[1].imshow(frame, cmap='inferno', vmax=vmax, vmin=vmin)

            plt.colorbar(cm1, ax=axs[1], pad=5e-3)

            for i in range(2):
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                
            plt.tight_layout()
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1e-3)

            plt.savefig(f'{exp_path}/{idx}.png', 
                        dpi=200, bbox_inches='tight')
            # plt.show()
            
            ncc_nn = utils.normalized_cross_corr(torch.tensor(gt_opd), torch.tensor(nn_opd)).item()
            ncc_admm = utils.normalized_cross_corr(torch.tensor(gt_opd), 1e6*torch.tensor(admm_opd)).item()
            
            with open(f'{DATASET_PATH}/dataset_details.csv', 'r') as fp:
                line = fp.readlines()[idx + 1]

            # save details in csv
            data_dict = {
                'img_id': int(line.split(',')[0]),
                'phase_mask_height': float(line.split(',')[1]),
                'phase_mask_smoothing': float(line.split(',')[2]),
                'dist_pm_im': float(line.split(',')[3]),
                'focus_plan_var': float(line.split(',')[4]),
                'NA': float(line.split(',')[5]),
                'num_ellipsoids': int(line.split(',')[6]),
                'mse_nn_um_opds_um': float(loss_nn)*1e6,
                'mse_admm_um_opds_um': float(loss_admm)*1e6,
                'ncc_nn': ncc_nn,
                'ncc_admm': ncc_admm
            }

            with open(f'{exp_path}/dataset_details.csv','a') as f:
                w = csv.writer(f)
                if i_batch == 0:
                    w.writerow(data_dict.keys())
                w.writerow(data_dict.values())
                
                
            # if i_batch == 2:
            #     break