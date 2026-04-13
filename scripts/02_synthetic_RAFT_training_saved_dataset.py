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
# from scipy.signal import correlate2d, convolve2d, tukey
# from scipy.interpolate import RegularGridInterpolator

import argparse
from datetime import datetime
from typing import Dict, Any
from torch import Tensor


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--exp_name", type=str, help='runs/date/exp_XXX', required=True)
    parser.add_argument("--train_params", type=str, help='configs/training_params.json', required=True)
    parser.add_argument("--dataset_path", type=str, help='data/dataset_name', required=True)
    
    args = parser.parse_args()

    EXP_NAME = args.exp_name
    date = f'{datetime.today()}'.split()[0]
    
    exp_path = f'../runs/{date}/{EXP_NAME}'
    assert os.path.isdir(exp_path) == False, "Experiment already exists."
    os.makedirs(exp_path)
    
    train_path = f'../configs/{args.train_params}.json'
    
    if os.path.exists(train_path):
        with open(train_path, 'r') as file:
            train_params = json.load(file)
            
        with open(os.path.join(exp_path, 'train_params.json'), 'w') as file:
            json.dump(train_params, file)
    else:
        print(f'{train_path} does not exist.')
        raise
    
    dataset_path = f'../data/{args.dataset_path}'
    assert os.path.exists(f'{dataset_path}/dataset_details.csv'), f"dataset_details.csv does not exist at {dataset_path}"
    assert os.path.exists(f'{dataset_path}/data_creation_cwfs_params.json'), f"data_creation_cwfs_params.json does not exist at {dataset_path}"

    DATA_SUBSET_PARAMS = train_params['dataset_subset']
    TRAIN_PARAMS = train_params['training_params']
    
    torch.manual_seed(TRAIN_PARAMS['SEED'])
    generator1 = torch.Generator().manual_seed(TRAIN_PARAMS['SEED'])
    
    # initializations
    if TRAIN_PARAMS['DEVICE'] == 'cuda':
        assert torch.cuda.is_available(), "CUDA not available. Switch to CPU."
    device = TRAIN_PARAMS['DEVICE'] 
    
    print(device)

    # TRAINING SETUP
    EPOCHS = TRAIN_PARAMS['EPOCHS']
    MINI_BATCH = TRAIN_PARAMS['MINI_BATCH']
    LR = TRAIN_PARAMS['LR']
    CKPT_SAVE = TRAIN_PARAMS['CKPT_SAVE']
    
    CKPT_PATH = TRAIN_PARAMS['CKPT_PATH'] if TRAIN_PARAMS['CKPT_LOAD'] else None
    model = hRAFT.init_model_RAFT(raft_large(progress=False), device=device, checkpoint=CKPT_PATH)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=TRAIN_PARAMS['WEIGHT_DECAY'])
    transforms = hRAFT.OpticalFlowTransformRAFT()
    val_loss = torch.nn.MSELoss()  # reduction = 'mean'

    # runs/dd_mm/model_data_v
    writer = SummaryWriter(exp_path)

    VAL_SPLIT = int(TRAIN_PARAMS['VAL_SPLIT'])
    TEST_SPLIT = int(TRAIN_PARAMS['TEST_SPLIT'])
    ellips_dataset = utils.SynthEllipsDataset(f'{dataset_path}/dataset_details.csv', 
                                              f'{dataset_path}/data_creation_cwfs_params.json', 
                                              f'{dataset_path}', subset=DATA_SUBSET_PARAMS)
    
    train_set, val_set, test_set = torch.utils.data.random_split(ellips_dataset, [len(ellips_dataset)- VAL_SPLIT - TEST_SPLIT, VAL_SPLIT, TEST_SPLIT], generator=generator1)
    print(f'Splitting dataset: Training-{len(train_set)}  Validation-{len(val_set)}  Test-{len(test_set)}')

    train_dataloader = DataLoader(train_set, batch_size=MINI_BATCH, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_set, batch_size=MINI_BATCH, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_set, batch_size=MINI_BATCH, shuffle=False, num_workers=0)
    
    i_batch_max = int(len(train_set)/MINI_BATCH)
    for it1 in range(EPOCHS):
        print(f'Epoch: {it1+1} / {EPOCHS}')
        
        model.train()
        for i_batch_train, (sample_batched, gt_batched, idx) in enumerate(train_dataloader):
            # print(f'It1-i_batch:{it1}-{i_batch_train}')
            ref_imgs = []
            obj_imgs = []
            gt_flows = []
            
            # for it2 in range(MINI_BATCH):
            for it2 in range(len(sample_batched[1][0])):
                
                ref_imgs.append(sample_batched[1][0][it2])
                obj_imgs.append(sample_batched[1][1][it2])
                gt_flows.append([gt_batched[0][it2], gt_batched[1][it2]])

            img1_batch, img2_batch = hRAFT.preprocess(ref_imgs, obj_imgs, transforms)
            list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
            
            targets = hRAFT.process_labels(gt_flows)
            loss = hRAFT.RAFT_loss(list_of_flows, targets.to(device), device)
        
            print(f'Minibatch-{i_batch_train+1}:{loss.item()}')
            writer.add_scalar('Loss Training', loss.item(), int(it1*i_batch_max + i_batch_train))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if int(it1*i_batch_max + i_batch_train) % TRAIN_PARAMS['VAL_EVERY'] == 0:
                # plot train images on tensorboard
                flow_imgs = flow_to_image(list_of_flows[-1].detach().cpu())
                flow_targets = flow_to_image(targets.detach().cpu())
                grid = np.hstack([np.concatenate([img1, flow_img], axis=2) for (img1, flow_img) in zip(flow_targets, flow_imgs)])
                grid = F.resize(torch.tensor(grid).detach(), (600, 900))
                
                writer.add_image('images_train', grid, int(it1*i_batch_max + i_batch_train))
                
                model.eval()
                with torch.no_grad():
                    # calculate validation loss and plot the first 3 images
                    running_loss = 0.0
                    for i_batch, (sample_batched, gt_batched, idx) in enumerate(val_dataloader):
                        
                        ref_imgs = []
                        obj_imgs = []
                        gt_flows = []
                        
                        # for it2 in range(MINI_BATCH):
                        for it2 in range(len(sample_batched[1][0])):

                            ref_imgs.append(sample_batched[1][0][it2])
                            obj_imgs.append(sample_batched[1][1][it2])
                            gt_flows.append(torch.stack([gt_batched[0][it2], gt_batched[1][it2]], dim=0))

                        img1_batch, img2_batch = hRAFT.preprocess(ref_imgs, obj_imgs, transforms)
                        list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
                        pred_flows = F.resize(list_of_flows[-1], size=gt_batched[0][0].shape)

                        running_loss += val_loss(torch.stack(gt_flows).to(device), pred_flows).item()*MINI_BATCH
                    
                    running_loss /= VAL_SPLIT
                    writer.add_scalar('Loss Validation', running_loss, int(it1*i_batch_max + i_batch_train))
                    
                    flow_imgs = flow_to_image(pred_flows.cpu())
                    flow_targets = flow_to_image(torch.stack(gt_flows).float())
                    grid = np.hstack([np.concatenate([img1, flow_img], axis=2) for (img1, flow_img) in zip(flow_targets, flow_imgs)])
                    grid = F.resize(torch.tensor(grid).detach(), (600, 900))
                    
                    writer.add_image('images_valid', grid, int(it1*i_batch_max + i_batch_train))
                    
            if CKPT_SAVE:
                if int(it1*i_batch_max + i_batch_train) % TRAIN_PARAMS['CKPT_SAVE_EVERY'] == 0:
                    if not os.path.exists(f'{exp_path}/ckpt'):
                        os.makedirs(f'{exp_path}/ckpt')
                    torch.save(model.state_dict(), f'{exp_path}/ckpt/{EXP_NAME}_{it1}_{i_batch_train}.pth')

            writer.flush()
        
            # Explicit cleanup
            del img1_batch, img2_batch, targets, list_of_flows
            torch.cuda.empty_cache()

            ref_imgs.clear()
            obj_imgs.clear()
            gt_flows.clear()
        
        model.eval()
        with torch.no_grad():
            # calculate test loss
            running_loss = 0.0
            for i_batch, (sample_batched, gt_batched, idx) in enumerate(test_dataloader):
                
                ref_imgs = []
                obj_imgs = []
                gt_flows = []
                
                # for it2 in range(MINI_BATCH):
                for it2 in range(len(sample_batched[1][0])):

                    ref_imgs.append(sample_batched[1][0][it2])
                    obj_imgs.append(sample_batched[1][1][it2])
                    gt_flows.append(torch.stack([gt_batched[0][it2], gt_batched[1][it2]], dim=0))

                img1_batch, img2_batch = hRAFT.preprocess(ref_imgs, obj_imgs, transforms)
                list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
                pred_flows = F.resize(list_of_flows[-1], size=gt_batched[0][0].shape)

                running_loss += val_loss(torch.stack(gt_flows).to(device), pred_flows).item()*MINI_BATCH
            
            running_loss /= TEST_SPLIT
            
            writer.add_scalar('Loss Test', running_loss, int(it1*i_batch_max + i_batch_train))
        

    writer.close()
