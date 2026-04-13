import torch
from torchvision.models.optical_flow import Raft_Large_Weights
import torchvision.transforms.functional as F
import numpy as np
from typing import Optional, Union
from torch import nn, Tensor


def init_model_RAFT(model, device='cuda', checkpoint=None):
    
    if checkpoint == None:
        print('Loading models with pretrained weights')
        weights = Raft_Large_Weights.DEFAULT.get_state_dict()
        weights['feature_encoder.convnormrelu.0.weight'] = torch.mean(weights['feature_encoder.convnormrelu.0.weight'], dim=1).unsqueeze(1)
        weights['context_encoder.convnormrelu.0.weight'] = torch.mean(weights['context_encoder.convnormrelu.0.weight'], dim=1).unsqueeze(1)
    else:
        print(f'Loading models with checkpoint: {checkpoint}')
        weights = torch.load(checkpoint, map_location=torch.device(device))
    
    # model = raft_large(progress=False)
    model.feature_encoder.convnormrelu[0] = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.context_encoder.convnormrelu[0] = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    
    model.load_state_dict(weights)
    
    return model.to(device)


def RAFT_loss(predictions, target, device='cuda'):
	loss_fn = torch.nn.L1Loss()
	N = len(predictions)
	
	w = torch.pow(torch.tensor(0.8), N-torch.tensor(range(1, N+1))).to(device)
	loss_n = torch.sum(torch.abs(torch.stack(predictions, dim=0) - target.unsqueeze(0).repeat(12, 1, 1, 1, 1)), dim=[1, 2, 3, 4])
	
	return torch.sum(w*loss_n).squeeze()


def process_labels(flows, size=[520, 960], antialias=True):
    
    imgs = []
    
    for im in range(len(flows)):
        imgs.append(F.resize(torch.stack(flows[im], dim=0).float(), size=size, antialias=antialias))

    return torch.stack(imgs)


class OpticalFlowTransformRAFT(nn.Module):
    def forward(self, img1, img2):
        if not isinstance(img1, Tensor):
            img1 = F.pil_to_tensor(img1)
        if not isinstance(img2, Tensor):
            img2 = F.pil_to_tensor(img2)

        img1 = F.convert_image_dtype(img1, torch.float)
        img2 = F.convert_image_dtype(img2, torch.float)

        # map [0, 1] into [-1, 1]
        img1 = F.normalize(img1, mean=[0.5], std=[0.5])
        img2 = F.normalize(img2, mean=[0.5], std=[0.5])

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        return img1, img2

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            "The images are rescaled to ``[-1.0, 1.0]``."
        )
        
def preprocess(ref_img, obj_img, transforms):
    img1_batch = torch.stack(ref_img)
    img2_batch = torch.stack(obj_img)

    img1_batch = F.resize(img1_batch, size=[520, 960], antialias=True)
    img2_batch = F.resize(img2_batch, size=[520, 960], antialias=True)

    # print(img1_batch.shape, img2_batch.shape)

    return transforms(img1_batch.unsqueeze(1), img2_batch.unsqueeze(1))


if __name__ == "__main__":
    pass
