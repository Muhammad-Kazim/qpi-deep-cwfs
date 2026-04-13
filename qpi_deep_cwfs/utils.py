import pickle
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d as convolve
from scipy.fftpack import dct, idct
import csv
from tifffile import tifffile
import json

from typing import Optional, Tuple, List
from torch import nn, Tensor

import torch
import torchvision
import torchvision.transforms.functional as F
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from typing import Optional, Tuple, List, Dict, Any

from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve


def sample_uniform(PARAM_RANGE: Dict[str, Any]) -> float:
    """Check whether scalar or dist. If dist, return sample. OW, return scalar.

    Args:
        PARAM_RANGE (Dict[str, Any]): range of uniform distribution

    Returns:
        float: sampled value
    """
    
    if PARAM_RANGE['set']:
        scalar_min = PARAM_RANGE['MIN']
        scalar_max = PARAM_RANGE['MAX']
        scalar = torch.rand(size=(1,))*(scalar_max - scalar_min) + scalar_min
    else:
        scalar = torch.tensor(PARAM_RANGE['DEFAULT'])
        
    return scalar.item()


def sample_sigma(PARAM_RANGE: Dict[str, Any]) -> float:
    """Check whether scalar or dist. If dist, return sample. OW, return scalar.

    Args:
        PARAM_RANGE (Dict[str, Any]): gaussian distribution

    Returns:
        float: sampled value
    """
    
    if PARAM_RANGE['set']:
        scalar = torch.randn(size=(1,))*PARAM_RANGE['SIGMA'] + PARAM_RANGE['DEFAULT']
    else:
        scalar = torch.tensor(PARAM_RANGE['DEFAULT'])
        
    return scalar.item()


def load_pkl(filename):
    """Reads in a geometry pickled object.

    Args:
        filename (str): path/to/file.pkl

    Returns:
        geometry: geometry object
    """
    
    if os.path.isfile(filename):
        print('Loading geometry object...')
        with open(filename, 'rb') as inp:
            geom = pickle.load(inp)
        
        return geom
    else:
        print('File does not exist.')
        

def normalization(field, totype='int16'):
    """normalizes field to 0-1.

    Args:
        field (float): 2d floating point arrays.
        totype (str): 'int16' or 'int8'
    """
    
    field = (field - field.min())/(field.max() - field.min())
    
    if totype == 'int16':
        return np.array(field*(2**16) - 1, dtype=np.uint16)
    elif totype == 'int8':
        return np.array(field*(2**8) - 1, dtype=np.uint8)
    else:
        print('Wrong totype')
        

def low_pass_filter_NA(wavefield, wl, spatial_resolution, NA):
    fmax = NA/wl
    dx, dy = spatial_resolution[:2]
    
    kx = np.fft.fftfreq(wavefield.shape[0], dx)
    ky = np.fft.fftfreq(wavefield.shape[1], dy)
    
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
    K = np.sqrt(Kx**2 + Ky**2)
    mask = np.ones_like(K)
    mask[K > fmax] = 0.
    
    wave_spectrum = np.fft.fft2(wavefield)*mask
    
    return np.fft.ifft2(wave_spectrum)


def high_pass_filter_NA(wavefield, wl, spatial_resolution, NA):
    fmax = NA/wl
    dx, dy = spatial_resolution[:2]
    
    kx = np.fft.fftfreq(wavefield.shape[0], dx)
    ky = np.fft.fftfreq(wavefield.shape[1], dy)
    
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
    K = np.sqrt(Kx**2 + Ky**2)
    mask = np.ones_like(K)
    mask[K < fmax] = 0.
    
    wave_spectrum = np.fft.fft2(wavefield)*mask
    
    return np.fft.ifft2(wave_spectrum)


def band_pass_filter_NA(wavefield, wl, spatial_resolution, NA, loc):
    # ideally loc in space, means loc/(wl*focal_length) in the spectreum
    # but here loc = f_c
    
    fmax = NA/wl
    dx, dy = spatial_resolution[:2]
    
    kx = np.fft.fftshift(np.fft.fftfreq(wavefield.shape[0], dx))
    ky = np.fft.fftshift(np.fft.fftfreq(wavefield.shape[1], dy))
    
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
    K = np.sqrt(Kx**2 + Ky**2)
    mask = np.ones_like(K)
    mask[K > fmax] = 0.
    
    wave_spectrum = np.fft.fftshift(np.fft.fft2(wavefield))*np.roll(mask, (loc[0], loc[1]), axis=(1, 0))
    
    return np.fft.ifft2(np.fft.ifftshift(wave_spectrum))


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()


def grad_optr(image):

    dx = image[:, 1:] - image[:, :-1]
    dy = image[1:, :] - image[:-1, :] 
    
    dx = np.pad(dx, [[0, 0], [0, 1]], mode='edge')
    dy = np.pad(dy, [[0, 1], [0, 0]], mode='edge')

    return [dy, dx]


def freq_array(shape, sampling):
    f_freq_1d_x = np.fft.fftfreq(shape[0], sampling)
    f_freq_1d_y = np.fft.fftfreq(shape[1], sampling)
    f_freq_mesh = np.meshgrid(f_freq_1d_x, f_freq_1d_y, indexing='ij')
    f_freq = np.hypot(f_freq_mesh[0], f_freq_mesh[1])

    return f_freq


def int_2d_fourier(arr_ij, sampling, indexing='xy'):
    
    if indexing == 'ij':
        # indexing ij -> xy
        arr = [arr_ij[1], arr_ij[0]]
    else:
        arr = arr_ij

    freqs = freq_array(arr[0].shape, sampling)

    k_sq = np.where(freqs != 0, freqs**2, 1e-9)
    k = np.meshgrid(np.fft.fftfreq(arr[0].shape[0], sampling), np.fft.fftfreq(arr[0].shape[1], sampling), indexing='ij')

    v_int_x = np.real(np.fft.ifft2((np.fft.fft2(arr[1]) * k[0]) / (2*np.pi * 1j * k_sq)))
    v_int_y = np.real(np.fft.ifft2((np.fft.fft2(arr[0]) * k[1]) / (2*np.pi * 1j * k_sq)))

    v_int_fs = v_int_x + v_int_y
    return v_int_fs

def torch_freq_array(shape, sampling):
    f_freq_1d_x = torch.fft.fftfreq(shape[0], sampling)
    f_freq_1d_y = torch.fft.fftfreq(shape[1], sampling)
    f_freq_mesh = torch.meshgrid(f_freq_1d_x, f_freq_1d_y, indexing='ij')
    f_freq = torch.hypot(f_freq_mesh[0], f_freq_mesh[1])

    return f_freq


def torch_int_2d_fourier(arr_ij, sampling, indexing='xy'):
    
    if indexing == 'ij':
        # indexing ij -> xy
        arr = [arr_ij[1], arr_ij[0]]
    else:
        arr = arr_ij

    freqs = torch_freq_array(arr[0].shape, sampling)

    k_sq = torch.where(freqs != 0, freqs**2, torch.tensor(1e-9, dtype=freqs.dtype))
    k = torch.meshgrid(torch.fft.fftfreq(arr[0].shape[0], sampling), torch.fft.fftfreq(arr[0].shape[1], sampling), indexing='ij')

    v_int_x = torch.fft.ifft2((torch.fft.fft2(arr[1]) * k[0]) / (2*torch.pi * 1j * k_sq)).real
    v_int_y = torch.fft.ifft2((torch.fft.fft2(arr[0]) * k[1]) / (2*torch.pi * 1j * k_sq)).real

    v_int_fs = v_int_x + v_int_y
    return v_int_fs


def torch_ASDI_even_gradients(Wx, Wy):
    Wxm = torch.flip(Wx, dims=[0,1])     # Wx(-x, -y)
    Wym = torch.flip(Wy, dims=[0,1])     # Wy(-x, -y)

    Wx_xm = torch.flip(Wx, dims=[1])     # Wx(x, -y)
    Wy_xm = torch.flip(Wy, dims=[1])     # Wy(x, -y)

    Wx_my = torch.flip(Wx, dims=[0])     # Wx(-x, y)
    Wy_my = torch.flip(Wy, dims=[0])     # Wy(-x, y)

    dWx_e = torch.cat([
        torch.cat([-Wxm, -Wx_my], dim=1),
        torch.cat([Wx_xm, Wx], dim=1)], dim=0)

    dWy_e = torch.cat([
        torch.cat([-Wym, Wy_my], dim=1),
        torch.cat([-Wy_xm, Wy], dim=1)], dim=0)

    return (dWx_e, dWy_e)


def torch_fourier_integration(gradients: Tuple[Tensor, Tensor], sampling: float, method: str = 'ASDI', 
                              eps: float =1e-9, DC: Tuple[float, float] = [0., 0.], delinearize: bool =  False,
                              indexing: str = 'ij'):
    """_summary_

    Args:
        gradients (Tuple): gradients along rows and columns in this order
        sampling (float): dx**2/z for CWFS or more generally the pixel pitch
        method (str, optional): ASDI, mirror (boundary forced to zeros), or no padding
        eps (float, optional): Defaults to 1e-9.
        DC (Tuple, optional): Explicitly remove the constant term for the integrated field. Defaults to [0., 0.].
        delinearize (bool, optional): Removes Ax+b fit from the integrated field. Defaults to False.
        indexing  (str, optional): always 'ij'. kept for legacy reasons.

    Returns:
        phi (Tensor): integrated field 
    """
    
    nx, ny = gradients[0].shape
    
    if method == 'ASDI': # mirroring with antisymmetrization
        gradients_updated = torch_ASDI_even_gradients(gradients[0], gradients[1])
    elif method == 'mirror':
        pad = torchvision.transforms.Pad(int(nx/2), padding_mode='reflect')
        gradients_updated = pad(gradients)
    else:
        gradients_updated = gradients
    
    nx_up, ny_up = gradients_updated[0].shape
    
    f_freq_1d_x = torch.fft.fftfreq(nx_up, sampling)
    f_freq_1d_y = torch.fft.fftfreq(ny_up, sampling)
    
    k = torch.meshgrid(f_freq_1d_x, f_freq_1d_y, indexing='ij')
    freqs = torch.hypot(k[0], k[1])
    k_sq = freqs**2 + eps
    
    v_int_x = torch.fft.fft2(gradients_updated[0]) * k[0]
    v_int_y = torch.fft.fft2(gradients_updated[1]) * k[1]

    phi_f = -1j*(v_int_x + v_int_y)/(2*torch.pi * k_sq)
    phi_f[0, 0] = DC[0] + 1j*DC[1] # set DC explicitly
    
    phi_up = torch.fft.ifft2(phi_f).real
    
    if method == 'ASDI': # mirroring with antisymmetrization
        phi = phi_up[-nx:, -ny:]
    elif method == 'mirror':
        phi = phi_up[int(nx/2):-1*int(nx/2), int(ny/2):-1*int(ny/2)]
    else:
        phi = phi_up
    
    lin = 0.
    if delinearize:
        _, lin = fit_linear(phi)        
        
    return phi - lin


def fit_linear(img: Tensor):
    
    nx, ny = img.shape
    
    X, Y = torch.meshgrid(torch.arange(nx), torch.arange(ny), indexing='ij')
    A = torch.hstack([X.reshape(-1, 1), Y.reshape(-1, 1), torch.ones([nx, ny]).reshape(-1, 1)])

    pinv = torch.linalg.inv(A.T @ A) @ A.T
    abc = pinv @ img.reshape(-1, 1)
    
    return (abc, abc[0]*X + abc[1]*Y + abc[2])

def fit_quadratic(img: Tensor):
    
    nx, ny = img.shape
    
    X, Y = torch.meshgrid(torch.arange(nx), torch.arange(ny), indexing='ij')
    A = torch.hstack([X.reshape(-1, 1)**2, Y.reshape(-1, 1)**2, X.reshape(-1, 1)*Y.reshape(-1, 1), X.reshape(-1, 1), Y.reshape(-1, 1), torch.ones([nx, ny]).reshape(-1, 1)])

    pinv = torch.linalg.inv(A.T @ A) @ A.T
    abc = pinv @ img.reshape(-1, 1)
    
    return (abc, abc[0]*X**2 + abc[0]*Y**2 + abc[0]*X*Y + abc[0]*X + abc[1]*Y + abc[2])

class ObjImgMap():
    def __init__(self, wavelength: float, spatial_resolution: Tuple[float, float], numPx: Tuple[int, int]):
        """maps the wavefield in object focus plane to image plane by modulating with the pupil

        Args:
            wavelength (float): wavelength
            spatial_resolution (Tuple[float, float]): lateral resolution in object space (dx, dy)
            numPx (Tuple[float, float]): (Nx, Ny) in object space
        """

        self.wl = wavelength
        self.dx, self.dy = spatial_resolution[:2]
        self.nx, self.ny = numPx[:2]
                
        kx = torch.fft.fftshift(torch.fft.fftfreq(self.nx, self.dx))
        ky = torch.fft.fftshift(torch.fft.fftfreq(self.ny, self.dy))
        self.Kx, self.Ky = torch.meshgrid(kx, ky, indexing='ij')
        
        # aperture does nothing
        self.pupil_amp = torch.ones([self.nx, self.ny])
        self.pupil_phase = torch.zeros([self.nx, self.ny])
        
    def forward(self, wavefield: Tensor):
        
        pupil = self.pupil_amp*torch.exp(1j*2*torch.pi/self.wl*self.pupil_phase)
        wave_spectrum = torch.fft.fftshift(torch.fft.fft2(wavefield))*pupil
        
        return torch.fft.ifft2(torch.fft.ifftshift(wave_spectrum))
    
    def displaced_unifrom_pupil_amp(self, pupil_center = [0., 0.], pupil_radius=1e8, softness=1e-12):
            
        K = (self.Kx - pupil_center[0])**2 + (self.Ky - pupil_center[1])**2
        self.pupil_amp = 1 - torch.sigmoid((K - pupil_radius**2) / softness)
        
    def low_pass_filter(self, NA: float, softness=1e-12):
        fmax = NA/self.wl
        self.displaced_unifrom_pupil_amp(pupil_center = [0., 0.], pupil_radius=fmax, softness=softness)
        
    def set_pupil_amp(self, amp):
        assert amp.size() == (self.nx, self.ny), f"Aperture amplitude must have sizes {self.nx}x{self.ny}"
        self.pupil_amp = amp
        
    def set_pupil_phase(self, phase):
        assert phase.size() == (self.nx, self.ny), f"Aperture phase must have sizes {self.nx}x{self.ny}"
        self.pupil_phase = phase
        

def low_pass_filter(wavelength, spatial_resolution, numPx, NA):
    lpf = ObjImgMap(wavelength, spatial_resolution, numPx)
    return lpf.low_pass_filter(NA)


def torch_grad_optr(image, mode='edge'):

    d0 = image[:-1, :] - image[1:, :] # rows or y-axis
    d1 = image[:, :-1] - image[:, 1:] # columns or x-axis
    
    d0 = torchvision.transforms.Pad((0, 0, 0, 1), padding_mode=mode)(d0)
    d1 = torchvision.transforms.Pad((0, 0, 1, 0), padding_mode=mode)(d1)

    return [d0, d1]


def median_filter_2d(input_tensor: Tensor, kernel_size: int = 3) -> Tensor:
        """
        Apply median filtering to a BxCxHxW tensor (C channels per image).

        Args:
            input_tensor: torch.Tensor of shape (B, C, H, W)
            kernel_size: int, odd filter size

        Returns:
            torch.Tensor of shape (B, C, H, W)
        """
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        B, C, H, W = input_tensor.shape

        pad = kernel_size // 2

        # Apply unfold to each channel separately
        filtered_channels = []
        for c in range(C):
            channel = input_tensor[:, c:c+1, :, :]  # shape: (B, 1, H, W)
            patches = torch.nn.functional.unfold(channel, kernel_size=kernel_size, padding=pad)  # shape: (B, K*K, H*W)
            median = patches.median(dim=1)[0]  # shape: (B, H*W)
            median = median.view(B, 1, H, W)   # shape: (B, 1, H, W)
            filtered_channels.append(median)

        # Concatenate filtered channels back along dim=1
        return torch.cat(filtered_channels, dim=1)  # shape: (B, C, H, W)
    

def poisson_solver(gx, gy):
    """
    A DCT-based Poisson solver to integrate the surface from gradients.

    Parameters:
    - gx (np.ndarray): Gradient along the x-axis.
    - gy (np.ndarray): Gradient along the y-axis.

    Returns:
    - np.ndarray: Reconstructed surface.
    """
    # Pad size
    wid = 1
    gx = np.pad(gx, ((wid, wid), (wid, wid)))
    gy = np.pad(gy, ((wid, wid), (wid, wid)))
    
    # Define operators in the spatial domain
    nabla_x_kern = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    nabla_y_kern = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    
    # Define adjoint operator
    def nablaT(gx, gy):
        return convolve(gx, np.rot90(nabla_x_kern, 2), boundary='symmetric', mode='same') + \
               convolve(gy, np.rot90(nabla_y_kern, 2), boundary='symmetric', mode='same')

    # Generate inverse kernel
    H, W = gx.shape
    x_coord, y_coord = np.meshgrid(np.arange(W), np.arange(H))
    mat_x_hat = 2 * np.cos(np.pi * x_coord / W) + 2 * np.cos(np.pi * y_coord / H) - 4
    mat_x_hat[0, 0] = 1

    # Perform inverse filtering
    dct2 = lambda x: dct(dct(x.T, norm='ortho').T, norm='ortho')
    idct2 = lambda x: idct(idct(x.T, norm='ortho').T, norm='ortho')
    
    rec = idct2(dct2(nablaT(gx, gy)) / -mat_x_hat)
    rec = rec[wid:-wid, wid:-wid]
    rec = np.pad(rec[1:-1, 1:-1], ((1, 1), (1, 1)), mode='edge')
    rec -= np.mean(rec)
    
    return rec


def auto_corr_fn(image: Tensor):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.fft2(image).abs()**2).real)


def torch_TV(image: Tensor, eps: float = 1e-6):
    del0, del1 = torch_grad_optr(image)
    norm = torch.sqrt(del0**2 + del1**2 + eps)
    
    return norm.sum()/(image.view(-1).size()[0])


def torch_L2_grad(image: Tensor):
    del0, del1 = torch_grad_optr(image)
    norm = del0**2 + del1**2
    
    return norm.sum()/(image.view(-1).size()[0])


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class SynthEllipsDataset(Dataset):
    """SynthEllips dataset."""

    def __init__(self, csv_file, data_json, root_dir, subset=None, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            data_json (string): Path to json file with details about the data generation variables.
            subset (dict): filters to select only a subset of data from root_dir
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        assert os.path.exists(data_json), f"{data_json} does not exist."
        with open(data_json, 'r') as file:
            data_gen_params = json.load(file)['data_saving_params']
        
        self.grad_bias = data_gen_params["GRAD_DATA_BIAS"]
        self.grad_scale = data_gen_params["GRAD_DATA_SCALE"]
        self.speckle_scale = data_gen_params["SPECKLE_DATA_SCALE"]
        self.amp_scale = data_gen_params["AMP_DATA_SCALE"]
        
        assert os.path.exists(csv_file), f"{csv_file} does not exist."
        mydatacsv = csv.DictReader(open(csv_file))
        self.myimgIDs = []
        
        if subset is None:
            for img_file in mydatacsv:
                self.myimgIDs.append(img_file['img_id'])
        else:
            for img_file in mydatacsv:
                PM_H_BOOL =  float(img_file['phase_mask_height']) >= subset['PM_H_MIN'] and float(img_file['phase_mask_height']) <= subset['PM_H_MAX']
                PM_S_BOOL =  float(img_file['phase_mask_smoothing']) >= subset['PM_S_MIN'] and float(img_file['phase_mask_smoothing']) <= subset['PM_S_MAX']
                D_PM_IM_BOOL = float(img_file['dist_pm_im']) >= subset['D_PM_IM_MIN'] and float(img_file['dist_pm_im']) <= subset['D_PM_IM_MAX']
                NA_BOOL = float(img_file['NA']) >= subset['NA_MIN'] and float(img_file['NA']) <= subset['NA_MAX']
                NUM_ELLIPSOIDS_BOOL = float(img_file['num_ellipsoids']) >= subset['NUM_ELLIPSOIDS_MIN'] and float(img_file['num_ellipsoids']) <= subset['NUM_ELLIPSOIDS_MAX']

                if PM_H_BOOL and PM_S_BOOL and D_PM_IM_BOOL and NA_BOOL and NUM_ELLIPSOIDS_BOOL:
                    self.myimgIDs.append(img_file['img_id'])
            
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.myimgIDs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        amp = tifffile.imread(f'{self.root_dir}/amplitude/{self.myimgIDs[idx]}.tiff')/self.amp_scale
        obj = tifffile.imread(f'{self.root_dir}/specimen/{self.myimgIDs[idx]}.tiff')/self.speckle_scale
        ref = tifffile.imread(f'{self.root_dir}/reference/{self.myimgIDs[idx]}.tiff')/self.speckle_scale
        gt0 = tifffile.imread(f'{self.root_dir}/gt_gradients/{self.myimgIDs[idx]}_0.tiff')/self.grad_scale - self.grad_bias
        gt1 = tifffile.imread(f'{self.root_dir}/gt_gradients/{self.myimgIDs[idx]}_1.tiff')/self.grad_scale - self.grad_bias
        
        # if self.transform:
            # pass
            
        sample = ((amp), (ref, obj))
        gt = (gt0, gt1)

        return sample, gt, self.myimgIDs[idx]
    

def normalized_cross_corr(x, y, eps=1e-12):
    """Normalized Cross Correlation"""
    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)

    numerator = torch.sum(x_centered * y_centered)

    x_variance = torch.sqrt(torch.sum(x_centered ** 2) + eps)
    y_variance = torch.sqrt(torch.sum(y_centered ** 2) + eps)

    denominator = x_variance * y_variance

    ncc = numerator / denominator
    return ncc

def integrate_flow_field(flo):
    """
    Python equivalent of the MATLAB function integrate_flow_field.

    Parameters
    ----------
    flo : ndarray (H, W, 2)
        Flow field where flo[...,0] = Ix and flo[...,1] = Iy

    Returns
    -------
    I : ndarray (H, W)
        Integrated scalar field
    """

    Ix = flo[:, :, 0]
    Iy = flo[:, :, 1]

    H, W = Ix.shape

    # Create mask with Dirichlet boundary conditions
    mask = np.ones((H, W), dtype=bool)
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False

    # Total number of pixels
    Isize = H * W

    # Construct full Poisson matrix
    p = np.vstack([
        np.ones(Isize),
        np.ones(Isize),
        -4 * np.ones(Isize),
        np.ones(Isize),
        np.ones(Isize)
    ])

    offsets = [-H, -1, 0, 1, H]
    P = spdiags(p, offsets, Isize, Isize, format="csr")

    # Vectorized mask
    mask_vec = mask.ravel()

    # Restrict columns
    Pr = P[:, mask_vec]

    # Compute divergence of gradient (right-hand side)
    # MATLAB:
    # padarray(diff(Ix,1,2), [0,1], 'replicate', 'post')
    dIx = np.diff(Ix, axis=1)
    dIx = np.pad(dIx, ((0, 0), (0, 1)), mode='edge')

    # padarray(diff(Iy,1,1), [1,0], 'replicate', 'post')
    dIy = np.diff(Iy, axis=0)
    dIy = np.pad(dIy, ((0, 1), (0, 0)), mode='edge')

    divgradI = dIx + dIy

    rhs = divgradI.ravel()[mask_vec]

    # Restrict rows and columns
    Ph = Pr[mask_vec, :]

    # Solve linear system
    sol = spsolve(Ph, rhs)

    # Put solution back into image
    I = np.zeros((H, W))
    I.ravel()[mask_vec] = sol

    return I.reshape(Ix.shape)


if __name__=='__main__':
    pass