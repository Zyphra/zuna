# %%
from apps.AY2latent.transformer import *
from apps.AY2latent.data_lean import AudioDatasetArgs, AutoencoderDataset, create_dataloader
import torch.distributed.checkpoint as dcp
import einops
from tqdm import tqdm
import torch
from omegaconf import OmegaConf
#import garbage collector
import gc
gc.collect()
from apps.AY2latent.data_lean import STFTProcessor
config_name = "/workspace/AY2latent/lingua/apps/AY2latent/configs/AY2vec2_repa2_fmask_new_data_fat.yaml"
conf = OmegaConf.load(config_name)

args = AudioDatasetArgs(**conf['data'])
args.sample_duration_seconds = 2.0
args.fmax_masking = True
args.load_csv()
dataset = AutoencoderDataset(args)
data = dataset[0]

args.batch_size = 128
args.num_workers = 16
args.shuffle = True
args.prefetch_factor = 2
args.persistent_workers = False
args.pin_memory = False
dataloader = create_dataloader(args, 69, 0)
iter_dataloader = iter(dataloader)

args.diffusion_forcing=True
stft_processor = STFTProcessor(args)

# processed = stft_processor.process(**data)

stft_processor.to('cuda')
batches = []
for i in tqdm(range(1024*2)):
    with torch.no_grad():
        data = next(iter_dataloader)
        
        # Move data to GPU with non_blocking=True
        data = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        
        # Process data on GPU
        stft = einops.rearrange(torch.view_as_real(stft_processor.decoder_transform(data['decoder_audio'])), "b f t c -> b t (f c)")
        # processed['orig_fmax'] = data['fmax']
        # processed['signal_sum'] = data['decoder_audio'].sum(dim=1)
        data['stft'] = stft
        data['signal_sum'] = data['decoder_audio'].sum(dim=-1).view(-1)

        #delete encoder_audio and decoder_audio
        del data['encoder_audio']
        del data['decoder_audio']
        
        # Move processed data back to CPU before appending to list
        data = {k: v.to('cpu', non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        
        batches.append(data)
        
        # Optional: clear cache periodically
        if i % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()


# %%
torch.cuda.synchronize()

#delete dataloader, args, dataset, batches
del dataloader
del args
del dataset

# for batch in batches:
#     del batch['encoder_audio']
#     batch['signal_sum'] = batch['decoder_audio'].sum(dim=-1).view(-1)
#     del batch['decoder_audio']

torch.cuda.empty_cache()
gc.collect()

test_collated = torch.utils.data.default_collate(batches)

del batches
torch.cuda.empty_cache()
gc.collect()


# %%
test_collated['decoder_audio'].shape, test_collated['stft'].shape

# %%
signal_sum = test_collated['signal_sum']#test_collated['decoder_audio'].sum(dim=-1).view(-1)
signal_sum_masked = ((signal_sum) < -1e2) | ((signal_sum) > 1e2) | (signal_sum==0)
(signal_sum_masked.sum(), signal_sum_masked.shape)
mask = signal_sum_masked
print(mask.sum(), mask.shape)
#invert mask
mask = ~mask.view(-1)
print(mask.sum(), mask.shape)

# %%
stft_shape = test_collated['stft'].shape
batch_stft = test_collated['stft'].view(-1, stft_shape[-2], stft_shape[-1])[mask]
print(batch_stft.shape)

# %%

args = AudioDatasetArgs(**conf['data'])
args.sample_duration_seconds = 2.0
args.fmax_masking = True

# %%
torch.cuda.empty_cache()
gc.collect()

# %%
#make histogram of test_collated['fmax'].view(-1, 1, 1)[mask]
import matplotlib.pyplot as plt
import numpy as np
fmax = test_collated['fmax'].view(-1, 1, 1)[mask]
print(fmax.shape)
plt.hist(fmax.view(-1).cpu().numpy(), bins=100)
plt.xlabel('fmax')
plt.ylabel('Frequency')
plt.title('Histogram of fmax')
plt.show()

# %%
freq_mask = torch.arange(args.decoder_input_channels).view(1, 1, args.decoder_input_channels) < test_collated['fmax'].view(-1, 1, 1)[mask]
print(freq_mask.shape)
freq_mask = einops.repeat(freq_mask, "b t c -> b t (f c)", f=2)
print(freq_mask.shape)

# %%
classic_sugma = batch_stft.std(dim=(0,1))
std_dim1_sugma = batch_stft.std(dim=1)
new_sugma = std_dim1_sugma.mean(0)


# %%
freq_mask = freq_mask.squeeze(1)
fmax_sugma = std_dim1_sugma * freq_mask
fmax_sugma = fmax_sugma.sum(dim=0) / freq_mask.sum(dim=0)

print(classic_sugma.shape, new_sugma.shape, fmax_sugma.shape)

# %%
#it's of shape [2048] we want to convolve the first [1024] and the latter [1024] separately
smooth_sugma = fmax_sugma.clone()
smooth_sugma[:1024] = F.conv1d(fmax_sugma[:1024].unsqueeze(0).unsqueeze(0), torch.ones(1, 1, 3)/3, padding=1,).squeeze()
smooth_sugma[1024:] = F.conv1d(fmax_sugma[1024:].unsqueeze(0).unsqueeze(0), torch.ones(1, 1, 3)/3, padding=1).squeeze()

#remoe the edge artifacts
smooth_sugma[:1] = fmax_sugma[:1]
smooth_sugma[1024:1025] = fmax_sugma[1024:1025]
smooth_sugma[2047:] = fmax_sugma[2047:]
smooth_sugma[1023:1024] = fmax_sugma[1023:1024]

# %%
# Minimal code for Gaussian smoothing applied separately

import torch
import torch.nn.functional as F
import math


def gaussian_kernel1d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Creates a 1D Gaussian kernel for convolution."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd for symmetric Gaussian kernel")
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum() # Normalize
    return g.view(1, 1, kernel_size)

def smooth_sugma_gaussian(sugma: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    Applies Gaussian smoothing separately to the first and second halves
    of the input tensor using reflection padding (manually applied).

    Args:
        sugma: The input tensor of shape [2048] to smooth.
        kernel_size: The size of the Gaussian kernel (must be odd).
        sigma: The standard deviation of the Gaussian kernel.

    Returns:
        The smoothed tensor of shape [2048].
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd for symmetric padding")
    if sugma.shape[0] % 2 != 0:
        raise ValueError("Input tensor length must be even")

    device = sugma.device
    n_freq_bins_half = sugma.shape[0] // 2
    sugma_float = sugma.float() # Ensure float for convolution

    gauss_k = gaussian_kernel1d(kernel_size, sigma, device)
    pad_amount = (kernel_size - 1) // 2
    # *** Correction: Manual padding using F.pad ***
    padding_tuple_manual = (pad_amount, pad_amount) # Pad left and right

    # Magnitude part
    mag_part = sugma_float[:n_freq_bins_half].view(1, 1, -1)
    mag_part_padded = F.pad(mag_part, padding_tuple_manual, mode='reflect')
    mag_smoothed = F.conv1d(mag_part_padded, gauss_k, padding=0) # Use padding=0 now

    # Phase part
    phase_part = sugma_float[n_freq_bins_half:].view(1, 1, -1)
    phase_part_padded = F.pad(phase_part, padding_tuple_manual, mode='reflect')
    phase_smoothed = F.conv1d(phase_part_padded, gauss_k, padding=0) # Use padding=0 now

    smoothed_sugma = torch.cat([mag_smoothed.squeeze(0).squeeze(0), phase_smoothed.squeeze(0).squeeze(0)], dim=0)

    return smoothed_sugma.to(sugma.dtype) # Restore original dtype
# --- Example Usage ---
# Assuming fmax_sugma is your [2048] tensor calculated previously
# fmax_sugma = ... 

# Apply the smarter smoothing
smooth_sugma = fmax_sugma.clone()
print(smooth_sugma[:32],'\n', fmax_sugma[:32])
# smooth_sugma[-1] = smooth_sugma[-2]
smooth_sugma = torch.clip(smooth_sugma, min=0.01, max=1e3)
print(smooth_sugma[:32],'\n', fmax_sugma[:32])
smooth_sugma = smooth_sugma_gaussian(smooth_sugma, kernel_size=7, sigma=1.0) 
# Or apply to classic_sugma if desired
# smooth_classic_sugma = smooth_sugma_gaussian(classic_sugma, kernel_size=5, sigma=1.0)

# Add the smoothing part to your existing script:
# Replace the F.conv1d and manual edge fixing section with:
# smooth_sugma = smooth_sugma_gaussian(fmax_sugma, kernel_size=5, sigma=1.0)
# (Make sure the gaussian_kernel1d and smooth_sugma_gaussian functions are defined above)

print(smooth_sugma[:32],'\n', fmax_sugma[:32])

# %%
import torch
import torch.nn.functional as F
import math

def gaussian_kernel1d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Creates a 1D Gaussian kernel for convolution."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd for symmetric Gaussian kernel")
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum() # Normalize
    return g.view(1, 1, kernel_size)

def conditional_median_filter1d(
    tensor: torch.Tensor, 
    kernel_size: int = 5, 
    threshold_factor: float = 1.5
) -> torch.Tensor:
    """
    Applies a median filter conditionally to a 1D tensor based on deviation 
    from the local median relative to the overall standard deviation.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Median kernel_size must be odd")
    if tensor.ndim != 1:
         raise ValueError("Input tensor must be 1D")

    padding = (kernel_size - 1) // 2
    padded_tensor = F.pad(tensor.view(1, 1, -1), (padding, padding), mode='reflect')
    
    windows = padded_tensor.unfold(dimension=-1, size=kernel_size, step=1)
    median_filtered_tensor = windows.median(dim=-1).values.squeeze(0).squeeze(0)

    overall_std = tensor.std().clamp(min=1e-6) 
    threshold = threshold_factor * overall_std
    difference = torch.abs(tensor - median_filtered_tensor)
    is_outlier = difference > threshold

    output_tensor = tensor.clone()
    output_tensor[is_outlier] = median_filtered_tensor[is_outlier]
    return output_tensor

def smooth_sugma_median_gaussian(
    sugma: torch.Tensor, 
    median_kernel_size: int = 5, 
    median_threshold_factor: float = 1.5, 
    gaussian_kernel_size: int = 5, 
    gaussian_sigma: float = 1.0
) -> torch.Tensor:
    """
    Applies a conditional median filter followed by Gaussian smoothing, 
    separately to the first and second halves of the input tensor.
    """
    if gaussian_kernel_size % 2 == 0:
        raise ValueError("Gaussian kernel_size must be odd")
    if sugma.shape[0] % 2 != 0:
        raise ValueError("Input tensor length must be even")

    device = sugma.device
    n_freq_bins_half = sugma.shape[0] // 2
    sugma_float = sugma.float()

    # --- Conditional Median Filter ---
    mag_median_filtered = conditional_median_filter1d(
        sugma_float[:n_freq_bins_half], median_kernel_size, median_threshold_factor
    )
    phase_median_filtered = conditional_median_filter1d(
        sugma_float[n_freq_bins_half:], median_kernel_size, median_threshold_factor
    )

    # --- Gaussian Smoothing ---
    gauss_k = gaussian_kernel1d(gaussian_kernel_size, gaussian_sigma, device)
    pad_amount = (gaussian_kernel_size - 1) // 2
    padding_tuple_manual = (pad_amount, pad_amount)

    # Magnitude part
    mag_part = mag_median_filtered.view(1, 1, -1)
    mag_part_padded = F.pad(mag_part, padding_tuple_manual, mode='reflect')
    mag_smoothed = F.conv1d(mag_part_padded, gauss_k, padding=0)

    # Phase part
    phase_part = phase_median_filtered.view(1, 1, -1)
    phase_part_padded = F.pad(phase_part, padding_tuple_manual, mode='reflect')
    phase_smoothed = F.conv1d(phase_part_padded, gauss_k, padding=0)

    smoothed_sugma = torch.cat([mag_smoothed.squeeze(0).squeeze(0), phase_smoothed.squeeze(0).squeeze(0)], dim=0)

    return smoothed_sugma.to(sugma.dtype)


smooth_sugma_2 = smooth_sugma_median_gaussian(
    fmax_sugma.clone(), 
    median_kernel_size=3, 
    median_threshold_factor=1.5, 
    gaussian_kernel_size=5, 
    gaussian_sigma=1.0
) 

# %%
import matplotlib.pyplot as plt

# smooth_sugma = new_sugma.clone()
plt.figure(figsize=(20, 10), dpi=300)

plt.plot(classic_sugma.cpu().numpy())#[:32])
plt.title("Classic Sugma")
plt.xlabel("Frequency Bin")
plt.ylabel("Standard Deviation")
plt.show()

plt.figure(figsize=(20, 10), dpi=300)

plt.plot(new_sugma.cpu().numpy())#[:32])
plt.title("New Sugma")
plt.xlabel("Frequency Bin")
plt.ylabel("Standard Deviation")
plt.show()


plt.figure(figsize=(20, 10), dpi=300)

plt.plot(smooth_sugma.cpu().numpy())#[:32])
plt.title("FMax Sugma")
plt.xlabel("Frequency Bin")
plt.ylabel("Standard Deviation")
plt.show()

plt.figure(figsize=(20, 10), dpi=300)

plt.plot(smooth_sugma_2.cpu().numpy())#[:32])
plt.title("Smooth FMax Sugma")
plt.xlabel("Frequency Bin")
plt.ylabel("Standard Deviation")
plt.show()

# %%
#grab filename of config_name
filename = config_name.split("/")[-1].split(".")[0]
#get current date in yyyy-mm-dd-hh-mm-ss format
from datetime import datetime
now = datetime.now()
date_string = now.strftime("%Y-%m-%d-%H-%M-%S")

print(filename, date_string)

torch.save(classic_sugma, "/workspace/AY2latent/lingua/apps/AY2latent/sugma/" + "_classic_sugma_" + filename +  "_" + date_string + ".pt")
torch.save(new_sugma, "/workspace/AY2latent/lingua/apps/AY2latent/sugma/" + "_new_sugma_" + filename +  "_" + date_string + ".pt")
torch.save(fmax_sugma, "/workspace/AY2latent/lingua/apps/AY2latent/sugma/" + "_fmax_sugma_" + filename +  "_" + date_string + ".pt")
torch.save(smooth_sugma_2, "/workspace/AY2latent/lingua/apps/AY2latent/sugma/" + "_smooth_sugma_" + filename +  "_" + date_string + ".pt")

# %%



