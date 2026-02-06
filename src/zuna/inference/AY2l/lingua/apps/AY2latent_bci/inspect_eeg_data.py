# 1st, setup tmux and docker with lingua.sh
#   >> bash /mnt/home/chris/workspace/AY2l/lingua/lingua.sh (on Crusoe)
#
# 2nd, run something like:
#   >> python3 apps/AY2latent_bci/inspect_eeg_data.py

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def plot_eeg_sample(data,
                    fs=512,
                    sample=0,
                    idx=0,
                    fname_tag="",
                    dir_base="figures"):
    """
    Plot data EEG time trace, each channel on a different subplot.
    """

    chans,num_t = data.shape
    t = np.arange(num_t) / fs

    print(f"{chans=}, {num_t=}")

    dim = int(np.ceil(np.sqrt(chans)))
    fig, axes = plt.subplots(dim, dim, figsize=(24, 12))

    # Loop through each subplot and plot something
    ch=-1
    for i in range(dim):
        for j in range(dim):
            try:
                ch+=1
                # vals = set(data[ch]) # Checking for quantization artefacts. (SLOW and not needed)

                # Plot time-domain EEG (offset by channel index)
                axes[i, j].plot(t, data[ch], "b-", linewidth=0.5)
                axes[i, j].set_xlim(t[0],t[-1])
                axes[i, j].tick_params(axis='x', labelsize=10)
                axes[i, j].tick_params(axis='y', labelsize=10)
                axes[i, j].grid(True)
                axes[i, j].text(.98, .98, f"Ch{ch+1}", transform=axes[i, j].transAxes, ha='right', va='bottom', fontsize=12, color='black')
                # axes[i, j].text(.98, .98, f"#vals={len(vals)}", transform=axes[i, j].transAxes, ha='right', va='top', fontsize=12, color='red')
            
                if i==(dim-1) and j==0:
                    axes[i, j].set_xlabel("Time (s)")
                    axes[i, j].set_ylabel("Amp")
            except:
                break # If we run out of channels, just break
        
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # leave some space at top for suptitle
    plt.suptitle(f"EEG data {fname_tag} - ({idx=}, {sample=})", fontsize=16, fontweight='bold')

    plt.savefig(f"{dir_base}/eeg_data_{fname_tag}_S{sample}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_samples_from_file(data_path, save_dir, num_samps, fs):
    fname_tag = data_path.name.removesuffix(".pt")
    print(f"Loading in: {data_path}") # {i}/{len(data_paths)}
    mmap = torch.load(data_path).float().numpy()
    print(f"\t {mmap.shape}")
    print(f"\t Plotting {num_samps} random samples to {save_dir}")
    try:
        samples = np.random.choice(mmap.shape[0], num_samps, replace=False)
    except:
        pass # If there are not enough samples, just pass.

    for s in samples:
        plot_eeg_sample(data=mmap[s],
                        fs=fs,
                        sample=s,
                        idx=s,
                        fname_tag=fname_tag,
                        dir_base=save_dir)



def compute_stats_from_file(data_path,
                            divby=1):
    """
    Compute stats from a single file.
    """
    mmap = torch.load(data_path).float().numpy()/divby
    abs_mmap = np.abs(mmap)

    # Determine good channels (not all-zero) and construct masked mmap data.
    good_chan_mask = abs_mmap.sum(-1)!=0 # mask for active channels, ones that are not all-zero    
    num_good_chans = good_chan_mask.sum(-1) # how many good channels per sample
    #
    masked_mmap = np.where(good_chan_mask[..., None], mmap, np.nan)                     # [B,C,T] - w/ nans for masked channels
    masked_abs_mmap = np.where(good_chan_mask[..., None], abs_mmap, np.nan)             # [B,C,T] - w/ nans for masked channels
    masked_mmap_sample = masked_mmap.reshape(masked_mmap.shape[0], -1)                  # [B,CT]  - w/ nans for masked channels
    masked_abs_mmap_sample = masked_abs_mmap.reshape(masked_abs_mmap.shape[0], -1)      # [B,CT]  - w/ nans for masked channels

    # Sample-wise stats:
    sample_mean = np.nanmean(masked_mmap_sample, axis=-1)
    sample_std = np.nanstd(masked_mmap_sample, axis=-1)
    sample_absmax = np.nanmax(masked_abs_mmap_sample, axis=-1)
    sample_abs50CI = np.nanmedian(masked_abs_mmap_sample, axis=-1)
    sample_abs95CI = np.nanquantile(masked_abs_mmap_sample, 0.95, axis=-1)
    sample_abs05CI = np.nanquantile(masked_abs_mmap_sample, 0.05, axis=-1)

    # Channel-wise stats:
    channel_mean = np.nanmean(masked_mmap, axis=-1)
    channel_std = np.nanstd(masked_mmap, axis=-1)
    channel_absmax = np.nanmax(masked_abs_mmap, axis=-1)
    channel_abs50CI = np.nanmedian(masked_abs_mmap, axis=-1)
    channel_abs95CI = np.nanquantile(masked_abs_mmap, 0.95, axis=-1)
    channel_abs05CI = np.nanquantile(masked_abs_mmap, 0.05, axis=-1)
    #
    channel_mean_of_mean = np.nanmean(channel_mean,axis=-1)
    channel_mean_of_std = np.nanmean(channel_std,axis=-1)
    channel_mean_of_absmax = np.nanmean(channel_absmax,axis=-1)
    channel_mean_of_abs50CI = np.nanmean(channel_abs50CI,axis=-1)
    channel_mean_of_abs95CI = np.nanmean(channel_abs95CI,axis=-1)
    channel_mean_of_abs05CI = np.nanmean(channel_abs05CI,axis=-1)
    #
    channel_std_of_mean = np.nanstd(channel_mean,axis=-1)
    channel_std_of_std = np.nanstd(channel_std,axis=-1)
    channel_std_of_absmax = np.nanstd(channel_absmax,axis=-1)
    channel_std_of_abs50CI = np.nanstd(channel_abs50CI,axis=-1)
    channel_std_of_abs95CI = np.nanstd(channel_abs95CI,axis=-1)
    channel_std_of_abs05CI = np.nanstd(channel_abs05CI,axis=-1)


    return num_good_chans, sample_mean, sample_std, sample_absmax, sample_abs50CI, sample_abs95CI, sample_abs05CI, \
           channel_mean_of_mean, channel_mean_of_std, channel_mean_of_absmax, channel_mean_of_abs50CI, channel_mean_of_abs95CI, channel_mean_of_abs05CI, \
           channel_std_of_mean, channel_std_of_std, channel_std_of_absmax, channel_std_of_abs50CI, channel_std_of_abs95CI, channel_std_of_abs05CI


def plot_histogram(data, title, xlabel, ylabel="Frequency", fname=None, bins=100):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='blue', alpha=0.7)
    plt.title(f"{title} : {len(data)} samples")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.yscale('log')
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == "__main__":

    # Various things to plot:
    plot_EEG_random_samples = False
    compute_stats = True


    ## Dataset #1:
    data_dir = Path('/workspace/bci/data/mmap_june16_padded_fp32_HPF_chunked2')
    save_dir = "figures/inspect_datasets_1st_stats/"
    fs = 512
    chans = 64
    divby = 5 # divide the input by this value to get the correct scale

    ## Dataset #2:
    # data_dir = Path('/workspace/bci/data/tmp_tuh/')
    # save_dir = "figures/inspect_datasets_tuh_stats/"
    # fs = 256
    # chans = 23
    # divby = 1 # divide the input by this value to get the correct scale

    num_samps = 2   # number of samples from each file to plot
    n_proc = 128    # number of processes to use for multiprocessing

    data_paths = list(data_dir.glob('*.pt'))
    os.makedirs(save_dir, exist_ok=True)
    print(f"Found {len(data_paths)} data files in {data_dir}")

    # (1). Plot raw EEG from random samples from each .pt file
    if plot_EEG_random_samples:
        if n_proc == 1:
            ## Single process way. (for debugging)
            for data_path in data_paths:
                plot_samples_from_file(data_path, save_dir, num_samps, fs)
        else:
            ## Multiprocess way.
            args_list = [(path, save_dir, num_samps, fs) for path in data_paths]
            with Pool(processes=n_proc) as pool: # os.cpu_count()
                pool.starmap(plot_samples_from_file, args_list)



    # (2). Compute stats on the data.
    # Note: This is a bit slow, so prob need to do multiprocessing here too.
    if compute_stats:    
        if n_proc == 1:
            ## Single process way. (for debugging)
            results = []
            for i,data_path in enumerate(data_paths):
                res = compute_stats_from_file(data_path, divby=divby)
                results.append(res)
                if i>5:
                    break
        else:
            ## Multiprocess way.
            st = time.time()
            args_list = [(path, divby) for path in data_paths]
            with Pool(processes=n_proc) as pool: # os.cpu_count()
                results = pool.starmap(compute_stats_from_file, args_list)
            print(f"Elapsed time = {time.time() - st:.2f} seconds.")



        grouped = list(zip(*results)) 
        grouped_flat = [np.concatenate(g, axis=0) for g in grouped]


        # Plot histograms of the stats:
        plot_histogram(data=grouped_flat[0], 
                       title="Number of Good Channels per Sample", 
                       xlabel="Number of Good Channels", 
                       fname=f"{save_dir}/num_good_chans_hist.png", 
                       bins=list(range(chans+1))  # bins from 0 to chans
        )
        plot_histogram(data=grouped_flat[1],
                       title="Sample Mean of EEG Data",
                       xlabel="Sample Mean",
                       fname=f"{save_dir}/sample_mean_hist.png",
        )
        plot_histogram(data=grouped_flat[2],
                       title="Sample Standard Deviation of EEG Data",
                       xlabel="Sample Standard Deviation",
                       fname=f"{save_dir}/sample_std_hist.png",
        )
        plot_histogram(data=grouped_flat[3],
                       title="Sample Absolute Maximum of EEG Data",
                       xlabel="Sample Absolute Maximum",
                       fname=f"{save_dir}/sample_absmax_hist.png",
        )
        plot_histogram(data=grouped_flat[4],
                       title="Sample Absolute Median of EEG Data",
                       xlabel="Sample Absolute Median",
                       fname=f"{save_dir}/sample_abs50CI_hist.png",
        )
        plot_histogram(data=grouped_flat[5],
                       title="Sample Absolute 95% CI of EEG Data",
                       xlabel="Sample Absolute 95% CI",
                       fname=f"{save_dir}/sample_abs95CI_hist.png",
        )
        plot_histogram(data=grouped_flat[6],
                       title="Sample Absolute 05% CI of EEG Data",
                       xlabel="Sample Absolute 05% CI",
                       fname=f"{save_dir}/sample_abs05CI_hist.png",
        )
        plot_histogram(data=grouped_flat[7],
                       title="Channel Mean of Mean EEG Data",
                       xlabel="Channel Mean of Mean",
                       fname=f"{save_dir}/channel_mean_of_mean_hist.png",
        )
        plot_histogram(data=grouped_flat[8],
                       title="Channel Mean of Standard Deviation EEG Data",
                       xlabel="Channel Mean of Standard Deviation",
                       fname=f"{save_dir}/channel_mean_of_std_hist.png",
        )
        plot_histogram(data=grouped_flat[9],
                       title="Channel Mean of Absolute Maximum EEG Data",
                       xlabel="Channel Mean of Absolute Maximum",
                       fname=f"{save_dir}/channel_mean_of_absmax_hist.png",
        )
        plot_histogram(data=grouped_flat[10],
                       title="Channel Mean of Absolute Median EEG Data",
                       xlabel="Channel Mean of Absolute Median",
                       fname=f"{save_dir}/channel_mean_of_abs50CI_hist.png",
        )
        plot_histogram(data=grouped_flat[11],
                       title="Channel Mean of Absolute 95% CI EEG Data",
                       xlabel="Channel Mean of Absolute 95% CI",
                       fname=f"{save_dir}/channel_mean_of_abs95CI_hist.png",
        )
        plot_histogram(data=grouped_flat[12],
                       title="Channel Mean of Absolute 05% CI EEG Data",
                       xlabel="Channel Mean of Absolute 05% CI",
                       fname=f"{save_dir}/channel_mean_of_abs05CI_hist.png",
        )
        plot_histogram(data=grouped_flat[13],
                       title="Channel Standard Deviation of Mean EEG Data",
                       xlabel="Channel Standard Deviation of Mean",
                       fname=f"{save_dir}/channel_std_of_mean_hist.png",
        )
        plot_histogram(data=grouped_flat[14],
                       title="Channel Standard Deviation of Standard Deviation EEG Data",
                       xlabel="Channel Standard Deviation of Standard Deviation",
                       fname=f"{save_dir}/channel_std_of_std_hist.png",
        )
        plot_histogram(data=grouped_flat[15],
                       title="Channel Standard Deviation of Absolute Maximum EEG Data",
                       xlabel="Channel Standard Deviation of Absolute Maximum",
                       fname=f"{save_dir}/channel_std_of_absmax_hist.png",
        )
        plot_histogram(data=grouped_flat[16],
                       title="Channel Standard Deviation of Absolute Median EEG Data",
                       xlabel="Channel Standard Deviation of Absolute Median",
                       fname=f"{save_dir}/channel_std_of_abs50CI_hist.png",
        )
        plot_histogram(data=grouped_flat[17],
                       title="Channel Standard Deviation of Absolute 95% CI EEG Data",
                       xlabel="Channel Standard Deviation of Absolute 95% CI",
                       fname=f"{save_dir}/channel_std_of_abs95CI_hist.png",
        )
        plot_histogram(data=grouped_flat[18],
                       title="Channel Standard Deviation of Absolute 05% CI EEG Data",
                       xlabel="Channel Standard Deviation of Absolute 05% CI",
                       fname=f"{save_dir}/channel_std_of_abs05CI_hist.png",
        )


        # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)









    