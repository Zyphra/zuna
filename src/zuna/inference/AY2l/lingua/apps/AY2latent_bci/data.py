from torch.utils.data import Dataset
import torch
import torchaudio
from .audiotools.audiotools import AudioSignal, transforms, datasets
from dataclasses import dataclass
from dataclasses import field
import random
from typing import Optional, Tuple, Union, List
import numpy as np
from .feats import AmplitudeCompressedComplexSTFT
from .augs import RandomCompression
import einops  # Add this import


@dataclass
class AudioDatasetArgs:
    data_csv_paths: List[str] = field(default_factory=list)
    data_csv_repeats: List[float] = field(default_factory=list)
    sample_rate: int = 48000
    sample_duration_seconds: float = 15.0

    encoder_transform: str = "stft"
    encoder_input_channels: int = 1024

    decoder_transform: str = "stft"
    decoder_input_channels: int = 1024

    batch_size: int = 48
    num_workers: int = 64
    pin_memory: bool = True
    shuffle: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    diffusion_forcing: bool = False
    diffusion_noise_schedule: str = "linear"

    patching_type: str = "frames"

    stft_alpha: float = 0.3
    stft_beta: float = 0.3
    stft_global_sigma: Union[str, float] = 0.1
    seq_len: int = field(init=False)

    def __post_init__(self):
        self.seq_len = (self.sample_rate // self.encoder_input_channels) + 1



class AutoencoderDataset(Dataset):
    def __init__(self, args: AudioDatasetArgs):
        self.args = args

        assert args.diffusion_noise_schedule == "linear", "Only linear diffusion noise schedule is supported for now"
        # assert args.patching_type == "frames", "Only frame patching is supported for now"

        # self.loaders = [datasets.AudioLoader([csv_path]) for csv_path in args.data_csv_paths]

        # # Calculate the number of files per loader
        # self.files_per_loader = [sum(len(audio_list) for audio_list in loader.audio_lists) for loader in self.loaders]
        
        # # Calculate effective length per loader (files * repeats)
        # self.effective_len_per_loader = [int(files * repeat) for files, repeat in zip(self.files_per_loader, args.data_csv_repeats)]
        
        # # Calculate weights for sampling loaders
        # total_effective_len = sum(self.effective_len_per_loader)
        # if total_effective_len > 0:
        #     self.weights = np.array(self.effective_len_per_loader) / total_effective_len
        # else:
        #     print("Warning: No data found in the dataset?? Dummybro??")
        #     self.weights = np.ones(len(self.loaders)) / len(self.loaders)

        #read csvs into a list where each element is a line. drop the headers.
        self.csv_contents = []

        for csv_path in args.data_csv_paths:
            with open(csv_path, 'r') as f:
                lines = f.readlines()[1:]
                self.csv_contents.append(lines)

        self.total_effective_len = sum([len(lines) * repeat for lines, repeat in zip(self.csv_contents, args.data_csv_repeats)])
        # self.weights = [len(lines) * repeat / self.total_effective_len for lines, repeat in zip(self.csv_contents, args.data_csv_repeats)]
        self.weights = [repeat / sum(args.data_csv_repeats) for repeat in args.data_csv_repeats]


        if args.encoder_transform == "stft":
            self.encoder_transform = AmplitudeCompressedComplexSTFT(
                n_fft = (args.encoder_input_channels*2)-2,
                hop_length=args.encoder_input_channels,
                sampling_rate=args.sample_rate,
                alpha=args.stft_alpha,
                beta=args.stft_beta,
                window_fn="hann"
            )
        else:
            raise ValueError(f"Unknown encoder transform, not implemented: {args.encoder_transform}")
        
        if args.decoder_transform == "stft":
            self.decoder_transform = AmplitudeCompressedComplexSTFT(
                n_fft = (args.decoder_input_channels*2)-2,
                hop_length=args.decoder_input_channels,
                sampling_rate=args.sample_rate,
                alpha=args.stft_alpha,
                beta=args.stft_beta,
                window_fn="hann"
            )
        else:
            raise ValueError(f"Unknown decoder transform, not implemented: {args.decoder_transform}")
        
        self.audiotools_transforms = [
            transforms.VolumeNorm(["const", -16], prob=1.0),
            # transforms.ShiftPhase(prob=0.2),
            # transforms.FrequencyMask(prob=0.2),
            transforms.RescaleAudio(prob=1.0),
        ]

        self.waveform_transforms = [
            RandomCompression(min_bitrate=8, max_bitrate=32, prob=0.1),
        ]

        self.seq_len = (args.sample_rate // args.encoder_input_channels)+1

        print("USING WEIGHTS OF", self.weights)

        if type(args.stft_global_sigma) == str:
            self.global_sigma = torch.load(args.stft_global_sigma)
            print("Loaded global sigma from", args.stft_global_sigma)
        else:
            self.global_sigma = args.stft_global_sigma

        self.state = np.random.RandomState()

        self.patch_type = args.patching_type

        
    def apply_audiotools_transforms(self, audio_item: AudioSignal):
        for transform in self.audiotools_transforms:
            kwargs = transform.instantiate(signal=audio_item)
            audio_item = transform(audio_item, **kwargs)
            # print(f"Applied transform: {transform.__class__.__name__}, audio shape: {audio_item.audio_data.shape}")
        return audio_item
    
    def apply_waveform_transforms(self, waveform: torch.Tensor):
        for transform in self.waveform_transforms:
            waveform = transform(waveform)
        return waveform
    
    def dataload(self, path, duration, sample_rate, num_channels, loudness_cutoff):
        signal = AudioSignal.zeros(duration, sample_rate, num_channels)

        signal = AudioSignal.salient_excerpt(
            path,
            duration=duration,
            state=self.state,
            loudness_cutoff=loudness_cutoff,
        )

        if num_channels == 1:
            signal = signal.to_mono()
        signal = signal.resample(sample_rate)

        if signal.duration < duration:
            signal = signal.zero_pad_to(int(duration * sample_rate))
        
        return signal


    def __getitem__(self, idx):
        # Create random state using idx as seed for reproducibility
        # state = np.random.RandomState(idx)
        
        # # Select a loader randomly based on the weights
        # loader_idx = state.choice(len(self.loaders), p=self.weights)
        # selected_loader = self.loaders[loader_idx]
        
        # Get audio sample from the selected loader
        # audio_item = selected_loader(
        #     state=state,
        #     sample_rate=self.args.sample_rate,
        #     duration=self.args.sample_duration_seconds,
        #     num_channels=1,  # Using mono audio
        # )

        try:
            selected_source = random.choices(self.csv_contents, weights=self.weights)[0]
            selected_path = random.choice(selected_source)

            signal = self.dataload(selected_path.strip().strip('"'), self.args.sample_duration_seconds, self.args.sample_rate, 1, -50)
        except Exception as e:
            print("Error loading audio", e)
            return self.__getitem__(idx+1)
        
        encoder_signal = self.apply_audiotools_transforms(signal.deepcopy())
        encoder_signal = self.apply_waveform_transforms(encoder_signal.audio_data)
        
        # Compute STFTs and convert to real representation
        raw_encoder_stft = self.encoder_transform(encoder_signal)
        raw_decoder_stft = self.decoder_transform(signal.audio_data)
        
        # Convert complex to real representation
        encoder_stft_real = torch.view_as_real(raw_encoder_stft)  # [batch, freq, time, 2]
        decoder_stft_real = torch.view_as_real(raw_decoder_stft)  # [batch, freq, time, 2]

        # print(encoder_stft_real.shape)
                
        # Reshape to [batch, time, features]
        encoder_stft = einops.rearrange(encoder_stft_real, 'x b f t c -> (x b) t (f c)')
        decoder_stft = einops.rearrange(decoder_stft_real, 'x b f t c -> (x b) t (f c)')
        
        t_shape = (1, 1, decoder_stft.shape[-2], 1) if self.args.diffusion_forcing else (1,1,1,1)
        t = torch.rand(*t_shape, device=decoder_stft.device)
        
        # Handle global_sigma based on its shape
        if isinstance(self.global_sigma, torch.Tensor) and self.global_sigma.numel() > 1:
            # If sigma is per-frequency, duplicate for both real and imaginary parts
            sigma = einops.repeat(self.global_sigma, 'f -> 1 1 (f 2)')
        else:
            sigma = self.global_sigma
            
        noise = torch.randn_like(decoder_stft) * sigma

        noisy_decoder_stft = (1-t) * decoder_stft + t * noise
        decoder_targets = noise - decoder_stft


        out_dict = {"encoder_stft": encoder_stft, "decoder_input_stft": noisy_decoder_stft, "target": decoder_targets, "t": t.squeeze(-1),}


        if self.patch_type == "conv_stem":
            #go to (B, 2, T, C//2)
            for k, v in out_dict.items():
                if k != "t":
                    out_dict[k] = einops.rearrange(v, 'b t (f c) -> b c f t', c=2)

        out_dict = {k: v.squeeze(0).contiguous() for k,v in out_dict.items()}
        
        return out_dict

    def __len__(self):
        return int(self.total_effective_len)
    

def worker_init_fn(worker_id, seed=42, rank=0):
    """Initialize worker with unique seed."""
    # Create unique seed for this worker and rank
    worker_seed = seed + worker_id + rank * 10000
    
    # Set all random seeds for this worker
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    
    # Set the dataset's random state
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:  # In multiprocessing
        worker_info.dataset.state = np.random.RandomState(worker_seed)
        print(f"Worker {worker_id} on rank {rank} using seed {worker_seed}")

def create_dataloader(args: AudioDatasetArgs, seed, rank):
    dataset = AutoencoderDataset(args)

        
    import functools
    init_fn = functools.partial(worker_init_fn, seed=seed, rank=rank)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=args.shuffle,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        worker_init_fn=init_fn,  # Add this line
        
    )