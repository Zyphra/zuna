from functools import lru_cache
from torch.utils.data import Dataset
import torch
import torchaudio
from .audiotools.audiotools import AudioSignal, transforms
from dataclasses import dataclass
from dataclasses import field
import random
from typing import Union, List
import numpy as np
from .feats import AmplitudeCompressedComplexSTFT
from .augs import RandomCompression, get_resample_object, get_resample_object_soxr
import einops
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from .serialize_list import TorchSerializedList
from .timeout import Timeout
import logging
import os

import pandas as pd
import math

import soxr

@dataclass
class AudioDatasetArgs:
    data_csv_paths: List[str] = field(default_factory=list)
    data_csv_repeats: List[int] = field(default_factory=list)
    sample_rate: int = 48000
    sample_duration_seconds: float = 10.0
    getitem_timeout: int = 10

    encoder_transform: str = "stft"
    encoder_input_channels: int = 1024

    decoder_transform: str = "stft"
    decoder_input_channels: int = 1024

    batch_size: int = 40
    num_workers: int = 16
    pin_memory: bool = True
    shuffle: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    time_masking_noise_prob: float = 0.1
    time_masking_noise_margin: float = 0.2

    diffusion_forcing: bool = False
    diffusion_noise_schedule: str = "linear"
    diffusion_forcing_num_frames: int = 1

    patching_type: str = "frames"

    stft_alpha: float = 0.3
    stft_beta: float = 0.3
    stft_global_sigma: Union[str, float] = 0.1

    fmax_masking: bool = False

    random_downsamples: bool = False

    seq_len: int = field(init=False)

    csv_contents: List[str] = field(default_factory=list)
    generate_16khz_decoder: bool = False

    fmax_bias: bool = False

    norm_by_sigma: bool = False

    def __post_init__(self):
        self.seq_len = (
            int(
                ((self.sample_rate // self.encoder_input_channels) + 1)
                * self.sample_duration_seconds
            )
            - 1
        )

    # def load_csv(self):
    #     print("Entered load_csv and now loading csvs", self.data_csv_paths, self.data_csv_repeats)
    #     csv_contents = []
    #     for csv_path in self.data_csv_paths:
    #         with open(csv_path, "r") as f:
    #             lines = f.readlines()[1:]
    #             csv_contents.append(lines)

    #     #remove any lines that end with .mp3
    #     # self.csv_contents = [[line for line in csv if not line.strip().endswith(".mp3")] for csv in self.csv_contents]

    #     csv_contents = [
    #         csv * int(repeats)
    #         for csv, repeats in zip(csv_contents, self.data_csv_repeats)
    #     ]
    #     # merge and shuffle csv_contents
    #     csv_contents = sorted(
    #         [item for sublist in csv_contents for item in sublist]
    #     )
    #     # print(self.csv_contents)

    #     print("WE FOUND CSV CONTENTS OF LENGTH", len(csv_contents))
    #     try:
    #         self.csv_contents = TorchSerializedList(csv_contents)
    #     except Exception as e:
    #         print(f"Error loading CSV contents: {e}")
    #         self.csv_contents = csv_contents

    # def load_csv(self):
    #     all_paths = []
    #     # Pad repeats list with 1s if shorter than paths list
    #     repeats = self.data_csv_repeats + [1] * (len(self.data_csv_paths) - len(self.data_csv_repeats))

    #     for path, r in zip(self.data_csv_paths, repeats):
    #         if not os.path.exists(path): continue # Skip non-existent files
    #         try:
    #             # Try reading with header=0 (standard), skip bad lines
    #             df = pd.read_csv(path, header=0, on_bad_lines='skip', low_memory=False, skipinitialspace=True)
    #             # Find 'path' column (case-insensitive) or default to the first column (index 0)
    #             col = next((c for c in df.columns if str(c).strip().lower() == 'path'), df.columns[0])
    #             # Extract paths, clean them, drop any NaNs/empty strings
    #             paths = df[col].astype(str).str.strip().str.strip('"').dropna().tolist()
    #             all_paths.extend(paths * int(r)) # Repeat and add to list
    #         except Exception: # Broad except to catch file reading errors or missing columns in headerless files
    #             try: # Fallback: Try reading with no header
    #                 df = pd.read_csv(path, header=None, on_bad_lines='skip', low_memory=False, skipinitialspace=True)
    #                 paths = df.iloc[:, 0].astype(str).str.strip().str.strip('"').dropna().tolist() # Assume first col
    #                 all_paths.extend(paths * int(r))
    #             except Exception as e_fallback:
    #                 print(f"Skipping file {path} after failing to read with/without header: {e_fallback}")

    #     if not all_paths:
    #         print("No paths loaded from any CSV file.")
    #         self.csv_contents = []
    #         return

    #     all_paths.sort() # Sort all collected paths
    #     random.Random(6969).shuffle(all_paths)
    #     try: # Attempt to use TorchSerializedList
    #         self.csv_contents = TorchSerializedList(all_paths)
    #     except Exception: # Fallback to regular list
    #         self.csv_contents = all_paths
    #         print("Failed to create TorchSerializedList, using standard list.")

    def load_csv(self):
        """
        Load CSVs expecting exactly 3 columns (path, duration, sample_rate)
        in that exact order. Produces a list of dicts for each row, e.g.:
            {"path": ..., "duration": ..., "sample_rate": ...}
        Then sorts, shuffles, and (if possible) packs into a TorchSerializedList.
        """
        try:
            import polars as pl
        except ImportError:
            print(
                "Polars not installed; falling back to pandas. For best performance, install polars."
            )
            import pandas as pd

        all_rows = []
        # Ensure repeats list is at least as long as paths list
        repeats = self.data_csv_repeats + [1] * (
            len(self.data_csv_paths) - len(self.data_csv_repeats)
        )

        for csv_path, r in zip(self.data_csv_paths, repeats):
            if not os.path.exists(csv_path):
                print(f"Skipping non-existent file: {csv_path}")
                continue

            # Try polars first, fallback to pandas if polars fails or isn't available
            try:
                # --- Polars read ---
                df = pl.read_csv(
                    csv_path,
                    has_header=True,
                    ignore_errors=True,  # robust skipping of corrupt lines
                )
                # Ensure the first three columns match exactly ["path", "duration", "sample_rate"]
                if len(df.columns) < 3 or df.columns[:3] != [
                    "path",
                    "duration",
                    "sample_rate",
                ]:
                    raise ValueError(
                        "Columns are not in the exact order: path, duration, sample_rate."
                    )

                # Optionally, you can also enforce *exactly* 3 columns:
                # if df.shape[1] != 3:
                #     raise ValueError("CSV must have exactly 3 columns: path, duration, sample_rate.")

                # Slice out just the first three columns
                df = df.select(df.columns[:3])
                # Convert to list-of-dicts
                rows = df.to_dicts()
            except Exception as e_polars:
                print(f"Polars read failed for {csv_path} with error: {e_polars}")
                # --- Fallback: pandas ---
                try:
                    import pandas as pd

                    # We'll read with 'on_bad_lines' to skip corrupt rows
                    pd_df = pd.read_csv(
                        csv_path,
                        header=0,
                        on_bad_lines="skip",
                        low_memory=False,
                        skipinitialspace=True,
                    )
                    # Check columns
                    if len(pd_df.columns) < 3:
                        raise ValueError(
                            "Not enough columns to match path,duration,sample_rate."
                        )
                    # Because we need them in order, rename or reorder explicitly:
                    pd_df = pd_df.iloc[:, :3]
                    pd_df.columns = ["path", "duration", "sample_rate"]
                    
                    # Convert to list-of-dicts
                    rows = pd_df.to_dict("records")
                except Exception as e_pandas:
                    print(f"Failed reading {csv_path} with pandas fallback: {e_pandas}")
                    continue  # skip this file altogether

            # Repeat and accumulate
            # but first log the number of rows found and the repeat factor
            #compute total duration of csv by summing
            total_duration = sum(float(row["duration"]) for row in rows)
            print(
                f"Found {len(rows)} rows in {csv_path}, repeating {r} times. Total duration: {total_duration/3600} hours."
            )
            rows *= int(r)
            all_rows.extend(rows)

        # If we found nothing at all, just store an empty list
        if not all_rows:
            print("No valid rows found from any CSV file.")
            self.csv_contents = []
            return

        # Sort by 'path' then shuffle (seeded with 6969 for consistency)
        all_rows.sort(key=lambda d: d["path"])
        random.Random(6969).shuffle(all_rows)

        # Finally, attempt to pack into TorchSerializedList for memory benefits
        try:
            self.csv_contents = TorchSerializedList(all_rows)
        except Exception as e:
            print(
                f"Failed to create TorchSerializedList, using standard list. Error: {e}"
            )
            self.csv_contents = all_rows

    # exclude csv_contents when saving
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "csv_contents"}

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # Remove csv_contents from the state dictionary
    #     if 'csv_contents' in state:
    #         del state['csv_contents']
    #     return state
# @torch.compile()
def estimate_fmax(
    wav: torch.Tensor,
    sample_rate: int,
    threshold: float = 0.001,
) -> torch.Tensor:
    assert wav.ndim < 3
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    freqs = torch.linspace(0, sample_rate // 2, 1 + wav.shape[-1] // 2, device=wav.device).unsqueeze(0).repeat(wav.shape[0], 1)
    fft_mag = torch.fft.rfft(wav).abs()
    mask = fft_mag > threshold * fft_mag.max(dim=-1, keepdim=True)[0]
    freqs.masked_fill_(~mask, -torch.inf)
    return freqs.max(dim=-1)[0]


def beta_sched(t_shape, device, dtype):
    t = torch.randn(t_shape, device=device, dtype=dtype) * 2 + 0.3
    t = torch.sigmoid_(t) * 1.02 - 0.01
    t = t.clamp_(0,1)#.round_(decimals=5)
    return t
class STFTProcessor:
    def __init__(self, args: AudioDatasetArgs):
        # self.args = args
        self.diffusion_noise_schedule = args.diffusion_noise_schedule
        subtractor = 1 if args.sample_rate % args.encoder_input_channels == 0 else 2

        if args.encoder_transform == "stft":
            self.encoder_transform = AmplitudeCompressedComplexSTFT(
                n_fft=(args.encoder_input_channels * 2) - subtractor,
                hop_length=args.encoder_input_channels,
                sampling_rate=args.sample_rate,
                alpha=args.stft_alpha,
                beta=args.stft_beta,
                window_fn="hann",
            )
        else:
            raise ValueError(
                f"Unknown encoder transform, not implemented: {args.encoder_transform}"
            )

        if args.decoder_transform == "stft":
            self.decoder_transform = AmplitudeCompressedComplexSTFT(
                n_fft=(args.decoder_input_channels * 2) - subtractor,
                hop_length=args.decoder_input_channels,
                sampling_rate=args.sample_rate,
                alpha=args.stft_alpha,
                beta=args.stft_beta,
                window_fn="hann",
            )
        else:
            raise ValueError(
                f"Unknown decoder transform, not implemented: {args.decoder_transform}"
            )

        if type(args.stft_global_sigma) == str:
            self.global_sigma = torch.load(args.stft_global_sigma)
            print("!!!!!!!! Loaded global sigma from", args.stft_global_sigma)
        else:
            self.global_sigma = args.stft_global_sigma

        self.patch_type = args.patching_type
        self.diffusion_forcing = args.diffusion_forcing
        if self.diffusion_forcing:
            self.diffusion_forcing_num_frames = args.diffusion_forcing_num_frames
            # assert args.seq_len % self.diffusion_forcing_num_frames == 0, (
            #     "Diffusion forcing num frames must be divisible by seq_len"
            # )

        self.channel_arange = torch.arange(args.decoder_input_channels).view(1, 1, args.decoder_input_channels)

        self.fmax_bias = args.fmax_bias

        self.norm_by_sigma = args.norm_by_sigma

    def to(self, device):
        """Move components to specified device."""
        self.encoder_transform = self.encoder_transform.to(device)
        self.decoder_transform = self.decoder_transform.to(device)
        if isinstance(self.global_sigma, torch.Tensor):
            self.global_sigma = self.global_sigma.to(device)

        self.channel_arange = self.channel_arange.to(device)
        return self

    @torch.compile()
    def process(self, encoder_audio, decoder_audio, time_masks=None, fmax=None, distill_16khz_decoder_audio=None):
        """
        Process audio waveforms to create STFT and apply noise.

        Args:
            encoder_audio: Audio tensor for encoder [B, T]
            decoder_audio: Audio tensor for decoder [B, T]
            time_masks: Mask for padding [B, T] where T is post-processed time
            fmax: Maximum frequency band (int from 0 to num_bands) for each batch element [B]


        Returns:
            Dictionary containing processed tensors
        """
        # Compute STFTs and convert to real representation
        raw_encoder_stft = self.encoder_transform(encoder_audio)
        raw_decoder_stft = self.decoder_transform(decoder_audio)

        # Convert complex to real representation
        encoder_stft_real = torch.view_as_real(
            raw_encoder_stft
        )  # [batch, freq, time, 2]
        decoder_stft_real = torch.view_as_real(
            raw_decoder_stft
        )  # [batch, freq, time, 2]

        # Reshape to [batch, time, features]
        # encoder_stft = einops.rearrange(encoder_stft_real, "b f t c -> b t (f c)") #fucking bug, interleaved instead of concating!
        # decoder_stft = einops.rearrange(decoder_stft_real, "b f t c -> b t (f c)")
        encoder_stft = torch.cat([encoder_stft_real[...,0], encoder_stft_real[...,1]], dim=1).mT #[batch, time, freq*2 ]
        decoder_stft = torch.cat([decoder_stft_real[...,0], decoder_stft_real[...,1]], dim=1).mT #[batch, time, freq*2 ]


        assert time_masks.shape[0] == encoder_stft.shape[0], (
            f"Time masks shape {time_masks.shape} does not match encoder_stft shape {encoder_stft.shape}"
        )
        assert time_masks.shape[1] == encoder_stft.shape[1], (
            f"Time masks shape {time_masks.shape} does not match encoder_stft shape {encoder_stft.shape}"
        )


        batch, seq_len, channels = decoder_stft.shape
        # print("Decoder STFT shape", decoder_stft.shape)

        freq_mask = None
        if fmax is not None:
            if self.fmax_bias:
                fmax = torch.abs(fmax)
                fmax = torch.where(fmax != channels//2, fmax - 86, fmax)
                fmax = torch.clamp(fmax, 1, channels//2)

            freq_mask = self.channel_arange < (torch.abs(fmax).view(-1, 1, 1))
            freq_mask = einops.repeat(freq_mask, "b t c -> b t (f c)", f=2)

        t_shape = (
            (batch, (seq_len // self.diffusion_forcing_num_frames)+1, 1)
            if self.diffusion_forcing
            else (batch, 1, 1)
        )
        if self.diffusion_noise_schedule == "linear":
            t = torch.rand(*t_shape, device=decoder_audio.device)
        elif self.diffusion_noise_schedule == "beta":
            t = beta_sched(t_shape, device=decoder_audio.device, dtype=decoder_audio.dtype)

        # if diffusion forcing, duplicate dim 1 to match decoder_stft seq_len such that t1 t2 t3 -> t1 t1 ... t2 t2 ... t3 t3 ..
        if self.diffusion_forcing:
            t = torch.repeat_interleave(t, self.diffusion_forcing_num_frames, dim=1)[:, :seq_len, :]

        # Handle global_sigma based on its shape
        if (
            isinstance(self.global_sigma, torch.Tensor)
            and self.global_sigma.numel() > 1
        ):
            # If sigma is per-frequency, duplicate for both real and imaginary parts
            sigma = einops.repeat(
                self.global_sigma,
                "f -> 1 1 (f c)",
                c=encoder_stft.shape[-1] // self.global_sigma.numel(),
            )
        else:
            sigma = self.global_sigma

        noise = torch.randn_like(decoder_stft) * (1 if self.norm_by_sigma else sigma)
        decoder_stft = decoder_stft / (sigma if self.norm_by_sigma else 1)

        noisy_decoder_stft = (1 - t) * decoder_stft + t * noise
        decoder_targets = noise - decoder_stft

        out_dict = {
            "encoder_input": encoder_stft,
            "decoder_input": noisy_decoder_stft,
            "target": decoder_targets,
            "t": t,
            "time_masks": time_masks,
            "freq_mask": freq_mask,
        }

        if self.patch_type == "conv_stem":
            # Go to (B, 2, T, C//2)
            assert self.patch_type != "conv_stem", (
                "Not implemented for now due to time masking shape bullshit"
            )
            for k, v in out_dict.items():
                if k != "t":
                    out_dict[k] = einops.rearrange(v, "b t (f c) -> b c f t", c=2)

        return out_dict





class AutoencoderDataset(Dataset):
    def __init__(self, args: AudioDatasetArgs):
        # self.args = args

        # assert args.diffusion_noise_schedule == "linear", (
        #     "Only linear diffusion noise schedule is supported for now"
        # )

        # # Read CSVs into a list where each element is a line. Drop the headers.
        # self.csv_contents = []

        # for csv_path in args.data_csv_paths:
        #     with open(csv_path, "r") as f:
        #         lines = f.readlines()[1:]
        #         self.csv_contents.append(lines)

        # # self.total_effective_len = sum(
        # #     [
        # #         len(lines) * repeat
        # #         for lines, repeat in zip(self.csv_contents, args.data_csv_repeats)
        # #     ]
        # # )
        # # self.weights = [
        # #     repeat / sum(args.data_csv_repeats) for repeat in args.data_csv_repeats
        # # ]

        # self.csv_contents = [
        #     csv * int(repeats)
        #     for csv, repeats in zip(self.csv_contents, args.data_csv_repeats)
        # ]
        # # merge and shuffle csv_contents
        # self.csv_contents = sorted(
        #     [item for sublist in self.csv_contents for item in sublist]
        # )
        self.csv_contents = args.csv_contents
        self.total_effective_len = len(self.csv_contents)
        self.encoder_input_channels = args.encoder_input_channels
        self.sample_rate = args.sample_rate
        self.sample_duration_seconds = args.sample_duration_seconds
        self.dataload_timeout = args.getitem_timeout

        self.decoder_input_channels = args.decoder_input_channels

        # Create STFTProcessor (only for reference, actual processing will happen elsewhere)
        # We'll keep this here so the dataset knows about the transform parameters
        subtractor = 1 if args.sample_rate % args.encoder_input_channels == 0 else 2
        if args.encoder_transform == "stft":
            self.encoder_transform = AmplitudeCompressedComplexSTFT(
                n_fft=(args.encoder_input_channels * 2) - subtractor,
                hop_length=args.encoder_input_channels,
                sampling_rate=args.sample_rate,
                alpha=args.stft_alpha,
                beta=args.stft_beta,
                window_fn="hann",
            )
        else:
            raise ValueError(
                f"Unknown encoder transform, not implemented: {args.encoder_transform}"
            )

        if args.decoder_transform == "stft":
            self.decoder_transform = AmplitudeCompressedComplexSTFT(
                n_fft=(args.decoder_input_channels * 2) - subtractor,
                hop_length=args.decoder_input_channels,
                sampling_rate=args.sample_rate,
                alpha=args.stft_alpha,
                beta=args.stft_beta,
                window_fn="hann",
            )
        else:
            raise ValueError(
                f"Unknown decoder transform, not implemented: {args.decoder_transform}"
            )

        self.audiotools_transforms = [
            # transforms.VolumeNorm(["const", -16], prob=1.0),
            transforms.VolumeNorm(["uniform", -50, 1], prob=0.75),
            transforms.RescaleAudio(prob=1.0),
        ]

        self.waveform_transforms = [
            RandomCompression(min_bitrate=8, max_bitrate=32, prob=0.1),
        ]

        self.seq_len = (args.sample_rate // args.encoder_input_channels) + 1

        # print("USING WEIGHTS OF", self.weights)

        self.state = np.random.RandomState()
        self.patch_type = args.patching_type

        self.time_masking_noise_prob = args.time_masking_noise_prob
        self.time_masking_noise_margin = args.time_masking_noise_margin

        self.generate_16khz_decoder = args.generate_16khz_decoder
        # if self.generate_16khz_decoder:
            # self.decoder_16khz_downsampler = torchaudio.transforms.Resample(
            #     orig_freq=self.sample_rate,
            #     new_freq=16000,
            #     # lowpass_filter_width=64,
            #     # rolloff=0.9475937167399596,
            #     # resampling_method="sinc_interp_kaiser",
            #     # beta=14.769656459379492,
            #     #use low quality resampling for speed here
            #     lowpass_filter_width=16,
            #     rolloff=0.85,
            #     resampling_method="sinc_interp_kaiser",
            #     beta=8.555504641634386,
            # )
            # self.decoder_16khz_downsampler = get_resample_object_soxr(self.sample_rate, 16000)

        self.fmax_masking = args.fmax_masking
        if args.random_downsamples:
            from .augs import RandomResampling
            self.waveform_transforms.append(RandomResampling(target_rate=args.sample_rate, min_rate=4000, max_rate=48000, prob=0.2))

    def apply_audiotools_transforms(self, audio_item: AudioSignal):
        for transform in self.audiotools_transforms:
            kwargs = transform.instantiate(signal=audio_item)
            audio_item = transform(audio_item, **kwargs)
        return audio_item

    def apply_waveform_transforms(self, waveform: torch.Tensor):
        for transform in self.waveform_transforms:
            waveform = transform(waveform, self.sample_rate)
        return waveform

    def dataload(
        self,
        path,
        duration,
        sample_rate,
        num_channels,
        loudness_cutoff,
        original_duration,
        original_sample_rate,
    ):
        signal = AudioSignal.salient_excerpt(
            path,
            duration=duration,
            state=self.state,
            loudness_cutoff=loudness_cutoff,
            original_total_duration=original_duration,
            original_sample_rate=original_sample_rate,
        )

        if num_channels == 1:
            signal = signal.to_mono()

        # signal = signal.resample(sample_rate)
        if signal.sample_rate != sample_rate:
            resampler = get_resample_object_soxr(signal.sample_rate, sample_rate)
            signal.audio_data = resampler(signal.audio_data.view(-1)).view(1,1,-1)
            signal.sample_rate = sample_rate

        true_duration = signal.signal_length

        if signal.duration < duration:
            signal = signal.zero_pad_to(int(duration * sample_rate))


        return signal[..., :int(duration * sample_rate)], true_duration

    def __getitem__(self, idx):
        idx = idx % int(self.total_effective_len)
        try:
            # with Timeout(
            #     seconds=self.dataload_timeout,
            #     error_message=f"Dataload timeout for idx {idx}, {self.csv_contents[idx]}",
            # ):
            selected_path = self.csv_contents[idx]

            signal, true_duration = self.dataload(
                selected_path["path"].strip().strip('"'),
                self.sample_duration_seconds,
                self.sample_rate,
                1,
                -50,
                selected_path["duration"],
                selected_path["sample_rate"],
            )
        except Exception as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Error loading audio", e)
            return self.__getitem__(random.randint(0, self.total_effective_len))

        signal = self.apply_audiotools_transforms(signal)
        encoder_signal = self.apply_waveform_transforms(signal.audio_data.clone())
        # Just return the raw audio waveforms

        if (
            true_duration < self.sample_duration_seconds
            and random.random() < self.time_masking_noise_prob
        ):  # make the true duration noisy with a 10% chance but don't exceed the max sample duration
            maximum_noise = (
                min(0, self.sample_duration_seconds - true_duration)
                * self.time_masking_noise_margin
            )
            true_duration += random.uniform(0, maximum_noise)
        decoder_signal = signal.audio_data.squeeze(0).squeeze(0)
        return_dict= {
                "encoder_audio": encoder_signal.squeeze(0).squeeze(0),  # This is already a tensor
                "decoder_audio": decoder_signal,  # This is a tensor
                "true_lengths": true_duration,
                "input_channels": self.encoder_input_channels,
        }
        if self.fmax_masking:
            fmax = -self.decoder_input_channels
            try:
                fmax = estimate_fmax(decoder_signal, self.sample_rate).item() // ((self.sample_rate//2) / self.decoder_input_channels)
                #if nan or inf, set to decoder_input_channels
                if math.isnan(fmax) or math.isinf(fmax) or fmax <= 0 or fmax > self.decoder_input_channels:
                    print(fmax, "Fmax is nan or inf or bullshit, setting to decoder_input_channels, this happened on clip ", selected_path["path"])
                    fmax = -self.decoder_input_channels
                else:
                    fmax = int(fmax)
            except Exception as e:
                print(fmax, "EXCEPTION!! Fmax is nan or inf or bullshit, setting to decoder_input_channels", selected_path["path"], e)
                fmax = -self.decoder_input_channels

            return_dict['fmax'] = fmax# abs(fmax)

        if self.generate_16khz_decoder:
            # return_dict['distill_16khz_decoder_audio'] = self.decoder_16khz_downsampler(decoder_signal)
            return_dict['distill_16khz_decoder_audio'] = get_resample_object_soxr(self.sample_rate, 16000)(decoder_signal)
        # #check all return_dict keys for NaN
        # for k, v in return_dict.items():
        #     if isinstance(v, torch.Tensor):
        #         if torch.isnan(v).any() or torch.isinf(v).any():
        #             print(f"NaN or inf found in {k} for idx {idx}")
        #     elif isinstance(v, float):
        #         if math.isnan(v) or math.isinf(v):
        #             print(f"NaN or inf found in {k} for idx {idx}")
        #     elif isinstance(v, int):
        #         if math.isnan(v) or math.isinf(v):
        #             print(f"NaN or inf found in {k} for idx {idx}")
        return return_dict
        


    def __len__(self):
        return int(self.total_effective_len)  # * 999


def collate_fn(batch):
    # generate mask for pads
    true_length = [b["true_lengths"] for b in batch]
    input_channels = batch[0]["input_channels"]

    additive = 0 if batch[0]["decoder_audio"].shape[-1] % input_channels == 0 else 1

    true_length = [(tl // input_channels) + additive for tl in true_length]
    mask = torch.zeros(
        len(batch),
        (batch[0]["decoder_audio"].shape[-1] // input_channels) + additive,
        requires_grad=False,
    )
    # put 1s until true_length
    for m in range(mask.shape[0]):
        mask[m, : true_length[m]] = 1



    # actual batch collate on just keys with "audio" contained in the string
    audios = torch.utils.data.default_collate(
        [{k: v for k, v in b.items() if "audio" in k} for b in batch]
    )
    # add mask
    audios["time_masks"] = mask
    #if fmax exists
    if "fmax" in batch[0]:
        fmaxes = [b["fmax"] for b in batch]
        #convert to tensor of shape [bsz]
        audios['fmax'] = torch.tensor(fmaxes, dtype=torch.int64)
    return audios


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


def create_dataloader(args: AudioDatasetArgs, seed, rank, timeout=200):
    dataset = AutoencoderDataset(args)

    # --- BEGIN ADDITION ---
    is_distributed = dist.is_available() and dist.is_initialized()
    sampler = None
    shuffle = args.shuffle  # Keep original shuffle intent if not distributed

    if is_distributed:
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()  # Use global rank for sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=args.shuffle,  # Sampler handles shuffling
            seed=seed,  # Ensure consistent shuffle across epochs/resumption
        )
        shuffle = False  # Sampler handles shuffling, so DataLoader shouldn't
        print(f"Rank {global_rank}/{world_size}: Using DistributedSampler.")
        sampler.set_epoch(0)
    # --- END ADDITION ---

    import functools

    init_fn = functools.partial(worker_init_fn, seed=seed, rank=rank)

    # return torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_memory,
    #     shuffle=args.shuffle,
    #     persistent_workers=args.persistent_workers,
    #     prefetch_factor=args.prefetch_factor,
    #     worker_init_fn=init_fn,  # Add this line
    # )
    # import multiprocessing as mp
    # print(f"Rank {global_rank}/{world_size}: Default multiprocessing start method: {mp.get_start_method()}")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        # shuffle=args.shuffle, # <<< Let sampler handle shuffle in DDP
        shuffle=shuffle,  # <<< Use updated shuffle flag
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        worker_init_fn=init_fn,
        sampler=sampler,  # <<< Pass the sampler here
        # Consider drop_last=True if partial batches cause issues with DDP
        drop_last=is_distributed,
        collate_fn=collate_fn,
        timeout=timeout,
        in_order=False,
    )
