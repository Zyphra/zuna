import random
import torch
import io
import torchaudio
from torchaudio.io import CodecConfig
from functools import cache, partial
import soxr

class RandomCompression:
    def __init__(self, min_bitrate=8, max_bitrate=32, prob=0.2,):
        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        self.prob = prob
        
    def __call__(self, waveform, sample_rate=16000):
        # Handle [bsz, channels, time] format where bsz=1, channels=1
        original_shape = waveform.shape
        waveform_unbatched = waveform.squeeze(0)  # Remove batch dimension -> [channels, time]

        # Randomly apply compression with probability `prob`
        if random.random() > self.prob:
            return waveform
        
        try:
            # Randomly select codec format and bitrate
            codec_format = random.choice(['mp3', 'vorbis'])
            audio_io = io.BytesIO()
            
            torchaudio.save(audio_io, waveform_unbatched, sample_rate, format=codec_format, 
                            compression=CodecConfig(bit_rate=random.randint(self.min_bitrate, self.max_bitrate)))
            audio_io.seek(0)
            
            compressed_waveform, _ = torchaudio.load(audio_io, backend="soundfile")
        except:
            return waveform
        
        #pad if it's shorter or cut if it's longer
        if compressed_waveform.shape[1] < waveform_unbatched.shape[1]:
            compressed_waveform = torch.nn.functional.pad(compressed_waveform, (0, waveform_unbatched.shape[1] - compressed_waveform.shape[1]))
        elif compressed_waveform.shape[1] > waveform_unbatched.shape[1]:
            compressed_waveform = compressed_waveform[:, :waveform_unbatched.shape[1]]
            
        #check if dim 0 is the same
        if compressed_waveform.shape[0] != waveform_unbatched.shape[0]:
            #reduce to 1 channel
            compressed_waveform = compressed_waveform.mean(0, keepdim=True)
            
        # Add batch dimension back to restore [bsz, channels, time] shape
        compressed_waveform = compressed_waveform.unsqueeze(0)
        
        return compressed_waveform
    

@cache
def get_resample_object(original_sample_rate, target_sample_rate):
    return torchaudio.transforms.Resample(
        orig_freq=original_sample_rate,
        new_freq=target_sample_rate,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )
@cache
def get_resample_object_soxr(original_sample_rate, target_sample_rate):
    resampler = partial(
        soxr.resample,
        in_rate = original_sample_rate,
        out_rate = target_sample_rate,
        quality = 'HQ',)
    
    def wrapper(input_tensor):
        return torch.from_numpy(resampler(input_tensor.numpy()))
        
    return wrapper

class RandomResampling:
    def __init__(self, target_rate=48000, min_rate=4000, max_rate=48000, prob=0.2):
        self.target_rate = target_rate
        self.min_rate = min_rate
        self.max_rate = max_rate

        # Create a resampler for each step rate between min_rate and max_rate
        # self.down_resamplers = {rate: torchaudio.transforms.Resample(orig_freq=target_rate, new_freq=rate) for rate in range(min_rate, max_rate+1, 1000)}
        # self.up_resamplers = {rate: [torchaudio.transforms.Resample(orig_freq=rate, new_freq=target_rate,), torchaudio.transforms.Resample(orig_freq=rate, new_freq=target_rate, lowpass_filter_width=1024, rolloff=0.999)] for rate in range(min_rate, max_rate+1, 1000)}
        #just use get_resample_object to populate the dicts


        # self.down_resamplers = {rate: get_resample_object(target_rate, rate) for rate in range(min_rate, max_rate+1, 1000)}
        # self.up_resamplers = {rate: [get_resample_object(rate, target_rate), get_resample_object(rate, target_rate)] for rate in range(min_rate, max_rate+1, 1000)}

        self.down_resamplers = {rate: {'original_sample_rate': target_rate, 'target_sample_rate': rate} 
                                for rate in range(min_rate, max_rate+1, 1000)}
        self.up_resamplers = {rate: [{'original_sample_rate': rate, 'target_sample_rate': target_rate}, 
                                    {'original_sample_rate': rate, 'target_sample_rate': target_rate}] 
                            for rate in range(min_rate, max_rate+1, 1000)}

        self.prob = prob

        
    # def __call__(self, waveform, sample_rate_unused):
    #     # Randomly apply compression with probability `prob`
    #     if random.random() > self.prob:
    #         return waveform
    #     waveform = waveform.squeeze(0)  # Remove batch dimension -> [channels, time]
    #     original_shape = waveform.shape

    #     #convert waveform to float32 if it's not in float32
    #     if waveform.dtype != torch.float32:
    #         waveform = waveform.float()

    #     # Determine the time dimension
    #     if waveform.dim() == 1:
    #         time_dim = 0
    #     elif waveform.dim() == 2:
    #         time_dim = 1
    #     else:
    #         raise ValueError("Unsupported waveform shape.")

    #     original_length = original_shape[time_dim]

    #     # Select a random resample rate between min_rate and max_rate
    #     resample_rate = random.choice(list(self.down_resamplers.keys()))
        
    #     # Resample down
    #     downsampled_waveform = self.down_resamplers[resample_rate](waveform)
        
    #     # Resample back to original picking one of the 2 resamplers at random
    #     upsampled_waveform = random.choice(self.up_resamplers[resample_rate])(downsampled_waveform)

    #     # Adjusting for potential length differences
    #     difference = original_length - upsampled_waveform.shape[time_dim]
    #     if difference > 0:  # The upsampled waveform is shorter than the original
    #         # Reflect-padding
    #         pad_args = (0, difference) if time_dim == 1 else (difference, 0)
    #         upsampled_waveform = torch.nn.functional.pad(upsampled_waveform, pad_args, mode='reflect')
    #     elif difference < 0:  # The upsampled waveform is longer than the original
    #         slice_args = (slice(None), slice(original_length)) if time_dim == 1 else (slice(original_length), )
    #         upsampled_waveform = upsampled_waveform[slice_args]

    #     # Ensure the final shape matches the original shape
    #     upsampled_waveform = upsampled_waveform.view(*original_shape)
    #     return upsampled_waveform.unsqueeze(0)
    def __call__(self, waveform, sample_rate_unused):
        # Randomly apply resampling with probability `prob`
        if random.random() > self.prob:
            return waveform
        
        # Store original shape for later restoration
        original_shape = waveform.shape
        
        # Assert mono audio (all dims except time dim should be 1)
        if len(original_shape) > 1:
            for dim_size in original_shape[:-1]:
                assert dim_size == 1, f"Expected mono audio with batch size 1, got shape {original_shape}"
        
        # Flatten to 1D tensor
        waveform_flat = waveform.view(-1)
        
        # Convert to float32 if needed
        if waveform_flat.dtype != torch.float32:
            waveform_flat = waveform_flat.float()
        
        # Get original length
        original_length = waveform_flat.shape[0]
        
        # Select a random resample rate
        resample_rate = random.choice(list(self.down_resamplers.keys()))
        
        # Resample down and then back up
        downsampled_waveform = get_resample_object_soxr(**self.down_resamplers[resample_rate])(waveform_flat)
        upsampled_waveform = get_resample_object_soxr(**random.choice(self.up_resamplers[resample_rate]))(downsampled_waveform)
        
        # Adjust length to match original
        if upsampled_waveform.shape[0] < original_length:
            # Pad if too short
            upsampled_waveform = torch.nn.functional.pad(
                upsampled_waveform, (0, original_length - upsampled_waveform.shape[0]), mode='reflect'
            )
        elif upsampled_waveform.shape[0] > original_length:
            # Trim if too long
            upsampled_waveform = upsampled_waveform[:original_length]
        
        # Restore to original shape
        return upsampled_waveform.view(*original_shape)