"""
Filtering operations: highpass, notch, resampling.
"""
import numpy as np
import mne
from scipy.signal import find_peaks, medfilt, peak_widths
from typing import List


class Filter:
    """Handles filtering operations."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : ProcessingConfig
            Configuration object
        """
        self.config = config

    def resample(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Resample to target sampling rate.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data

        Returns
        -------
        raw_resampled : mne.io.Raw
            Resampled raw data (modified in place)
        """
        if raw.info['sfreq'] != self.config.target_sfreq:
            raw.resample(self.config.target_sfreq, verbose=False)
        return raw

    def apply_highpass(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply highpass filter.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data

        Returns
        -------
        raw_filtered : mne.io.Raw
            Filtered raw data (modified in place)
        """
        if self.config.apply_highpass_filter:
            raw.filter(l_freq=self.config.hpf_freq, h_freq=None, verbose=False)
        return raw

    def apply_notch(self, raw: mne.io.Raw) -> tuple[mne.io.Raw, List[float]]:
        """
        Auto-detect and apply notch filter for line noise.

        Uses PSD analysis to detect narrow peaks (likely line noise)
        and applies notch filtering at those frequencies.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data

        Returns
        -------
        raw_notched : mne.io.Raw
            Notch-filtered raw data (modified in place)
        notch_freqs : list
            List of frequencies where notch was applied
        """
        if not self.config.apply_notch_filter:
            return raw, []

        good_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False, exclude='bads')
        all_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False, exclude=[])

        if len(good_picks) == 0:
            return raw, []

        sfreq = raw.info['sfreq']
        fmin = 45.0
        fmax = min(sfreq / 2 - 1.0, 250.0)

        # Compute PSD
        # Compute PSD (cap n_fft at signal length; skip if too short for useful spectral analysis)
        n_fft = min(4096, raw.n_times)
        if n_fft < 512:
            return raw, []
        psd = raw.compute_psd(method='welch', fmin=fmin, fmax=fmax, picks=good_picks, n_fft=n_fft)
        freqs = psd.freqs
        P = 10 * np.log10(np.median(psd.get_data(), axis=0))

        # Remove 1/f trend with median filter
        df = np.diff(freqs)[0]
        k = int(np.round(5.0 / df)) | 1
        baseline = medfilt(P, kernel_size=max(k, 3))
        resid = P - baseline

        # Peak detection
        resid_mean = np.mean(resid)
        resid_std = np.std(resid)
        height_thresh = resid_mean + 4.0 * resid_std

        peaks, props = find_peaks(resid, height=height_thresh, prominence=2.0,
                                 width=(1, int(np.round(2.0 / df))))

        # Keep only narrow peaks
        w_hz = peak_widths(resid, peaks, rel_height=0.5)[0] * df
        keep = w_hz <= 2.0
        pk_freqs = freqs[peaks][keep]

        notch_frequencies_applied = []

        if pk_freqs.size:
            detected_freqs = np.unique(np.round(pk_freqs, 2))

            # Ensure minimum 2Hz separation
            if len(detected_freqs) > 1:
                filtered_freqs = [detected_freqs[0]]
                for freq in detected_freqs[1:]:
                    if freq - filtered_freqs[-1] >= 2.0:
                        filtered_freqs.append(freq)
                detected_freqs = np.array(filtered_freqs)

            try:
                raw.notch_filter(freqs=detected_freqs, picks=all_picks,
                               filter_length='auto', phase='zero', verbose=False)
                notch_frequencies_applied = detected_freqs.tolist()
            except ValueError as e:
                if "Stop bands are not sufficiently separated" in str(e):
                    # Fallback: just notch the first (strongest) peak
                    raw.notch_filter(freqs=[detected_freqs[0]], picks=all_picks,
                                   filter_length='auto', phase='zero', verbose=False)
                    notch_frequencies_applied = [detected_freqs[0]]
                else:
                    raise e

        return raw, notch_frequencies_applied

    def apply_reference(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply average reference.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data

        Returns
        -------
        raw_referenced : mne.io.Raw
            Referenced raw data (modified in place)
        """
        if self.config.apply_average_reference:
            raw.set_eeg_reference('average', projection=False, verbose=False)
        return raw
