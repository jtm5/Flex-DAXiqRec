"""
Plot Digital RF (DRF) narrowband data.
Usage:
    python plot_drf.py /path/to/CHU7_K1FR_20260621_192006_narrowband_drf

Produces four plots:
    1. Spectrogram (time vs frequency)
    2. Power spectral density (PSD)
    3. Time-domain I/Q waveform (first 10k samples)
    4. I/Q constellation
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone

# ── Try to import digital_rf; fall back gracefully ──────────────────────────
try:
    import digital_rf as drf
    HAS_DRF = True
except ImportError:
    HAS_DRF = False
    print("[WARN] digital_rf not installed.  Install with:  pip install digital-rf")
    print("       Falling back to raw HDF5 read via h5py.\n")
    import h5py


# ── Config ───────────────────────────────────────────────────────────────────
NFFT          = 1024          # FFT size for spectrogram / PSD
OVERLAP       = 512           # Overlap between STFT frames
MAX_SAMPLES   = 10_000_000    # Cap for memory safety (~10 M samples)
PLOT_IQ_LEN   = 10_000        # Samples shown in time-domain / constellation
CMAP          = "viridis"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_via_digital_rf(data_dir: str):
    """Load samples using the official digital_rf library."""
    dro = drf.DigitalRFReader(data_dir)
    channels = dro.get_channels()
    if not channels:
        raise RuntimeError("No channels found in DRF directory.")
    ch = channels[0]
    print(f"  Channel      : {ch}")

    props = dro.get_properties(ch)
    fs    = props["samples_per_second"]
    print(f"  Sample rate  : {fs/1e3:.3f} kHz")

    bounds = dro.get_bounds(ch)
    start, end = bounds
    n_avail = end - start + 1
    n_read  = min(n_avail, MAX_SAMPLES)
    print(f"  Samples avail: {n_avail:,}  (reading first {n_read:,})")

    raw = dro.read_vector(start, n_read, ch)
    # digital_rf returns structured or complex array
    if raw.dtype.names:
        samples = raw["r"].astype(np.float32) + 1j * raw["i"].astype(np.float32)
    else:
        samples = raw.astype(np.complex64)

    # UTC start time (digital_rf uses samples since unix epoch)
    t0_unix = start / float(fs)
    t0_utc  = datetime.fromtimestamp(t0_unix, tz=timezone.utc)
    print(f"  Start (UTC)  : {t0_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    return samples, float(fs), t0_utc, ch


def load_via_h5py(data_dir: str):
    """
    Fallback: walk the DRF directory tree and read HDF5 files directly.
    DRF stores data in  <channel>/<YYYY-MM-DDTHH>/<unix_second>@<offset>.h5
    """
    h5_files = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if f.endswith(".h5") and not f.startswith("drf_properties"):
                h5_files.append(os.path.join(root, f))

    if not h5_files:
        raise FileNotFoundError(f"No HDF5 data files found under {data_dir}")

    print(f"  Found {len(h5_files)} HDF5 file(s).  Reading …")

    chunks, fs, t0_utc, ch = [], None, None, "unknown"
    total = 0
    for fpath in h5_files:
        if total >= MAX_SAMPLES:
            break
        with h5py.File(fpath, "r") as hf:
            # Grab sample rate from properties if present
            if fs is None:
                for key in ("samples_per_second", "sample_rate", "fs"):
                    if key in hf.attrs:
                        fs = float(hf.attrs[key])
                        break
                # Some DRF stores metadata in a sub-group
                if fs is None and "rf_data_offsets" in hf:
                    pass  # will try dataset attrs below

            # The data dataset is usually named "rf_data"
            ds_name = "rf_data" if "rf_data" in hf else list(hf.keys())[0]
            ds = hf[ds_name]

            if fs is None:
                for key in ("samples_per_second", "sample_rate"):
                    if key in ds.attrs:
                        fs = float(ds.attrs[key])
                        break

            # Recover start time from filename (unix seconds embedded in name)
            basename = os.path.basename(fpath)          # e.g. 1750000000@0.h5
            try:
                unix_sec = float(basename.split("@")[0])
                if t0_utc is None:
                    t0_utc = datetime.fromtimestamp(unix_sec, tz=timezone.utc)
            except ValueError:
                pass

            raw = ds[:]
            want = MAX_SAMPLES - total
            raw  = raw[:want]

            if raw.dtype.names:                         # structured r/i
                c = raw["r"].astype(np.float32) + 1j * raw["i"].astype(np.float32)
            elif np.issubdtype(raw.dtype, np.complexfloating):
                c = raw.astype(np.complex64)
            else:
                # interleaved int16 → complex
                flat = raw.flatten().astype(np.float32)
                c    = flat[0::2] + 1j * flat[1::2]

            chunks.append(c)
            total += len(c)

    samples = np.concatenate(chunks)
    if fs is None:
        fs = 250_000          # common default for narrowband SDR
        print(f"  [WARN] Sample rate not found in metadata; assuming {fs/1e3:.1f} kHz")
    if t0_utc is None:
        t0_utc = datetime.now(tz=timezone.utc)

    print(f"  Sample rate  : {fs/1e3:.3f} kHz")
    print(f"  Start (UTC)  : {t0_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Samples read : {len(samples):,}")
    return samples, fs, t0_utc, ch


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_drf(samples: np.ndarray, fs: float, t0_utc: datetime, channel: str,
             data_dir: str):

    duration_s = len(samples) / fs
    t_axis     = np.arange(len(samples)) / fs          # seconds from start
    freq_center = 0.0                                   # assume baseband

    # Try to parse centre frequency from folder name  (e.g. CHU7 → 7.335 MHz)
    dirname = os.path.basename(data_dir.rstrip("/\\"))
    if "CHU7" in dirname.upper():
        freq_center = 7.335e6
    elif "CHU3" in dirname.upper():
        freq_center = 3.330e6
    elif "CHU14" in dirname.upper():
        freq_center = 14.670e6

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"DRF · {dirname}\n"
        f"Channel: {channel}   |   Fs = {fs/1e3:.3f} kHz   |   "
        f"Start: {t0_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        fontsize=11
    )

    # ── 1. Spectrogram ───────────────────────────────────────────────────────
    ax = axes[0, 0]
    # Downsample for the spectrogram if very long
    
#########################################################################################
    # samples = samples * 100.0
#####################################################################################


    spec_samples = samples
    if len(samples) > 2_000_000:
        stride       = len(samples) // 2_000_000
        spec_samples = samples[::stride]
        fs_spec      = fs / stride
    else:
        fs_spec      = fs

    Pxx, freqs, bins, im = ax.specgram(
        spec_samples, NFFT=NFFT, Fs=fs_spec, noverlap=OVERLAP,
        cmap=CMAP, scale="dB", mode="psd"
    )
    freqs_mhz = (freqs + freq_center) / 1e6
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (MHz)" if freq_center else "Frequency (Hz)")
    ax.set_title("Spectrogram")
    # fix y-tick labels to show real frequency
    yticks = ax.get_yticks()
    ax.set_yticklabels([f"{(y + freq_center)/1e6:.4f}" for y in yticks])
    plt.colorbar(im, ax=ax, label="Power (dB)")

    # ── 2. Power Spectral Density ────────────────────────────────────────────
    ax = axes[0, 1]
    # Welch PSD over at most 1 M samples
    psd_samples = samples[:1_000_000]
    # Drop NaN/inf samples (e.g. DRF cadence/day padding) -- a single invalid
    # sample poisons the FFT for its whole segment, which then poisons the
    # plain (non-nan-aware) mean across every frequency bin.
    psd_samples = psd_samples[np.isfinite(psd_samples)]
    from numpy.fft import fftshift, fft
    n_seg  = NFFT
    if len(psd_samples) == 0:
        ax.text(0.5, 0.5, "No valid (non-NaN) samples in this window",
                 ha="center", va="center", transform=ax.transAxes)
    else:
        n_segs = len(psd_samples) // n_seg
        if n_segs < 1:
            n_segs = 1
            n_seg  = len(psd_samples)
        reshaped = psd_samples[: n_segs * n_seg].reshape(n_segs, n_seg)
        window   = np.hanning(n_seg)
        Pxx_2d   = np.abs(fftshift(fft(reshaped * window, axis=1), axes=1)) ** 2
        psd_avg  = 10 * np.log10(Pxx_2d.mean(axis=0) + 1e-20)
        f_bins   = (np.fft.fftshift(np.fft.fftfreq(n_seg, d=1/fs)) + freq_center) / 1e6
        ax.plot(f_bins, psd_avg, lw=0.8, color="steelblue")
    ax.set_xlabel("Frequency (MHz)" if freq_center else "Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title("Power Spectral Density (Welch)")
    ax.grid(True, alpha=0.3)

    # ── 3. Time-domain I/Q ───────────────────────────────────────────────────
    ax = axes[1, 0]
    n_td = min(PLOT_IQ_LEN, len(samples))
    t_td = np.arange(n_td) / fs * 1e3               # ms
    ax.plot(t_td, samples[:n_td].real, lw=0.6, label="I (real)", alpha=0.8)
    ax.plot(t_td, samples[:n_td].imag, lw=0.6, label="Q (imag)", alpha=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (ADC counts)")
    ax.set_title(f"I/Q Time Domain  (first {n_td:,} samples)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 4. I/Q Constellation ────────────────────────────────────────────────
    ax = axes[1, 1]
    n_con = min(50_000, len(samples))
    ax.scatter(samples[:n_con].real, samples[:n_con].imag,
               s=0.5, alpha=0.3, color="steelblue")
    ax.set_xlabel("I (real)")
    ax.set_ylabel("Q (imag)")
    ax.set_title(f"I/Q Constellation  (first {n_con:,} samples)")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(data_dir.rstrip("/\\")),
                           dirname + "_plot.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_png}")
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        data_dir = "D:\\Data\\Ham Radio\\HAMSci Local Experiments\\HF9930_K1FR_20260711_133801_narrowband_drf"
        print(f"No path supplied; using default:\n  {data_dir}\n")
    else:
        data_dir = sys.argv[1].rstrip("/\\")

    if not os.path.isdir(data_dir):
        sys.exit(f"[ERROR] Directory not found: {data_dir}")

    print(f"Loading DRF data from:\n  {data_dir}\n")

    if HAS_DRF:
        samples, fs, t0_utc, ch = load_via_digital_rf(data_dir)
    else:
        samples, fs, t0_utc, ch = load_via_h5py(data_dir)

    print(f"\nPlotting {len(samples):,} samples …")
    plot_drf(samples, fs, t0_utc, ch, data_dir)


if __name__ == "__main__":
    main()
