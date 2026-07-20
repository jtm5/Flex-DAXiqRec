import digital_rf as drf
from matplotlib.pylab import sample
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from skimage.filters import frangi
from scipy.signal import spectrogram, butter, filtfilt



drf_path = 'D:\\Data\\Ham Radio\\HAMSci Local Experiments\\WWV10_K1FR_20260719_195254_narrowband_drf'
do = drf.DigitalRFReader(drf_path)
channel = do.get_channels()[0]

props = do.get_properties(channel)
sample_rate = props['samples_per_second']

start_sample, end_sample = do.get_bounds(channel)

# get_bounds() reports the cadence-padded file extent, not the true recorded
# span (continuous-mode files are pre-sized to a full cadence period, and the
# recording can start partway through one too). get_continuous_blocks() finds
# the actual written sample range instead -- this is plain DigitalRFReader
# API, so it works the same on our own DRFs and on real HamSCI DRFs (which
# don't carry our old custom capture_start_sample/sample_count metadata).
blocks = do.get_continuous_blocks(start_sample, end_sample, channel)
start_sample = next(iter(blocks))
recorded_sample_count = int(blocks[start_sample])

# How much data to look at, and FFT size
fft_size = 1024 #1024
available_ffts = recorded_sample_count // fft_size
n_ffts = min(7200, available_ffts)        # how many time-slices to compute, capped at what was recorded
block_size = fft_size * n_ffts    # total samples to read for this view

data = do.read_vector(start_sample, block_size, channel)

# Reshape into (n_ffts, fft_size) and apply a window + FFT per slice
window = np.hanning(fft_size)
data_reshaped = data[:n_ffts * fft_size].reshape(n_ffts, fft_size)

spectrum = np.fft.fftshift(np.fft.fft(data_reshaped * window, axis=1), axes=1)
spectrogram = 20 * np.log10(np.abs(spectrum) + 1e-12)  # dB scale

# Plot
freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1/sample_rate))

# DRF sample indices are referenced to the Unix epoch via sample_rate, so
# convert the start/end sample of this view to real UTC timestamps for the
# x-axis instead of relative seconds.
sample_rate_f = float(sample_rate)
start_time = datetime.fromtimestamp(start_sample / sample_rate_f, tz=timezone.utc)
end_time = datetime.fromtimestamp((start_sample + n_ffts * fft_size) / sample_rate_f, tz=timezone.utc)

NFFT          = 1024          # FFT size for spectrogram / PSD
OVERLAP       = 512           # Overlap between STFT frames
MAX_SAMPLES   = 10_000_000    # Cap for memory safety (~10 M samples)
PLOT_IQ_LEN   = 10_000        # Samples shown in time-domain / constellation
CMAP          = "rainbow"


spec_start_time = datetime.fromtimestamp(start_sample / sample_rate_f, tz=timezone.utc)
spec_end_time = datetime.fromtimestamp((start_sample + len(data)) / sample_rate_f, tz=timezone.utc)
plt.figure(figsize=(10, 6))
Pxx, freqs, bins, im = plt.specgram(
    data, NFFT=NFFT, Fs=sample_rate, noverlap=OVERLAP,
    cmap=CMAP, scale="dB", mode="psd",
    xextent=(mdates.date2num(spec_start_time), mdates.date2num(spec_end_time))
)

# get_continuous_blocks() doesn't catch every internal dropout -- DigitalRF
# still fills genuinely missing samples with NaN inside a nominally
# "continuous" span. A single all-NaN time bin is enough to poison frangi()
# below: it normalizes by a single scalar gamma = max(...)/2 over the whole
# image, so one NaN column turns the *entire* ridge map into NaN. Interpolate
# any NaN time bins away here so downstream analysis sees a clean array.
nan_cols = np.isnan(Pxx).any(axis=0)
if nan_cols.any():
    good_cols = ~nan_cols
    print(f"Warning: {nan_cols.sum()} of {Pxx.shape[1]} spectrogram time bins "
          f"were NaN (data dropout); interpolating over them.")
    x = np.arange(Pxx.shape[1])
    for row in range(Pxx.shape[0]):
        Pxx[row, nan_cols] = np.interp(x[nan_cols], x[good_cols], Pxx[row, good_cols])

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gcf().autofmt_xdate()
plt.xlabel("Time (UTC)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram\n" +drf_path)
# fix y-tick labels to show real frequency
yticks = plt.gca().get_yticks()
# ax.set_yticklabels([f"{(y + freq_center)/1e6:.4f}" for y in yticks])
plt.colorbar(im, label="Power (dB)")
plt.show()


#######################################################################
#  EXPERIMENTAL CODE
#################################################################

# ==========================================
# 3. Hessian/Frangi Ridge Detection
# ==========================================
# Scale-space filtering: sigmas match the expected width of the spectrogram ridge
sigmas = range(1, 4) 
ridge_map = frangi(Pxx, sigmas=sigmas, black_ridges=False)
# do a plot of the ridge map for visual inspection
# plt.figure(figsize=(10, 6))
# plt.imshow(ridge_map, aspect='auto', origin='lower', cmap='rainbow')
# plt.colorbar(label="Ridge Intensity")
# plt.title("Ridge Map (Normalized)")
# plt.xlabel("Time Bin")
# plt.ylabel("Frequency Bin")
# plt.show()

# Normalize the ridge map for easier peak tracking
ridge_map_normalized = (ridge_map - np.min(ridge_map)) / (np.max(ridge_map) - np.min(ridge_map))

# ==========================================
# 4. Extract and Track the Ridge Path
# ==========================================
# For each time bin, locate the peak of the ridge map
tracked_freq_indices = np.argmax(ridge_map_normalized, axis=0)
tracked_frequencies = freqs[tracked_freq_indices]

# Optional: Apply a low-pass Butterworth filter to smooth tracking anomalies
b, a = butter(3, 0.1)
smoothed_tracked_frequencies = filtfilt(b, a, tracked_frequencies)

# ==========================================
# 5. Visualizing the Process
# ==========================================
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Use the spectrogram time-bin centers returned by specgram.
times = bins

# Use robust color limits to avoid a washed-out spectrogram from outliers.
pxx_vmin, pxx_vmax = np.nanpercentile(Pxx, [5, 99])
if not np.isfinite(pxx_vmin) or not np.isfinite(pxx_vmax) or pxx_vmax <= pxx_vmin:
    pxx_vmin, pxx_vmax = np.nanmin(Pxx), np.nanmax(Pxx)

# Plot Raw Spectrogram
im0 = axes[0].pcolormesh(
    times,
    freqs,
    Pxx,
    shading='nearest',
    cmap='rainbow',
    vmin=pxx_vmin,
    vmax=pxx_vmax,
)
axes[0].set_title("1. Raw Spectrogram ")
axes[0].set_ylabel("Frequency (Hz)")
fig.colorbar(im0, ax=axes[0], label="Power (dB)")

# Plot Isolated Ridge Map
im1 = axes[1].pcolormesh(times, freqs, ridge_map_normalized, shading='none', cmap='rainbow',vmin=pxx_vmin, vmax=pxx_vmax)
axes[1].set_title("2. Frangi Filter Ridge Map (Background Noise Removed)")
axes[1].set_ylabel("Frequency (Hz)")
fig.colorbar(im1, ax=axes[1], label="Ridge Intensity")

# Plot Final Tracked Route over original spectrogram
axes[2].pcolormesh(
    times,
    freqs,
    Pxx,
    shading='none',
    cmap='rainbow',
    alpha=0.5,
    vmin=pxx_vmin,
    vmax=pxx_vmax,
)
axes[2].plot(times, tracked_frequencies, '.', color='red', alpha=0.5, label='Raw Ridge Maxima')
axes[2].plot(times, smoothed_tracked_frequencies, color='cyan', linewidth=2.5, label='Smoothed Ridge Path')
axes[2].set_title("3. Extracted Doppler Shift Path")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Frequency (Hz)")
axes[2].legend(loc="upper right")

for ax in axes:
	ax.set_ylim(-2, 2)

plt.tight_layout()
plt.show()