import digital_rf as drf
from matplotlib.pylab import sample
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone

from plot_drf import NFFT

drf_path = 'D:\Data\Ham Radio\HAMSci Local Experiments\WWV10_K1FR_20260719_195254_narrowband_drf'
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

# plt.figure(figsize=(10, 6))
# plt.imshow(
#     spectrogram.T,
#     vmin= -50, vmax= 0,
#     aspect='auto',
#     origin='lower',
#     extent=[mdates.date2num(start_time), mdates.date2num(end_time), freqs[0], freqs[-1]],
#     cmap='viridis'
# )
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
# plt.xlabel('Time (UTC)')
# plt.ylabel('Frequency (Hz)')
# plt.colorbar(label='Power (dB)')
# plt.title(f'Spectrogram — channel: {channel}')
# plt.gcf().autofmt_xdate()
# # plt.show()



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