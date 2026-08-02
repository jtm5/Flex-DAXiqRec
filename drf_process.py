import os
import digital_rf as drf
from matplotlib.pylab import sample
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from PyQt5.QtWidgets import QApplication, QFileDialog
from skimage.filters import frangi, threshold_otsu
from skimage.measure import label, regionprops
from scipy.signal import spectrogram, butter, filtfilt, find_peaks
from skimage.exposure import rescale_intensity



# add a PyQt file selection dialong to chose he location and name for the DRF
app = QApplication([])
DEFAULT_DRF_DIR = r"D:\\Data\\Ham Radio\\HAMSci Local Experiments"
initial_dir = DEFAULT_DRF_DIR if os.path.isdir(DEFAULT_DRF_DIR) else os.path.expanduser("~")
drf_path = QFileDialog.getExistingDirectory(None, "Select DRF Directory", initial_dir)

RUN_EXPERIMENTAL_VESSELIZATION = False  # if True, run the experimental multi-vessel tracking code below

do = drf.DigitalRFReader(drf_path)
channel = do.get_channels()[0]

props = do.get_properties(channel)
sample_rate = props['samples_per_second']

# PSWS-style datasets pack several monitored frequencies as subchannels of
# one DRF channel (see psws_drf_subchannels note) -- read_vector() then
# returns a 2D (samples, num_subchannels) array instead of the 1D array our
# own single-frequency captures produce. Every subchannel gets processed
# below (see the loop near the bottom), rather than picking just one.
num_subchannels = int(props.get('num_subchannels', 1))

subchannel_labels = [f"Subchannel {i}" for i in range(num_subchannels)]
if num_subchannels > 1:
    meta_reader = drf.DigitalMetadataReader(os.path.join(drf_path, channel, "metadata"))
    meta_start, meta_end = meta_reader.get_bounds()
    latest_meta = meta_reader.read(meta_end, meta_end)[meta_end]
    center_frequencies = np.asarray(latest_meta['center_frequencies'], dtype=np.float64)
    # Units vary by writer -- our own DRFs store center_frequencies in Hz,
    # some PSWS datasets store it in MHz. Normalize to Hz before labeling.
    center_frequencies_hz = np.where(center_frequencies < 1000, center_frequencies * 1e6, center_frequencies)
    subchannel_labels = [f"Subchannel {i}: {f / 1e6:.4f} MHz" for i, f in enumerate(center_frequencies_hz)]
    print(f"Found {num_subchannels} subchannels (Hz): {center_frequencies_hz} "
          f"(callsign={latest_meta.get('callsign')}, receiver_name={latest_meta.get('receiver_name')})")
else:
    print("Single-subchannel dataset.")

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

# Read once (covers every subchannel), then slice out one column per
# subchannel in the loop below rather than re-reading from disk each time.
data_full = do.read_vector(start_sample, block_size, channel)

sample_rate_f = float(sample_rate)
spec_start_time = datetime.fromtimestamp(start_sample / sample_rate_f, tz=timezone.utc)
spec_end_time = datetime.fromtimestamp((start_sample + block_size) / sample_rate_f, tz=timezone.utc)

NFFT           = 1024          # FFT size for spectrogram / PSD
OVERLAP        = 512           # Overlap between STFT frames
CMAP           = "rainbow"
MAG_AVG_WINDOW = 500            # samples averaged into each plotted amplitude point


def process_subchannel(data, title_suffix):
    """Run the spectrogram/magnitude plot -- and, if enabled, the experimental
    vessel-finding pass -- for one subchannel's data."""

    fig, axes = plt.subplots(
        2, 2, figsize=(10, 8),
        gridspec_kw={"width_ratios": [25, 1]},
        constrained_layout=True,
    )
    ax_spec, ax_cbar = axes[0]
    ax_mag, ax_unused = axes[1]
    ax_mag.sharex(ax_spec)
    fig.delaxes(ax_unused)  # right column only exists to hold the colorbar next to ax_spec

    Pxx, freqs, bins, im = ax_spec.specgram(
        data, NFFT=NFFT, Fs=sample_rate, noverlap=OVERLAP,
        cmap=CMAP, scale="dB", mode="psd",
        xextent=(mdates.date2num(spec_start_time), mdates.date2num(spec_end_time))
    )
    # specgram()'s scale="dB" only affects the *displayed image* -- the Pxx
    # it returns to us is still linear power. Convert here so Pxx actually
    # matches what's on screen (and what "Power (dB)" on the colorbars
    # claims); everything downstream (NaN interpolation, the vessel-section
    # code, and the combined all-subchannels plot) assumes dB-scale values.
    Pxx = 10 * np.log10(Pxx + 1e-12)
    # Read off the color limits specgram() itself already computed and
    # displayed for this image, rather than re-deriving our own from Pxx --
    # matplotlib's autoscale quietly excludes non-finite bins (a -inf from
    # log10(0) on an exactly-zero-power bin) when picking this range, and a
    # plain nanmin/nanmax over our own epsilon-padded Pxx would not exclude
    # those same bins, giving a visibly wider (and mismatched) range.
    clim_vmin, clim_vmax = im.get_clim()
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title(f"Spectrogram\n{drf_path}\n{title_suffix}")
    # cax=ax_cbar (rather than ax=ax_spec) pins the colorbar to its own dedicated
    # grid cell, which only spans the top row -- so its height matches ax_spec
    # alone, while ax_spec and ax_mag stay the same width (same gridspec column).
    fig.colorbar(im, cax=ax_cbar, label="Power (dB)")

    # Magnitude of the raw received samples, on the same time axis as the
    # spectrogram above -- interpolated between the same start/end timestamps
    # (rather than recomputed per-sample) so it lines up with the spectrogram's
    # xextent exactly.
    mag_time_axis = np.linspace(
        mdates.date2num(spec_start_time), mdates.date2num(spec_end_time), len(data)
    )
    # Smooth with a rolling average, then only plot one point per window, rather
    # than a straight every-Nth-sample decimation -- each plotted point reflects
    # a local average instead of landing on one arbitrary raw sample.
    magnitude = np.abs(data)
    smoothed_magnitude = np.convolve(magnitude, np.ones(MAG_AVG_WINDOW) / MAG_AVG_WINDOW, mode='same')
    ax_mag.plot(mag_time_axis[::MAG_AVG_WINDOW], smoothed_magnitude[::MAG_AVG_WINDOW], linewidth=0.7)
    ax_mag.set_ylabel("Amplitude")
    ax_mag.set_title("Magnitude of Raw Data")
    ax_mag.set_xlabel("Time (UTC)")
    ax_mag.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    ax_spec.label_outer()
    ax_mag.label_outer()
    fig.autofmt_xdate()

    plt.show()

    # get_continuous_blocks() doesn't catch every internal dropout -- DigitalRF
    # still fills genuinely missing samples with NaN inside a nominally
    # "continuous" span. A single all-NaN time bin is enough to poison frangi()
    # below: it normalizes by a single scalar gamma = max(...)/2 over the whole
    # image, so one NaN column turns the *entire* ridge map into NaN. Interpolate
    # any NaN time bins away here so downstream analysis sees a clean array.
    nan_cols = np.isnan(Pxx).any(axis=0)
    if nan_cols.any():
        good_cols = ~nan_cols
        print(f"[{title_suffix}] Warning: {nan_cols.sum()} of {Pxx.shape[1]} spectrogram time bins "
              f"were NaN (data dropout); interpolating over them.")
        x = np.arange(Pxx.shape[1])
        for row in range(Pxx.shape[0]):
            Pxx[row, nan_cols] = np.interp(x[nan_cols], x[good_cols], Pxx[row, good_cols])

    if not RUN_EXPERIMENTAL_VESSELIZATION:
        return Pxx, freqs, bins, clim_vmin, clim_vmax

    # Robust color limits (shared by all spectrogram-backed plots below) so a
    # few outlier bins don't wash out the color scale.
    pxx_vmin, pxx_vmax = np.nanpercentile(Pxx, [5, 99])
    if not np.isfinite(pxx_vmin) or not np.isfinite(pxx_vmax) or pxx_vmax <= pxx_vmin:
        pxx_vmin, pxx_vmax = np.nanmin(Pxx), np.nanmax(Pxx)

    #########################################################################
    #  EXPERIMENTAL CODE LOOK FOR "VESSELIZATION" to find possible prop modes
    #########################################################################

    # Scale-space filtering: sigmas match the expected width of the spectrogram ridge
    sigmas = range(1, 5)
    frangi_response = frangi(Pxx, sigmas=sigmas, black_ridges=False)
    percentile = 99
    thresh_val = np.percentile(frangi_response, percentile)
    vessel_mask = frangi_response > thresh_val



    # # Normalize the ridge map for easier peak tracking
    # ridge_map_normalized = (frangi_response - np.min(frangi_response)) / (np.max(frangi_response) - np.min(frangi_response))

    # # Automatic thresholding, or set a manual scalar (e.g., vessel_mask = frangi_response > 0.05)
    # thresh = threshold_otsu(frangi_response)
    # vessel_mask = ridge_map_normalized > thresh
    # add a plot of vessel_mask
    times = bins
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(times, freqs, vessel_mask, shading='none', cmap='rainbow')
    ax.set_title(f"Vessel Mask\n{title_suffix}")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (UTC)")
    fig.colorbar(im, ax=ax, label="Vessel Mask Intensity")
    plt.show()

    # first method of extracting vessel centerlines - no use of argmax()
    # A pixel-connectivity approach (skeletonize, or labeling connected
    # components of vessel_mask) turned out not to work on this data: the two
    # vessels' frequencies pass close enough to each other that their mask
    # regions touch and get merged into a single connected component, so there's
    # no way to tell them apart by connectivity alone -- and skeletonizing the
    # unmerged mask just gives back scattered points since the mask is too
    # fragmented in time (the weaker vessel dips below threshold repeatedly).
    #
    # Instead, track by frequency continuity, the same principle the argmax
    # method below uses for one ridge, generalized to several: find local peaks
    # in each time bin's ridge-response column, then link peaks across time bins
    # into persistent tracks (nearest active track within a max per-step
    # frequency jump; a track survives a short gap -- a brief fade below
    # threshold -- by coasting a few bins before being retired).
    peak_height = np.percentile(frangi_response, 99)
    max_jump_hz = 0.15   # max plausible per-time-bin frequency change for one vessel
    max_coast_bins = 10  # how many consecutive missed bins before a track is retired

    active_tracks = []    # each: {"freqs": {col: freq}, "last_freq": f, "last_col": col}
    finished_tracks = []
    for col in range(len(times)):
        peak_idx, peak_props = find_peaks(frangi_response[:, col], height=peak_height, distance=3)
        peak_freqs = list(freqs[peak_idx])
        unmatched = list(range(len(peak_freqs)))

        for track in active_tracks:
            if not unmatched:
                break
            best_i = min(unmatched, key=lambda i: abs(peak_freqs[i] - track["last_freq"]))
            if abs(peak_freqs[best_i] - track["last_freq"]) <= max_jump_hz:
                track["freqs"][col] = peak_freqs[best_i]
                track["last_freq"] = peak_freqs[best_i]
                track["last_col"] = col
                unmatched.remove(best_i)

        for i in unmatched:
            active_tracks.append({"freqs": {col: peak_freqs[i]}, "last_freq": peak_freqs[i], "last_col": col})

        still_active = []
        for track in active_tracks:
            (finished_tracks if col - track["last_col"] > max_coast_bins else still_active).append(track)
        active_tracks = still_active
    finished_tracks.extend(active_tracks)

    # Separately, look for a persistent near-stationary component (e.g. a direct
    # / line-of-sight path riding at ~0 Hz Doppler): the frame-to-frame linker
    # above is tuned to follow *moving* ridges and only coasts through short
    # gaps, so a signal that's strong in aggregate but repeatedly dips below the
    # peak threshold gets fragmented into many short-lived tracks that
    # individually fail the length filter below. A signal that barely moves in
    # frequency instead shows up as an anomalously tall, narrow spike in the
    # histogram of ALL detected peak frequencies pooled across every time bin --
    # a wandering vessel spreads its detections across many different frequency
    # bins over the course of the recording, a near-motionless one repeatedly
    # lands in the same few bins.
    freq_res = freqs[1] - freqs[0]
    stationary_window_hz = 0.06      # half-width of the search window around a candidate frequency
    stationary_min_fraction = 0.25   # must be present in at least this fraction of time bins

    per_col_peaks = [freqs[find_peaks(frangi_response[:, c], height=peak_height, distance=3)[0]] for c in range(len(times))]
    all_peak_freqs = np.concatenate(per_col_peaks) if per_col_peaks else np.array([])

    hist_edges = np.arange(freqs[0], freqs[-1] + freq_res, freq_res)
    hist_counts, _ = np.histogram(all_peak_freqs, bins=hist_edges)
    window_bins = max(1, int(round(stationary_window_hz / freq_res)))
    smoothed_counts = np.convolve(hist_counts, np.ones(2 * window_bins + 1), mode="same")

    stationary_track = None
    peak_bin = np.argmax(smoothed_counts)
    if smoothed_counts[peak_bin] >= stationary_min_fraction * len(times):
        f0 = hist_edges[peak_bin] + freq_res / 2
        stationary_track = np.full(len(times), np.nan)
        for col, pf in enumerate(per_col_peaks):
            near = pf[np.abs(pf - f0) <= stationary_window_hz]
            if near.size:
                stationary_track[col] = near[np.argmin(np.abs(near - f0))]
        print(f"[{title_suffix}] Found a persistent near-stationary component at {f0:.3f} Hz "
            f"present in {np.sum(~np.isnan(stationary_track))}/{len(times)} time bins.")

    # Keep only tracks that persist across a meaningful fraction of the
    # recording -- short-lived ones are noise peaks, not vessels.
    min_track_len = 0.20 * len(times)
    vessel_track_dicts = [t for t in finished_tracks if len(t["freqs"]) >= min_track_len]
    print(f"[{title_suffix}] Found {len(vessel_track_dicts)} candidate vessel(s) out of {len(finished_tracks)} track(s).")

    # Densify to one array per vessel (NaN outside its detected span) and
    # lightly smooth, same as the argmax path below, but only interpolating
    # gaps *within* each track's own span rather than across its whole length.
    vessel_tracks = []
    for track in vessel_track_dicts:
        cols = np.array(sorted(track["freqs"]))
        vals = np.array([track["freqs"][c] for c in cols])
        span = np.arange(cols[0], cols[-1] + 1)
        interpolated = np.interp(span, cols, vals)
        track_freq = np.full(len(times), np.nan)
        track_freq[span] = interpolated
        vessel_tracks.append(track_freq)

    vessel_labels = [f"Vessel {i + 1}" for i in range(len(vessel_tracks))]
    if stationary_track is not None:
        vessel_tracks.append(stationary_track)
        vessel_labels.append("Stationary path")

    # Plot the extracted vessel centerlines over the spectrogram
    track_colors = ['red', 'cyan', 'lime', 'magenta', 'yellow']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.pcolormesh(times, freqs, Pxx, shading='nearest', cmap='rainbow', alpha=0.4,
                vmin=pxx_vmin, vmax=pxx_vmax)
    for i, track_freq in enumerate(vessel_tracks):
        ax.plot(times, track_freq, '.', color=track_colors[i % len(track_colors)],
                markersize=4, label=vessel_labels[i])
    ax.set_title(f"Vessel Centerlines (multi-vessel peak tracking)\n{title_suffix}")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (UTC)")
    ax.legend(loc="upper right")
    plt.show()


    #  OK, that was attempt to be able to find multiple vessels,
    #   this now uses the argmax() method to track the most prominent ridge in each time bin
    # For each time bin, locate the peak of the ridge map
    #### WARNING: This is a crude ridge tracking method based on the maximum response in the Frangi-filtered map.
    # The argmax() will kill the ability to find multiple ridges within the same time bin.
    tracked_freq_indices = np.argmax(frangi_response, axis=0)
    tracked_frequencies = freqs[tracked_freq_indices]

    # Apply a low-pass Butterworth filter to smooth tracking anomalies
    b, a = butter(3, 0.1)
    smoothed_tracked_frequencies = filtfilt(b, a, tracked_frequencies)


    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(title_suffix)

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
    im1 = axes[1].pcolormesh(times, freqs, frangi_response, shading='none', cmap='rainbow',vmin=pxx_vmin, vmax=pxx_vmax)
    axes[1].set_title("2. Frangi Filter Ridge Map")
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

    return Pxx, freqs, bins, clim_vmin, clim_vmax


def plot_all_subchannels(pxx_list, clim_list, freqs, bins):
    """One combined figure, one spectrogram subplot per subchannel stacked
    vertically (sharing the time axis), for a quick side-by-side comparison
    across frequencies after the individual per-subchannel plots above. Each
    row reuses that subchannel's own specgram() color limits (clim_list) so
    the combined view matches the individual plots' color/intensity profile
    exactly, rather than computing its own scale."""
    n = len(pxx_list)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True, constrained_layout=True)
    if n == 1:
        axes = [axes]  # plt.subplots(1, ...) returns a bare Axes, not an array

    # Same real-UTC time axis the individual spectrograms use via xextent --
    # freqs/bins are identical across subchannels (same NFFT/OVERLAP/Fs/data
    # length every time), so only Pxx differs per row.
    time_axis = np.linspace(
        mdates.date2num(spec_start_time), mdates.date2num(spec_end_time), len(bins)
    )
    for ax, Pxx, (vmin, vmax), label in zip(axes, pxx_list, clim_list, subchannel_labels):
        im = ax.pcolormesh(time_axis, freqs, Pxx, shading='nearest', cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, label="Power (dB)")
        ax.label_outer()

    axes[-1].set_xlabel("Time (UTC)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.suptitle(f"All Subchannels\n{drf_path}")
    fig.autofmt_xdate()
    plt.show()


_pxx_by_subchannel = []
_clim_by_subchannel = []
for _i in range(num_subchannels):
    _data = data_full[:, _i] if data_full.ndim > 1 else data_full
    _pxx, _freqs, _bins, _vmin, _vmax = process_subchannel(_data, subchannel_labels[_i])
    _pxx_by_subchannel.append(_pxx)
    _clim_by_subchannel.append((_vmin, _vmax))

if num_subchannels > 1:
    plot_all_subchannels(_pxx_by_subchannel, _clim_by_subchannel, _freqs, _bins)
