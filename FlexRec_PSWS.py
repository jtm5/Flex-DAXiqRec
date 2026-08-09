#########################################################################################################################
#
#   FlexRec_PSWS.py
#
#   Purpose: capture up to N simultaneous Flex DAX IQ channels (one Flex slice + one DAX IQ
#       audio device per channel), decimate each independently, and interleave them into a
#       SINGLE PSWS-style narrow-band DigitalRF channel with num_subchannels=N -- one column
#       per monitored frequency, matching how real PSWS/grape-style multi-frequency stations
#       structure their DRF output (see README.md -> "PSWS metadata compatibility").
#
#       This is a fork of DAXiqRec_capture_streaming.py, not a replacement for it -- that
#       script is untouched and remains the simple single-frequency/single-subchannel capture
#       path. Use this one only when you actually want N simultaneous frequencies written as
#       PSWS-style subchannels of one DRF channel.
#
#       Runs everything in a single process rather than N separate processes: N DigitalRFWriter
#       objects writing to the same DRF channel isn't supported (and one writer per channel
#       would produce N separate DRF datasets, not one PSWS-style multi-subchannel dataset
#       anyway), so one process has to own the single writer and see all N decimated streams
#       before it can interleave them into a write. Each capture channel still gets its own
#       sd.InputStream/callback/decimator -- the heavy FIR filtering runs inside NumPy/SciPy's
#       C code, which releases the GIL, so N lightweight real-time decimation callbacks sharing
#       one process is expected to be fine, but this hasn't been load-tested against real
#       hardware yet. If it turns out too CPU-tight in practice, the fallback is N separate
#       producer processes feeding this one over a queue -- meaningfully more complex (needs
#       cross-process sample-index alignment), so only worth it if the single-process version
#       actually struggles.
#
#   IMPORTANT -- radio control beyond channel 1 is UNVERIFIED:
#       DAXiqRec_capture_streaming.py's pan/DAX hex stream handles (0x42000000 / 0x40000000)
#       identify a panadapter + DAX IQ stream that already existed on this radio when that
#       script was built -- "dax iq set 1 pan 0x40000000 ..." mutates an EXISTING DAX IQ stream
#       object, it doesn't create a new one. Extending this to N simultaneous channels needs N
#       existing panadapter/DAX IQ stream handles, and there's no way to discover or verify the
#       correct handles for channels 2/3 without live access to the radio/SmartSDR session.
#       CAPTURE_CHANNELS below has placeholder handles for channels 2/3 (following the same hex
#       pattern as channel 1) that almost certainly need correcting against your actual
#       SmartSDR session before the radio-control commands do the right thing -- check the
#       "[RADIO]" responses printed to the console when this runs. The audio
#       capture/decimation/DRF-writing pipeline below does not depend on this part and should
#       work regardless of whether the radio commands are right.
#
#       Recommended first test: trim CAPTURE_CHANNELS down to its first entry only (comment out
#       the other two) and confirm the single-subchannel path works end to end, exactly like
#       DAXiqRec_capture_streaming.py's output, before adding the 2nd and 3rd channels back.
#
#########################################################################################################################

import os
import sys
import time
import datetime
import threading
from collections import deque
from contextlib import ExitStack

import numpy as np
import scipy.signal
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFormLayout, QVBoxLayout,
    QSpinBox, QDialogButtonBox, QLineEdit,
)
from PyQt5.QtGui import QFont
import digital_rf as drf
import sounddevice as sd
from TCP_Flex2 import start_telnet_client


fs = 48000  # audio sample rate per DAX IQ channel

REC_DURATION = 10    # this sets recording duration in seconds
RX_STATION = "K1FR"  # HamSCI/PSWS receiver station callsign -- echoed into metadata
GRID_SQUARE = "FM18kt"  # HamSCI/PSWS receiver station grid square -- echoed into metadata
LAT = 38.8  # receiver station latitude -- echoed into metadata
LON = -77.1  # receiver station longitude -- echoed into metadata
RECEIVER_NAME = "Flex 6700"  # HamSCI/PSWS receiver name -- echoed into metadata
STATION_UUID = "NoneAssigned"  # single station-identity string; used for the DRF dataset uuid_str and echoed into metadata
NARROWBAND_RATE = 10           # HamSCI/PSWS narrow-band output rate (sps) -- same for every subchannel

# One entry per simultaneous capture -- each becomes one subchannel (column)
# of the single PSWS-style DRF channel this script writes. Trim to 1 or 2
# entries for initial testing: everything below (writer subchannel count,
# buffers, dialog rows, radio commands) is driven by len(CAPTURE_CHANNELS).
#
# "pan_stream_id"/"dax_stream_id" must reference panadapter/DAX IQ streams
# that already exist in your SmartSDR session (see the module docstring
# above) -- entries 2 and 3 below are placeholders following channel 1's
# pattern, not verified values.
CAPTURE_CHANNELS = [
    {
        "device_name": "DAX IQ 1",      # substring matched against sd.query_devices() names
        "tx_station": "WWV5",
        "carrier_freq_hz": 5_000_000,
        "flex_slice": 0,
        "dax_channel": 1,
        "pan_stream_id": "0x42000000",   # existing panadapter stream handle for this slice
        "dax_stream_id": "0x40000000",   # existing DAX IQ stream handle for this slice
    },
    {
        "device_name": "DAX IQ 2",
        "tx_station": "WWV10",
        "carrier_freq_hz": 10_000_000,
        "flex_slice": 1,
        "dax_channel": 2,
        "pan_stream_id": "0x42000001",   # PLACEHOLDER -- verify against your SmartSDR session
        "dax_stream_id": "0x40000001",   # PLACEHOLDER -- verify against your SmartSDR session
    },
    {
        "device_name": "DAX IQ 3",
        "tx_station": "WWV15",
        "carrier_freq_hz": 15_000_000,
        "flex_slice": 2,
        "dax_channel": 3,
        "pan_stream_id": "0x42000002",   # PLACEHOLDER -- verify against your SmartSDR session
        "dax_stream_id": "0x40000002",   # PLACEHOLDER -- verify against your SmartSDR session
    },
        {
        "device_name": "DAX IQ 4",
        "tx_station": "WWV20",
        "carrier_freq_hz": 20_000_000,
        "flex_slice": 3,
        "dax_channel": 4,
        "pan_stream_id": "0x42000003",   # PLACEHOLDER -- verify against your SmartSDR session
        "dax_stream_id": "0x40000003",   # PLACEHOLDER -- verify against your SmartSDR session
    },
]


class SettingsDialog(QDialog):
    """One (TX station, center frequency) row per configured capture channel,
    plus the station-identity fields carried over from
    DAXiqRec_capture_streaming.py."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flex PSWS Multi-Frequency Recorder Settings")
        self.setMinimumSize(640, 260 + 40 * len(CAPTURE_CHANNELS))

        font = QFont()
        font.setPointSize(14)
        self.setFont(font)

        form = QFormLayout()
        form.setVerticalSpacing(16)

        self.rx_station_edit = QLineEdit(RX_STATION)
        self.grid_square_edit = QLineEdit(GRID_SQUARE)
        self.lat_edit = QLineEdit(str(LAT))
        self.lon_edit = QLineEdit(str(LON))
        self.receiver_name_edit = QLineEdit(RECEIVER_NAME)
        form.addRow("Receiver Station (callsign):", self.rx_station_edit)
        form.addRow("Grid Square:", self.grid_square_edit)
        form.addRow("Receiver Name:", self.receiver_name_edit)
        form.addRow("Latitude:", self.lat_edit)
        form.addRow("Longitude:", self.lon_edit)

        # One (TX station, frequency) row pair per configured channel.
        self.channel_widgets = []
        for i, chan in enumerate(CAPTURE_CHANNELS):
            tx_edit = QLineEdit(chan["tx_station"])
            freq_spin = QSpinBox()
            freq_spin.setRange(100_000, 30_000_000)
            freq_spin.setValue(chan["carrier_freq_hz"])
            freq_spin.setSuffix(" Hz")
            freq_spin.setSingleStep(1000)
            form.addRow(f"Ch {i + 1} ({chan['device_name']}) TX Station:", tx_edit)
            form.addRow(f"Ch {i + 1} ({chan['device_name']}) Frequency:", freq_spin)
            self.channel_widgets.append((tx_edit, freq_spin))

        self.spin_duration = QSpinBox()
        self.spin_duration.setRange(1, 72)
        self.spin_duration.setValue(REC_DURATION)
        self.spin_duration.setSuffix(" hr")
        form.addRow("Recording Duration", self.spin_duration)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def apply_to_globals(self):
        global REC_DURATION, RX_STATION, GRID_SQUARE, RECEIVER_NAME, LAT, LON
        RX_STATION    = self.rx_station_edit.text().strip()
        GRID_SQUARE   = self.grid_square_edit.text().strip()
        LAT           = float(self.lat_edit.text().strip())
        LON           = float(self.lon_edit.text().strip())
        RECEIVER_NAME = self.receiver_name_edit.text().strip()
        REC_DURATION  = self.spin_duration.value() * 3600
        # CAPTURE_CHANNELS entries are mutated in place -- no separate
        # globals needed for per-channel values.
        for chan, (tx_edit, freq_spin) in zip(CAPTURE_CHANNELS, self.channel_widgets):
            chan["tx_station"] = tx_edit.text().strip()
            chan["carrier_freq_hz"] = freq_spin.value()


RECORDING_DIR = "D:\\Data\\Ham Radio\\HAMSci Local Experiments"
DRF_CHANNEL_NAME = "ch0"
DRF_METADATA_DIRNAME = "metadata"
DRF_SUBDIR_CADENCE_SECS = 3600
DRF_METADATA_FILE_CADENCE_SECS = 3600
NARROWBAND_FILE_CADENCE_MILLISECONDS = 180000  # see DAXiqRec_capture.py for the 3-minute-cadence rationale


class _FIRDecimator:
    """Stateful lowpass-FIR-then-decimate stage. Filter and phase state carry across calls,
    so chunked processing has no discontinuities at chunk boundaries."""

    def __init__(self, factor, input_rate, numtaps):
        self.factor = factor
        self.taps = scipy.signal.firwin(numtaps, cutoff=input_rate / factor / 2, fs=input_rate, window=('kaiser', 8.0))
        self.zi = np.zeros(numtaps - 1, dtype=np.complex128)
        self._phase = 0  # index, within the next chunk, of the next sample to keep

    def process(self, chunk):
        if len(chunk) == 0:
            return chunk
        filtered, self.zi = scipy.signal.lfilter(self.taps, [1.0], chunk, zi=self.zi)
        out = filtered[self._phase::self.factor]
        remaining = (len(chunk) - self._phase) % self.factor
        self._phase = 0 if remaining == 0 else self.factor - remaining
        return out.astype(np.complex64)


class StreamingNarrowbandDecimator:
    """Decimates 48 kHz complex IQ down to NARROWBAND_RATE via a two-stage cascade rather than
    one single-stage filter. A direct 4800:1 decimation needs a ~65000-tap FIR for a clean 5 Hz
    cutoff -- fine for scipy.signal.resample_poly run once after the fact (DAXiqRec_capture.py),
    but far too slow to run per-chunk inside a real-time audio callback. Splitting into two stages
    (60x then 80x) keeps each stage's filter to a few hundred/thousand taps, comfortably real-time.

    One instance per capture channel -- each channel decimates independently, on its own
    PortAudio callback thread, before its output ever reaches the shared DRF writer."""

    def __init__(self, input_rate, output_rate):
        total_factor = input_rate // output_rate
        stage1_factor, stage2_factor = 60, 80
        assert stage1_factor * stage2_factor == total_factor, \
            "stage factors no longer match input_rate/output_rate -- update the cascade"
        stage1_rate = input_rate // stage1_factor
        self.stage1 = _FIRDecimator(stage1_factor, input_rate, numtaps=401)
        self.stage2 = _FIRDecimator(stage2_factor, stage1_rate, numtaps=2001)

    def process(self, chunk):
        return self.stage2.process(self.stage1.process(chunk))


class SampleBuffer:
    """Thread-safe FIFO of complex64 samples. Pushed in variable-size chunks by an audio
    callback thread (one per capture channel, each on its own PortAudio thread); drained in
    arbitrary-size chunks by the main writer loop, which waits until every channel's buffer has
    samples available so it can interleave them into one (n, num_channels) block per DRF write."""

    def __init__(self):
        self._lock = threading.Lock()
        self._chunks = deque()
        self._total = 0

    def push(self, chunk):
        if len(chunk) == 0:
            return
        with self._lock:
            self._chunks.append(chunk)
            self._total += len(chunk)

    def available(self):
        with self._lock:
            return self._total

    def pop(self, n):
        """Pop and return exactly n samples. Caller must ensure available() >= n
        (true in this script -- only the single writer loop thread ever pops,
        and it always computes n as the minimum across all channels first)."""
        with self._lock:
            out = np.empty(n, dtype=np.complex64)
            filled = 0
            while filled < n:
                chunk = self._chunks[0]
                take = min(len(chunk), n - filled)
                out[filled:filled + take] = chunk[:take]
                filled += take
                if take == len(chunk):
                    self._chunks.popleft()
                else:
                    self._chunks[0] = chunk[take:]
            self._total -= n
            return out


def find_input_device(name_substring):
    """Resolve a sounddevice input device index by (partial, case-insensitive) name match
    instead of a hardcoded index -- Windows renumbers device indices on driver
    updates/reconnects/reboots, so a name match is the robust way to find e.g. 'DAX IQ 2'
    regardless of its current index."""
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and name_substring.lower() in dev["name"].lower():
            return idx
    input_devices = "\n".join(
        f"  [{i}] {d['name']}" for i, d in enumerate(devices) if d["max_input_channels"] > 0
    )
    raise RuntimeError(
        f"No input device matching '{name_substring}' found. Available input devices:\n{input_devices}"
    )


# Set up capture parameters
print(sd.query_devices())

currentDateStamp = datetime.datetime.today()

_qt_app = QApplication.instance() or QApplication(sys.argv)
_dlg = SettingsDialog()
if _dlg.exec_() != QDialog.Accepted:
    sys.exit(0)
_dlg.apply_to_globals()

NUM_CHANNELS = len(CAPTURE_CHANNELS)

# Resolve each channel's audio device up front, before touching the radio or
# creating any DRF files -- fail fast if a configured DAX IQ device isn't present.
for _chan in CAPTURE_CHANNELS:
    _chan["device_index"] = find_input_device(_chan["device_name"])
    print(f"  {_chan['device_name']} -> device index {_chan['device_index']}")


#################################################
#################################################
###############################################
# Radio initialization: set frequency/mode and enable DAX on each configured slice before
# recording. See the module docstring above -- the pan/DAX stream handles for channels beyond
# the first are UNVERIFIED placeholders and most likely need correcting against your actual
# SmartSDR session. Watch the "[RADIO]" lines printed to the console for rejected commands.
FLEX_HOST = "10.0.0.252"
FLEX_PORT = 4992

_radio_send, _radio_stop = start_telnet_client(
    host=FLEX_HOST,
    port=FLEX_PORT,
    on_message=lambda line: print(f"[RADIO] {line}"),
)

for _chan in CAPTURE_CHANNELS:
    _freq_mhz = _chan["carrier_freq_hz"] / 1_000_000
    _radio_send(f"C11|display pan s {_chan['pan_stream_id']} center={_freq_mhz}")
    _radio_send(f"c12|dax iq set {_chan['dax_channel']} pan {_chan['dax_stream_id']}  daxiq_rate=48000")
    _radio_send(f"C1|slice tune {_chan['flex_slice']} {_freq_mhz:.6f}")
    _radio_send(f"C2|slice set {_chan['flex_slice']} mode=AM")
    _radio_send(f"C3|slice set {_chan['flex_slice']} dax={_chan['dax_channel']}")

# _radio_stop()


_tx_summary = "-".join(_chan["tx_station"] for _chan in CAPTURE_CHANNELS if _chan["tx_station"])
_rec_prefix = f"{_tx_summary}_{RX_STATION}" if RX_STATION else _tx_summary
_rec_timestamp = currentDateStamp.strftime("%Y%m%d_%H%M%S")
_rec_base = f"{_rec_prefix}_{_rec_timestamp}"
NARROWBAND_DRF_PATH = os.path.join(RECORDING_DIR, f"{_rec_base}_narrowband_drf")

rec_start_time = time.time()
start_global_index = int(round(rec_start_time * NARROWBAND_RATE))

channel_dir = os.path.join(NARROWBAND_DRF_PATH, DRF_CHANNEL_NAME)
metadata_dir = os.path.join(channel_dir, DRF_METADATA_DIRNAME)
os.makedirs(channel_dir, exist_ok=True)
os.makedirs(metadata_dir, exist_ok=True)

drf_writer = drf.DigitalRFWriter(
    channel_dir,
    np.complex64,
    DRF_SUBDIR_CADENCE_SECS,
    NARROWBAND_FILE_CADENCE_MILLISECONDS,
    start_global_index,
    NARROWBAND_RATE,
    1,
    uuid_str=STATION_UUID,
    compression_level=0,
    checksum=False,
    is_complex=True,
    num_subchannels=NUM_CHANNELS,
    is_continuous=True,
    marching_periods=True,
)

# Metadata is written twice, same reasoning/pattern as
# DAXiqRec_capture_streaming.py: an immediate provisional record right here
# (sample_count=0), and a final one with the true count once recording ends
# (finally block below). This guards against a hard process kill (e.g. VS
# Code's debug "Stop" button) that never runs any Python cleanup code at all
# -- see the memory/README notes on that gotcha.
metadata_writer = drf.DigitalMetadataWriter(
    metadata_dir,
    DRF_SUBDIR_CADENCE_SECS,
    DRF_METADATA_FILE_CADENCE_SECS,
    NARROWBAND_RATE,
    1,
    "metadata",
)


def _write_metadata(sample_index, count):
    metadata_writer.write(
        sample_index,
        {
            "callsign": RX_STATION,
            "center_frequencies": np.array(
                [chan["carrier_freq_hz"] for chan in CAPTURE_CHANNELS], dtype=np.float64
            ),
            "grid_square": GRID_SQUARE,
            "lat": np.float64(LAT),
            "long": np.float64(LON),
            "receiver_name": RECEIVER_NAME,
            "uuid_str": STATION_UUID,
            "capture_start_sample": np.int64(start_global_index),
            "sample_rate_hz": np.float64(NARROWBAND_RATE),
            "sample_count": np.int64(count),
        },
    )


_write_metadata(start_global_index, 0)

# One decimator + one sample buffer per channel. Callbacks run on separate
# PortAudio threads (one per InputStream); each just decimates and pushes
# into its own SampleBuffer -- no DRF writing happens on a callback thread,
# so there's no cross-thread contention on drf_writer.
for _chan in CAPTURE_CHANNELS:
    _chan["decimator"] = StreamingNarrowbandDecimator(fs, NARROWBAND_RATE)
    _chan["buffer"] = SampleBuffer()


def _make_callback(chan):
    def _callback(indata, frames, time_info, status):
        if status:
            print(f"[{chan['device_name']}] {status}", file=sys.stderr)
        complex_chunk = (indata[:, 0] + 1j * indata[:, 1]).astype(np.complex64)
        narrow_chunk = chan["decimator"].process(complex_chunk)
        chan["buffer"].push(narrow_chunk)
    return _callback


samples_written = 0

try:
    with ExitStack() as stack:
        for _chan in CAPTURE_CHANNELS:
            stack.enter_context(sd.InputStream(
                samplerate=fs, channels=2, blocksize=fs,
                device=_chan["device_index"], callback=_make_callback(_chan),
            ))

        # Main writer loop: every WRITER_POLL_SECS, drain however many
        # samples are available in *every* channel's buffer (the minimum
        # across all of them, so the written block stays time-aligned
        # across subchannels) and write one combined (n, NUM_CHANNELS)
        # block. time.sleep(), not sd.sleep() -- see
        # DAXiqRec_capture_streaming.py for why sd.sleep() silently defers
        # Ctrl+C until the call would have returned on its own anyway.
        WRITER_POLL_SECS = 0.5
        end_time = time.monotonic() + REC_DURATION
        while time.monotonic() < end_time:
            time.sleep(WRITER_POLL_SECS)
            n = min(chan["buffer"].available() for chan in CAPTURE_CHANNELS)
            if n > 0:
                combined = np.empty((n, NUM_CHANNELS), dtype=np.complex64)
                for i, chan in enumerate(CAPTURE_CHANNELS):
                    combined[:, i] = chan["buffer"].pop(n)
                drf_writer.rf_write(combined)
                samples_written += n
finally:
    # Streams are already closed by this point (ExitStack above unwound on
    # the way here) -- drain whatever's left in every buffer, same
    # minimum-across-channels alignment as the main loop.
    n = min(chan["buffer"].available() for chan in CAPTURE_CHANNELS)
    if n > 0:
        combined = np.empty((n, NUM_CHANNELS), dtype=np.complex64)
        for i, chan in enumerate(CAPTURE_CHANNELS):
            combined[:, i] = chan["buffer"].pop(n)
        drf_writer.rf_write(combined)
        samples_written += n

    drf_writer.close()
    # Final, accurate record -- written at a later sample index than the
    # provisional one above, so it's the one a reader sees as "latest".
    _write_metadata(start_global_index + max(samples_written, 1), samples_written)
    print(f"Saved PSWS-style narrow-band DRF ({samples_written} samples @ {NARROWBAND_RATE} sps, "
          f"{NUM_CHANNELS} subchannels): {NARROWBAND_DRF_PATH}")
