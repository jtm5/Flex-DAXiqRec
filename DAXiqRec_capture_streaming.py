#########################################################################################################################
#
#   DAXiqRec_capture_streaming.py
#
#   Purpose: record Flex DAX IQ via Windows Sound System and write the HamSCI narrow-band DigitalRF
#       output incrementally as samples arrive, instead of buffering the entire recording in memory and
#       decimating it all at once at the end (see DAXiqRec_capture.py). Memory use stays bounded
#       regardless of recording length, and a crash/kill mid-recording only loses the last few seconds
#       instead of the whole run.
#
#       Trade-offs vs. DAXiqRec_capture.py: no raw/real/AM-demod WAV output (skipped entirely -- use
#       DAXiqRec_capture.py when you actually need those), and no "load an existing recording and
#       reprocess it" mode (this script only does live streaming capture).
#
#########################################################################################################################

import os
import sys
import time
import datetime
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


fs = 48000  # sample rate

REC_DURATION = 10    # this sets recording duration in seconds
TX_STATION = "WWV10"
RX_STATION = ""
GRID_SQUARE = ""
RECEIVER_NAME = ""
STATION_UUID = "NoneAssigned"  # single station-identity string; used for the DRF dataset uuid_str and echoed into metadata
CARRIER_FREQ_HZ = 10_000_000  # actual RF carrier in Hz; written to narrowband DRF metadata
NARROWBAND_RATE = 10          # HamSCI narrow-band output rate (sps)


class SettingsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAX IQ Streaming Recorder Settings")
        self.setMinimumSize(600, 260)

        font = QFont()
        font.setPointSize(14)
        self.setFont(font)

        form = QFormLayout()
        form.setVerticalSpacing(16)

        self.tx_station_edit = QLineEdit(TX_STATION)
        self.rx_station_edit = QLineEdit(RX_STATION)
        self.grid_square_edit = QLineEdit(GRID_SQUARE)
        self.receiver_name_edit = QLineEdit(RECEIVER_NAME)
        form.addRow("Transmitter Station:", self.tx_station_edit)
        form.addRow("Receiver Station (callsign):", self.rx_station_edit)
        form.addRow("Grid Square:", self.grid_square_edit)
        form.addRow("Receiver Name:", self.receiver_name_edit)

        self.spin_center_freq = QSpinBox()
        self.spin_center_freq.setRange(100_000, 30_000_000)
        self.spin_center_freq.setValue(CARRIER_FREQ_HZ)
        self.spin_center_freq.setSuffix(" Hz")
        self.spin_center_freq.setSingleStep(1000)
        form.addRow("Center Frequency:", self.spin_center_freq)

        self.spin_duration = QSpinBox()
        self.spin_duration.setRange(1, 86400)
        self.spin_duration.setValue(REC_DURATION)
        self.spin_duration.setSuffix(" s")
        form.addRow("Recording Duration", self.spin_duration)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def apply_to_globals(self):
        global REC_DURATION, TX_STATION, RX_STATION, CARRIER_FREQ_HZ, GRID_SQUARE, RECEIVER_NAME
        TX_STATION       = self.tx_station_edit.text().strip()
        RX_STATION       = self.rx_station_edit.text().strip()
        CARRIER_FREQ_HZ  = self.spin_center_freq.value()
        REC_DURATION     = self.spin_duration.value()
        GRID_SQUARE      = self.grid_square_edit.text().strip()
        RECEIVER_NAME    = self.receiver_name_edit.text().strip()


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
    (60x then 80x) keeps each stage's filter to a few hundred/thousand taps, comfortably real-time."""

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


# Set up capture parameters
print(sd.query_devices())
sd.default.device = (13, 16)  # 13 is DAX I/Q 1 input, 16 is Speakers output

currentDateStamp = datetime.datetime.today()

_qt_app = QApplication.instance() or QApplication(sys.argv)
_dlg = SettingsDialog()
if _dlg.exec_() != QDialog.Accepted:
    sys.exit(0)
_dlg.apply_to_globals()


#################################################
#################################################
###############################################
# Radio initialization: set frequency/mode and enable DAX before recording.
FLEX_HOST = "10.0.0.252"
FLEX_PORT = 4992
FLEX_SLICE = 0

_radio_send, _radio_stop = start_telnet_client(
    host=FLEX_HOST,
    port=FLEX_PORT,
    on_message=lambda line: print(f"[RADIO] {line}"),
)

_freq_mhz = CARRIER_FREQ_HZ / 1_000_000
_radio_send(f"C11|display pan s 0x42000000 center={_freq_mhz}\r\n")
_radio_send("c12|dax iq set 1 pan 0x40000000  daxiq_rate=48000\r\n")
_radio_send(f"C1|slice tune {FLEX_SLICE} {_freq_mhz:.6f}")
_radio_send(f"C2|slice set {FLEX_SLICE} mode=AM")
_radio_send(f"C3|slice set {FLEX_SLICE} dax=1")

# _radio_stop()


_rec_prefix = f"{TX_STATION}_{RX_STATION}" if RX_STATION else TX_STATION
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
    num_subchannels=1,
    is_continuous=True,
    marching_periods=True,
)

# Metadata is written twice: an immediate provisional record (sample_count=0)
# right here, and a final one with the true count once the recording ends
# (see the finally block below). This exists because VS Code's debug "Stop"
# button (and other hard kills) terminates the process directly -- Python
# never runs any try/finally in that case, so the finally-block write never
# executes. The provisional record guarantees dmd_properties.h5 and at least
# one valid entry exist from the very start, so the dataset stays readable
# even if the run gets killed outright. A Ctrl+C in the actual console
# window (not the IDE Stop button) still raises a catchable
# KeyboardInterrupt and gets the accurate final write via finally.
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
            "center_frequencies": np.array([CARRIER_FREQ_HZ], dtype=np.float64),
            "grid_square": GRID_SQUARE,
            "lat": np.float64(38.8),
            "long": np.float64(-77.1),
            "receiver_name": RECEIVER_NAME,
            "uuid_str": STATION_UUID,
            "capture_start_sample": np.int64(start_global_index),
            "sample_rate_hz": np.float64(NARROWBAND_RATE),
            "sample_count": np.int64(count),
        },
    )


_write_metadata(start_global_index, 0)

decimator = StreamingNarrowbandDecimator(fs, NARROWBAND_RATE)
samples_written = 0


def _callback(indata, frames, time_info, status):
    global samples_written
    if status:
        print(status, file=sys.stderr)
    complex_chunk = (indata[:, 0] + 1j * indata[:, 1]).astype(np.complex64)
    narrow_chunk = decimator.process(complex_chunk)
    if len(narrow_chunk):
        drf_writer.rf_write(narrow_chunk.reshape(-1, 1))
        samples_written += len(narrow_chunk)


try:
    with sd.InputStream(samplerate=fs, channels=2, blocksize=fs, callback=_callback):
        sd.sleep(int(REC_DURATION * 1000))
finally:
    drf_writer.close()
    # Final, accurate record -- written at a later sample index than the
    # provisional one above, so it's the one a reader sees as "latest".
    _write_metadata(start_global_index + max(samples_written, 1), samples_written)
    print(f"Saved narrow-band DRF ({samples_written} samples @ {NARROWBAND_RATE} sps): {NARROWBAND_DRF_PATH}")
