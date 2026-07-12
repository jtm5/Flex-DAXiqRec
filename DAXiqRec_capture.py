#########################################################################################################################
#
#   DAXiqRec_capture.py
#
#   Purpose: record Flex DAX IQ via Windows Sound System, saving the raw capture as WAV and
#       producing a HamSCI narrow-band DigitalRF output.
#       Replaces DAXiqRec_record.py: that version ran every chunk of the recording through an FFT-based
#       overlap-add lowpass filter (sized for SSB/CW audio bandwidth) before decimating, which existed to
#       support live demodulated playback. It made a 24-hour run take hours just to produce the narrow-band
#       DRF. The narrow-band DRF only needs to survive a decimation to 10 sps (5 Hz Nyquist), and
#       scipy.signal.resample_poly already provides correctly-sized anti-aliasing for that on its own, so the
#       OVA filter step is gone -- decimation runs directly on the raw IQ.
#
#########################################################################################################################

import os
import sys
import time
import datetime
from tracemalloc import start
import scipy.signal
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFormLayout, QVBoxLayout, QHBoxLayout,
    QCheckBox, QSpinBox, QDialogButtonBox, QLabel, QLineEdit,
    QRadioButton, QButtonGroup, QWidget,
)
from PyQt5.QtGui import QFont
import digital_rf as drf
import sounddevice as sd
import soundfile as sf
import numpy as np  # Make sure NumPy is loaded before it is used in the callback
assert np  # avoid "imported but unused" message (W0611)


fs = 48000  # sample rate

DO_RECORD = False  # set to True to do the recording, False to read in an existing recording and process it
REC_DURATION = 10    # this sets recording duration in seconds
PLAY_RECORDING = True  # set to True to play the recording after processing, False to skip playback
DEMOD_AM = False
DEMOD_SSB_CW = True
TX_STATION = "WWV10"
RX_STATION = ""
CARRIER_FREQ_HZ = 10_000_000  # actual RF carrier in Hz; written to narrowband DRF metadata
NARROWBAND_RATE = 10          # HamSCI narrow-band output rate (sps)
from TCP_Flex2 import start_telnet_client


class SettingsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAX IQ Recorder Settings")
        self.setMinimumSize(600, 400)

        font = QFont()
        font.setPointSize(14)
        self.setFont(font)

        form = QFormLayout()
        form.setVerticalSpacing(16)

        self.tx_station_edit = QLineEdit(TX_STATION)
        self.rx_station_edit = QLineEdit(RX_STATION)
        form.addRow("Transmitter Station:", self.tx_station_edit)
        form.addRow("Receiver Station:", self.rx_station_edit)

        self.spin_center_freq = QSpinBox()
        self.spin_center_freq.setRange(100_000, 30_000_000)
        self.spin_center_freq.setValue(CARRIER_FREQ_HZ)
        self.spin_center_freq.setSuffix(" Hz")
        self.spin_center_freq.setSingleStep(1000)
        form.addRow("Center Frequency:", self.spin_center_freq)

        self.cb_record = QCheckBox()
        self.cb_record.setChecked(DO_RECORD)
        form.addRow("Record (uncheck = load existing)", self.cb_record)

        self.spin_duration = QSpinBox()
        self.spin_duration.setRange(1, 86400)
        self.spin_duration.setValue(REC_DURATION)
        self.spin_duration.setSuffix(" s")
        form.addRow("Recording Duration", self.spin_duration)

        self.cb_play = QCheckBox()
        self.cb_play.setChecked(PLAY_RECORDING)
        form.addRow("Play Recording After Processing", self.cb_play)

        self.demod_group = QButtonGroup(self)
        self.rb_demod_none = QRadioButton("None")
        self.rb_demod_am = QRadioButton("AM")
        self.rb_demod_ssb = QRadioButton("SSB/CW")
        self.demod_group.addButton(self.rb_demod_none)
        self.demod_group.addButton(self.rb_demod_am)
        self.demod_group.addButton(self.rb_demod_ssb)
        if DEMOD_AM:
            self.rb_demod_am.setChecked(True)
        elif DEMOD_SSB_CW:
            self.rb_demod_ssb.setChecked(True)
        else:
            self.rb_demod_none.setChecked(True)
        demod_widget = QWidget()
        demod_layout = QHBoxLayout(demod_widget)
        demod_layout.setContentsMargins(0, 0, 0, 0)
        demod_layout.addWidget(self.rb_demod_none)
        demod_layout.addWidget(self.rb_demod_am)
        demod_layout.addWidget(self.rb_demod_ssb)
        form.addRow("Demodulation", demod_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def apply_to_globals(self):
        global DO_RECORD, REC_DURATION, PLAY_RECORDING
        global DEMOD_AM, DEMOD_SSB_CW
        global TX_STATION, RX_STATION, CARRIER_FREQ_HZ
        TX_STATION       = self.tx_station_edit.text().strip()
        RX_STATION       = self.rx_station_edit.text().strip()
        CARRIER_FREQ_HZ  = self.spin_center_freq.value()
        DO_RECORD        = self.cb_record.isChecked()
        REC_DURATION     = self.spin_duration.value()
        PLAY_RECORDING   = self.cb_play.isChecked()
        DEMOD_AM         = self.rb_demod_am.isChecked()
        DEMOD_SSB_CW     = self.rb_demod_ssb.isChecked()


RECORDING_DIR = "D:\\Data\\Ham Radio\\HAMSci Local Experiments"
DRF_CHANNEL_NAME = "ch0"
DRF_METADATA_DIRNAME = "metadata"
DRF_SUBDIR_CADENCE_SECS = 3600
DRF_METADATA_FILE_CADENCE_SECS = 3600
# At 10 sps, a 1000 ms file cadence creates one HDF5 file per second (only 10
# samples each), and per-file HDF5 overhead dominates the dataset size. Match
# the real HamSCI narrowband convention instead: 180000 ms (3 min) per file,
# 1800 samples each. Continuous-mode writes always materialize a full file per
# cadence period, so any recording that doesn't start/end exactly on a 3-minute
# boundary will have up to one file's worth of zero/NaN-filled padding at each
# end -- negligible against a 24-hour run. Use DigitalRFReader.get_continuous_blocks()
# rather than get_bounds() when you need the true recorded extent of a short
# test recording.
NARROWBAND_FILE_CADENCE_MILLISECONDS = 180000


def write_narrowband_drf(file_path, narrow_iq, sample_rate, center_freq_hz, tx_station="", rx_station="", capture_start_time=None):
    """Write decimated narrow-band IQ as a HamSCI-compatible DigitalRF dataset (10 sps, complex64)."""
    if capture_start_time is None:
        capture_start_time = time.time()
    complex_iq = np.asarray(narrow_iq, dtype=np.complex64).reshape((-1, 1))

    if os.path.exists(file_path):
        for root, dirs, files in os.walk(file_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(file_path)

    channel_dir = os.path.join(file_path, DRF_CHANNEL_NAME)
    metadata_dir = os.path.join(channel_dir, DRF_METADATA_DIRNAME)
    os.makedirs(channel_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    start_global_index = int(round(capture_start_time * sample_rate))
    writer = drf.DigitalRFWriter(
        channel_dir,
        np.complex64,
        DRF_SUBDIR_CADENCE_SECS,
        NARROWBAND_FILE_CADENCE_MILLISECONDS,
        start_global_index,
        int(sample_rate),
        1,
        uuid_str="HFDoppTool_narrowband",
        compression_level=0,
        checksum=False,
        is_complex=True,
        num_subchannels=1,
        is_continuous=True,
        marching_periods=True,
    )
    writer.rf_write(complex_iq)
    writer.close()

    metadata_writer = drf.DigitalMetadataWriter(
        metadata_dir,
        DRF_SUBDIR_CADENCE_SECS,
        DRF_METADATA_FILE_CADENCE_SECS,
        int(sample_rate),
        1,
        "metadata",
    )
    metadata_writer.write(
        start_global_index,
        {
            "center_frequencies": np.array([center_freq_hz], dtype=np.float64),
            "lat": np.float64(38.8),
            "long": np.float64(-77.1),
            "capture_start_sample": np.int64(start_global_index),
            "sample_rate_hz": np.float64(sample_rate),
            "sample_count": np.int64(complex_iq.shape[0]),
            "tx_station": tx_station,
            "rx_station": rx_station,
        },
    )


# Set up capture parameters
print(sd.query_devices())
devs = sd.default.device

# USE default (47,24) if want to output the recording to HDSDR ************************************************************

sd.default.device = (13, 16)  # 47 is DAX I/Q 1 and #17 is speakers and #24 goes to HDSDR
devs = sd.default.device

# Do the recording

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
_mode = "AM" if DEMOD_AM else "USB"  # adjust to match the mode strings your waterfall program uses
_radio_send(f"C11|display pan s 0x42000000 center={_freq_mhz}\r\n")
_radio_send("c12|dax iq set 1 pan 0x40000000  daxiq_rate=48000\r\n")
_radio_send(f"C1|slice tune {FLEX_SLICE} {_freq_mhz:.6f}")
_radio_send(f"C2|slice set {FLEX_SLICE} mode={_mode}")
_radio_send(f"C3|slice set {FLEX_SLICE} dax=1")

# _radio_stop()


_rec_prefix = f"{TX_STATION}_{RX_STATION}" if RX_STATION else TX_STATION
_rec_timestamp = currentDateStamp.strftime("%Y%m%d_%H%M%S")
_rec_base = f"{_rec_prefix}_{_rec_timestamp}"
WAV_RECORDING_PATH   = os.path.join(RECORDING_DIR, f"{_rec_base}.wav")
WAV_REAL_PATH        = os.path.join(RECORDING_DIR, f"{_rec_base}_real.wav")
WAV_AMDEMOD_PATH     = os.path.join(RECORDING_DIR, f"{_rec_base}_AMdemod.wav")
NARROWBAND_DRF_PATH  = os.path.join(RECORDING_DIR, f"{_rec_base}_narrowband_drf")

if DO_RECORD:

    rec_start_time = time.time()
    myrecording = sd.rec(int(REC_DURATION * fs), samplerate=fs, channels=2)
    sd.wait()

    sf.write(WAV_RECORDING_PATH, myrecording * 12.0, 48000)

    myrecording_real = myrecording[:, 0].astype(np.float32)
    myrecording_AMdemod = np.sqrt(myrecording[:, 0] ** 2 + myrecording[:, 1] ** 2).astype(np.float32)

    sf.write(WAV_REAL_PATH, myrecording_real * 12.0, 48000)
    sf.write(WAV_AMDEMOD_PATH, myrecording_AMdemod * 12.0, 48000)


if (not DO_RECORD):
    # Paths built at startup use today's timestamp, so search for the most
    # recent existing recording instead of relying on the current-run path.
    import glob
    _wav_candidates = sorted(
        [f for f in glob.glob(os.path.join(RECORDING_DIR, "*.wav"))
         if not f.endswith(("_real.wav", "_AMdemod.wav"))],
        key=os.path.getmtime, reverse=True
    )
    if _wav_candidates:
        WAV_RECORDING_PATH = _wav_candidates[0]
        myrecording, fs = sf.read(WAV_RECORDING_PATH)
        # WAV files carry no capture-time metadata; approximate the start time
        # from the file's mtime (when the recording finished writing) minus its duration.
        rec_start_time = os.path.getmtime(WAV_RECORDING_PATH) - len(myrecording) / fs
        print(f"Loaded WAV recording: {WAV_RECORDING_PATH}")
    else:
        raise FileNotFoundError(f"No recording found in {RECORDING_DIR}")


myrecLen = len(myrecording)
print("myreclen = ", myrecLen)

complexSamps = (myrecording[:, 0] + 1j * myrecording[:, 1]).astype(np.complex64)

# Decimate straight from the raw 48kHz IQ to NARROWBAND_RATE (10 sps) for the HamSCI
# narrow-band DRF. resample_poly designs and applies its own anti-aliasing filter
# sized for this decimation ratio -- no separate filtering step needed.
_decim_factor = fs // NARROWBAND_RATE
narrow_iq = scipy.signal.resample_poly(complexSamps, 1, _decim_factor)
write_narrowband_drf(NARROWBAND_DRF_PATH, narrow_iq, NARROWBAND_RATE, CARRIER_FREQ_HZ, TX_STATION, RX_STATION, capture_start_time=rec_start_time)
print(f"Saved narrow-band DRF ({len(narrow_iq)} samples @ {NARROWBAND_RATE} sps): {NARROWBAND_DRF_PATH}")

if PLAY_RECORDING:
    _played = False
    if DEMOD_AM:
        sd.play(np.absolute(complexSamps) * 5.0)  # demods AM
        _played = True
    if DEMOD_SSB_CW:
        sd.play(complexSamps.real * 5.0)  # demods SSB and CW
        _played = True
    if _played:
        sd.wait()
