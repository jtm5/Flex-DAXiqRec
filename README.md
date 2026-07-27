# Flex DAX IQ Recorder

Records Flex 6000-series DAX IQ audio (via Windows Sound System) and produces a
HamSCI/PSWS-compatible narrow-band DigitalRF (10 sps) dataset for
Doppler/propagation monitoring. Can also drive the radio directly over the
SmartSDR TCP API to set frequency/mode and enable DAX before recording starts.

And, a nod to Claude Code for doing some of the heavy lifting on this project!

## Setup

This project uses a dedicated venv (`.venv`) to stay isolated from whatever
Python installs come and go on this machine via Visual Studio, the Windows
Store, etc.

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Point your editor's Python interpreter at `.venv\Scripts\python.exe`
(`.vscode/settings.json` already does this for VS Code).

**Known gotchas:**
- `requirements.txt` pins `sounddevice<0.5` — 0.5.x crashes the interpreter
  with `STATUS_HEAP_CORRUPTION` on import on this machine. Don't "helpfully"
  upgrade it.
- `sd.default.device = (13, 16)` is hardcoded near the top of each capture
  script (input = DAX IQ 1, output = Speakers). Windows periodically
  renumbers device indices (driver updates, reconnects, reboots) and breaks
  this silently. If recording/playback misbehaves, check
  `sd.query_devices()` against the hardcoded indices before assuming a logic
  bug. Future revisions will ditch the hard coding!
- **Stopping a script via VS Code's debug "Stop" button force-kills the
  process** rather than sending a signal Python can catch — no traceback, no
  console output, and any `try/finally` cleanup (closing writers, writing
  final metadata) simply never runs. Only a `Ctrl+C` in the actual terminal
  window running the script raises a catchable `KeyboardInterrupt`. See the
  provisional/final metadata write pattern below for how
  `DAXiqRec_capture_streaming.py` copes with this.
- `lat`/`long` in the narrow-band metadata are still hardcoded placeholder
  values in `DAXiqRec_capture_streaming.py` (near the DC area, not the actual
  station location) — real station coordinates aren't wired up yet.

## Capture scripts

### `DAXiqRec_capture_streaming.py` — current, recommended

Live capture straight to narrow-band DigitalRF. Uses `sd.InputStream` with a
callback that decimates and writes to the DRF writer incrementally as audio
arrives, instead of buffering the whole recording in memory:

- Memory use stays flat regardless of recording length (validated on a
  12-hour run).
- A crash or kill mid-recording only loses the last few seconds of DRF data,
  not the whole run.
- No raw/real/AM-demod WAV output — narrow-band DRF is the only artifact.
- No "load an existing recording and reprocess it" mode — capture-only.

The 4800:1 decimation (48 kHz → 10 sps) runs as a two-stage filter cascade
(60× then 80×) rather than one single-stage filter, since a direct 4800:1
decimation needs a ~65,000-tap FIR to hit a clean 5 Hz cutoff — far too slow
to run per-chunk in a real-time audio callback. Verified against
`scipy.signal.resample_poly` on a synthetic signal: correct frequency/amplitude
recovery, no discontinuities at chunk boundaries.

Known limitation: `blocksize=fs` (1-second blocks) means up to ~1 second of
audio at the very end of a recording can be dropped when the stream closes
before a final block completes. Negligible for real (multi-hour+) recordings;
visible on short test runs. Not worth fixing further unless that changes.

The settings dialog (shown at launch) collects transmitter station, receiver
callsign, grid square, receiver name, center frequency, and recording
duration — these feed directly into the narrow-band DRF metadata (see
**PSWS metadata compatibility** below).

**Metadata is written twice**, to survive a hard process kill:
1. A **provisional** record (`sample_count=0`) immediately after the DRF
   writer is created, before recording even starts.
2. A **final** record with the true sample count, written in the `finally`
   block once recording stops.

This exists because of the VS Code Stop-button gotcha above — a hard kill
skips step 2 entirely, but step 1 already guarantees `dmd_properties.h5` and
at least one valid metadata entry exist, so the dataset stays readable by
`DRFMetaReader.py` (or any other DigitalMetadataReader-based tool) either way.

### `DAXiqRec_capture.py` — deprecated

The original approach: `sd.rec()` blocks for the entire recording duration,
buffering it all in memory, then writes raw/real/AM-demod WAV files and
decimates the whole thing to narrow-band DRF in one shot via
`scipy.signal.resample_poly` at the end. Also supports loading and
reprocessing a previously-recorded WAV file (`DO_RECORD = False`).

Kept around only for when you actually need the big WAV outputs or want to
reprocess an old recording — use the streaming version for everything else.
**Note:** this script still writes metadata with the old `tx_station`/
`rx_station` field names and hasn't been updated to match PSWS naming — see
below. If you need PSWS-compatible metadata out of a WAV reprocessing run,
port the field names over from `DAXiqRec_capture_streaming.py` first.

## PSWS metadata compatibility

PSWS ("grape"-style, e.g. KA9Q-based) DigitalRF datasets store station
identity and frequency info in the channel's `DigitalMetadataReader` entries
using a specific set of field names: `callsign`, `center_frequencies`,
`grid_square`, `lat`, `long`, `receiver_name`, `uuid_str`. Our own narrow-band
captures (`DAXiqRec_capture_streaming.py`) now write metadata using those same
field names, so the resulting DRF datasets are readable by standard PSWS
tooling, not just our own scripts.

A structural difference remains, and is **not** something this project needs
to close right now: PSWS datasets can pack *multiple* simultaneously-monitored
frequencies as subchannels of one DRF channel (`num_subchannels > 1`, one
column per frequency), whereas our Flex-based captures record one frequency
at a time (`num_subchannels=1`). Doing true simultaneous multi-frequency
capture would need different receiver hardware (e.g. an RX-888) — using the
Flex for that would be overkill.

`drf_process.py` already handles both cases: if a dataset has more than one
subchannel, it reads the `center_frequencies` metadata array, picks whichever
subchannel is closest to the `TARGET_FREQ_HZ` constant near the top of the
script (handling the fact that PSWS datasets store that array in MHz while
our own captures store it in Hz), and warns if the closest match is more than
50 kHz off. Single-subchannel datasets (our own captures) are unaffected.

## Utilities

| Script | Purpose |
|---|---|
| `drf_process.py` | Loads a narrow-band DRF dataset (ours or a PSWS-style multi-subchannel one — see above), plots a raw spectrogram, then runs an experimental "vesselization" pass (Frangi ridge filter + peak tracking) over the spectrogram to pick out Doppler-shift/propagation-mode tracks. Set `drf_path`, `TARGET_FREQ_HZ`, and `fft_size` near the top before running. |
| `DRFMetaReader.py` | Dumps channel properties, subchannel metadata (callsign/grid square/receiver name/center frequencies, when present), and sample bounds for a DRF dataset to `drf header.txt`. Handles datasets with no metadata at all (e.g. an interrupted recording from before the provisional-write fix, or any other dataset missing `dmd_properties.h5`) by printing a message instead of crashing. |
| `plot_drf.py` | Loads a narrow-band DRF dataset and plots a spectrogram, PSD, time-domain I/Q, and I/Q constellation; saves a PNG alongside the dataset. `python plot_drf.py <path-to-narrowband-drf>`. Falls back to raw HDF5 reads via `h5py` if `digital_rf` isn't installed. Guesses a center frequency from the folder name (`CHU7`/`CHU3`/`CHU14`/`WWV10`) for frequency-axis labeling — update that list if you record other stations. |
| `DRF_reader.py` | Older reference script (originally by W. Engelke, AB4EJ) that reads a DRF dataset into a 2D magnitude array and prints per-minute metadata. Needs the `maidenhead` package (not in `requirements.txt`). |
| `TCP_Flex2.py` | The SmartSDR TCP client (`start_telnet_client`) actually used by the capture scripts for radio init. Plain two-way TCP client, no state object required — just `send_func`/`stop_func` closures. |
| `TCP_Flex.py`, `Flex_Parser.py` | Earlier scratch/prototype TCP clients used while developing `TCP_Flex2.py`. Not imported by anything; kept for reference. |

## Notes on DRF timing

`DigitalRFReader.get_bounds()` reports the cadence-padded file extent (files
are pre-sized to a full cadence period and zero/NaN-padded at the edges),
**not** the true recorded span — a recording that doesn't start/end exactly on
a cadence boundary will show an earlier "Start Index" than the real first
sample. Use `get_continuous_blocks()` instead when you need the actual
recorded extent, e.g. for a short test recording or one that was cut short.
This applies equally to our own DRFs and to real PSWS/HamSCI DRFs.

## Radio control

Both capture scripts open a `TCP_Flex2.start_telnet_client` connection to the
radio (`10.0.0.252:4992` by default) before recording and send SmartSDR
commands to set the pan center frequency, configure the DAX IQ stream, tune
the slice, and set its mode. Adjust `FLEX_HOST`/`FLEX_PORT`/`FLEX_SLICE` and
the command strings near the top of each script if your radio's address or
command syntax differs.
