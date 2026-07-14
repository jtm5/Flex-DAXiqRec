# Flex DAX IQ Recorder

Records Flex 6000-series DAX IQ audio (via Windows Sound System) and produces a
HamSCI-compatible narrow-band DigitalRF (10 sps) dataset for Doppler/propagation
monitoring. Can also drive the radio directly over the SmartSDR TCP API to set
frequency/mode and enable DAX before recording starts.

And, a nod to Claude code for doing some of the heavy lifting on this project!

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

### `DAXiqRec_capture.py` — deprecated

The original approach: `sd.rec()` blocks for the entire recording duration,
buffering it all in memory, then writes raw/real/AM-demod WAV files and
decimates the whole thing to narrow-band DRF in one shot via
`scipy.signal.resample_poly` at the end. Also supports loading and
reprocessing a previously-recorded WAV file (`DO_RECORD = False`).

Kept around only for when you actually need the big WAV outputs or want to
reprocess an old recording — use the streaming version for everything else.

## Utilities

| Script | Purpose |
|---|---|
| `DRFMetaReader.py` | Dumps channel properties, sample bounds, and metadata for a DRF dataset to a text file. Useful for sanity-checking a recording's timing (see note below on `get_bounds()`). |
| `plot_drf.py` | Loads a narrow-band DRF dataset and plots a spectrogram, PSD, time-domain I/Q, and I/Q constellation. `python plot_drf.py <path-to-narrowband-drf>`. |
| `drf test.py` | Quick spectrogram-only plot of a DRF dataset (lighter-weight than `plot_drf.py`). |
| `DRF_reader.py` | Older reference script (originally by W. Engelke, AB4EJ) that reads a DRF dataset into a 2D magnitude array and prints per-minute metadata. Needs the `maidenhead` package (not in `requirements.txt`). |
| `TCP_Flex2.py` | The SmartSDR TCP client (`start_telnet_client`) actually used by the capture scripts for radio init. Plain two-way TCP client, no state object required — just `send_func`/`stop_func` closures. |
| `TCP_Flex.py`, `Flex_Parser.py` | Earlier scratch/prototype TCP clients used while developing `TCP_Flex2.py`. Not imported by anything; kept for reference. |

## Notes on DRF timing

`DigitalRFReader.get_bounds()` reports the cadence-padded file extent (files
are pre-sized to a full 3-minute cadence period and zero/NaN-padded at the
edges), **not** the true recorded span — a recording that doesn't start/end
exactly on a 3-minute boundary will show an earlier "Start Index" than the
real first sample. Use `get_continuous_blocks()` instead when you need the
actual recorded extent, e.g. for a short test recording. The narrow-band
writer's `capture_start_sample` metadata field always reflects the true
start; real HamSCI/Grape recordings match `get_bounds()` exactly because
their capture pipeline starts precisely on a cadence boundary, not because
they avoid padding.

## Radio control

Both capture scripts open a `TCP_Flex2.start_telnet_client` connection to the
radio (`10.0.0.252:4992` by default) before recording and send SmartSDR
commands to set the pan center frequency, configure the DAX IQ stream, tune
the slice, and set its mode. Adjust `FLEX_HOST`/`FLEX_PORT`/`FLEX_SLICE` and
the command strings near the top of each script if your radio's address or
command syntax differs.
