# Flex DAX IQ Recorder

Records Flex 6000-series DAX IQ audio (via Windows Sound System) and produces
HamSCI/PSWS-compatible narrow-band DigitalRF (10 sps) datasets for
Doppler/propagation monitoring — either one frequency at a time, or up to 4
simultaneous frequencies interleaved as PSWS-style subchannels of a single
DRF channel. Can also drive the radio directly over the SmartSDR TCP API to
set frequency/mode and enable DAX before recording starts.

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
- `sd.default.device = (13, 16)` is hardcoded near the top of
  `DAXiqRec_capture_streaming.py`/`DAXiqRec_capture.py` (input = DAX IQ 1,
  output = Speakers). Windows periodically renumbers device indices (driver
  updates, reconnects, reboots) and breaks this silently. If
  recording/playback misbehaves, check `sd.query_devices()` against the
  hardcoded indices before assuming a logic bug. `FlexRec_PSWS.py` avoids
  this by resolving each device by name match instead (see below) — that
  fix hasn't been back-ported to the other two scripts.
- **Stopping a script via VS Code's debug "Stop" button force-kills the
  process** rather than sending a signal Python can catch — no traceback, no
  console output, and any `try/finally` cleanup (closing writers, writing
  final metadata) simply never runs. Only a `Ctrl+C` in the actual terminal
  window running the script raises a catchable `KeyboardInterrupt`. See the
  provisional/final metadata write pattern below for how the streaming
  capture scripts cope with this.
- Relatedly: don't use a third-party library's own blocking sleep/wait call
  (e.g. `sounddevice.sleep()`) where `time.sleep()` would do. `sd.sleep()`
  calls straight into PortAudio's C `Pa_Sleep()`, which is opaque to
  Python's signal handling — a `Ctrl+C` during that call gets silently
  deferred until it returns on its own, making `Ctrl+C` look like it does
  nothing on anything longer than a few seconds. Both streaming capture
  scripts use `time.sleep()` for exactly this reason; if you add a new wait
  anywhere in this codebase, use `time.sleep()` too.
- `lat`/`long` in the narrow-band metadata are still hardcoded placeholder
  values (near the DC area, not the actual station location) in both
  streaming capture scripts — real station coordinates aren't wired up yet.

## Capture scripts

### `DAXiqRec_capture_streaming.py` — single-frequency capture, current/recommended for that case

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

### `FlexRec_PSWS.py` — multi-frequency PSWS-subchannel capture

Captures up to 4 simultaneous Flex DAX IQ channels (one Flex slice + one DAX
IQ audio device each) and interleaves them into a **single** PSWS-style DRF
channel (`num_subchannels=N`, one column per frequency) — matching how real
PSWS/"grape"-style multi-frequency stations structure their DRF output,
rather than producing N separate single-frequency datasets. A fork of
`DAXiqRec_capture_streaming.py`, not a replacement for it — that script is
unchanged and remains the simple single-frequency path.

**Confirmed working (2026-07-29):** up to 4 simultaneous channels, including
a 12-hour unattended overnight 2-channel run (validating long-duration
stability — flat memory, no buffer drift). The design isn't hardcoded to any
particular channel count — it's entirely driven by how many entries are in
the `CAPTURE_CHANNELS` list near the top of the script.

Runs everything in **one process** rather than one process per channel — a
single `DigitalRFWriter` has to own the whole subchannel-interleaved write,
so the capture streams can't be fully independent. Each channel still gets
its own `sd.InputStream`/decimator running on its own PortAudio callback
thread, feeding a thread-safe FIFO (`SampleBuffer`); a main-thread loop
drains the minimum available sample count across *all* buffers every 0.5s
and writes one combined `(n, N)` block, keeping every subchannel
sample-aligned. The heavy FIR filtering runs inside NumPy/SciPy's C code
(which releases the GIL), so N lightweight decimation callbacks sharing one
process has held up fine in testing so far.

Each `CAPTURE_CHANNELS` entry also resolves its DAX IQ device by **name
substring match** (`find_input_device()`) rather than a hardcoded index —
this sidesteps the Windows device-renumbering gotcha described above, but
that fix hasn't been back-ported to the single-frequency script.

**Radio control requires manual setup first.** The pan/DAX-IQ stream
handles in `CAPTURE_CHANNELS` (`pan_stream_id`/`dax_stream_id`) mutate
*existing* panadapter/DAX-IQ stream objects on the radio — the script
doesn't create them. Before running with N channels, manually pre-create N
panadapters/DAX-IQ streams in SmartSDR first, and confirm the hex handles in
`CAPTURE_CHANNELS` actually match your session (deliberately mistune each
panadapter, run the script, and confirm it retunes each one back to the
configured frequency — that's how the confirmed channels above were
verified). If panadapters are ever closed and recreated, the real handles
will likely change and need re-verifying the same way.

## `DAXiqRec_capture.py` — deprecated

The original approach: `sd.rec()` blocks for the entire recording duration,
buffering it all in memory, then writes raw/real/AM-demod WAV files and
decimates the whole thing to narrow-band DRF in one shot via
`scipy.signal.resample_poly` at the end. Also supports loading and
reprocessing a previously-recorded WAV file (`DO_RECORD = False`).

Kept around only for when you actually need the big WAV outputs or want to
reprocess an old recording — use one of the streaming scripts for everything
else. **Note:** this script still writes metadata with the old `tx_station`/
`rx_station` field names and hasn't been updated to match PSWS naming — see
below. If you need PSWS-compatible metadata out of a WAV reprocessing run,
port the field names over from `DAXiqRec_capture_streaming.py` first.

## PSWS metadata compatibility

PSWS ("grape"-style, e.g. KA9Q-based) DigitalRF datasets store station
identity and frequency info in the channel's `DigitalMetadataReader` entries
using a specific set of field names: `callsign`, `center_frequencies`,
`grid_square`, `lat`, `long`, `receiver_name`, `uuid_str`. All three capture
scripts except `DAXiqRec_capture.py` (see above) write metadata using those
same field names, so the resulting DRF datasets are readable by standard
PSWS tooling, not just our own scripts.

PSWS datasets can pack *multiple* simultaneously-monitored frequencies as
subchannels of one DRF channel (`num_subchannels > 1`, one column per
frequency) — `FlexRec_PSWS.py` now does exactly this on the Flex (up to 4
channels confirmed working; see above). `DAXiqRec_capture_streaming.py`
still records one frequency at a time (`num_subchannels=1`) and is the
simpler choice when you only need a single frequency.

`drf_process.py` and `DRFMetaReader.py` both handle single- and
multi-subchannel datasets transparently — see the Utilities table below.

## Utilities

| Script | Purpose |
|---|---|
| `drf_process.py` | Prompts for a DRF dataset via a folder-picker dialog, then automatically detects how many subchannels (frequencies) are present and runs the full plot pipeline once per subchannel — a raw spectrogram + magnitude-of-raw-data figure for each, followed (for 2+ subchannels) by one combined figure stacking every subchannel's spectrogram for a quick side-by-side comparison, using the same color/intensity scale each individual plot uses. Set `RUN_EXPERIMENTAL_VESSELIZATION = True` near the top to also run the experimental Frangi-ridge-filter "vesselization" pass (Doppler-shift/propagation-mode track extraction) for each subchannel — off by default. |
| `DRFMetaReader.py` | Prompts for a DRF dataset via a folder-picker dialog, then dumps channel properties, subchannel metadata (callsign/grid square/receiver name/center frequencies, when present), and sample bounds to `drf header.txt`. Handles datasets with no metadata at all (e.g. an interrupted recording from before the provisional-write fix, or any other dataset missing `dmd_properties.h5`) by printing a message instead of crashing. |
| `plot_drf.py` | Loads a narrow-band DRF dataset and plots a spectrogram, PSD, time-domain I/Q, and I/Q constellation; saves a PNG alongside the dataset. `python plot_drf.py <path-to-narrowband-drf>`. Falls back to raw HDF5 reads via `h5py` if `digital_rf` isn't installed. Guesses a center frequency from the folder name (`CHU7`/`CHU3`/`CHU14`/`WWV10`) for frequency-axis labeling — update that list if you record other stations. Not yet subchannel-aware (see `drf_process.py` for that). |
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

All three capture scripts open a `TCP_Flex2.start_telnet_client` connection
to the radio (`10.0.0.252:4992` by default) before recording and send
SmartSDR commands to set the pan center frequency, configure the DAX IQ
stream, tune the slice, and set its mode. Adjust `FLEX_HOST`/`FLEX_PORT` (and
`FLEX_SLICE` in the single-frequency scripts, or the per-channel `flex_slice`
values in `FlexRec_PSWS.py`) and the command strings near the top of each
script if your radio's address or command syntax differs.

`FlexRec_PSWS.py` additionally requires the panadapters/DAX-IQ streams it
controls to already exist in your SmartSDR session — see that script's
section above before running it with more than one channel.
