import digital_rf as drf

filename = "drf header.txt"

def print_to_File(filename, content):
    with open(filename, "a") as f:
        f.write(content)

# 1. Initialize the reader with the path to your dataset
dataset_path = 'D:\\Data\\Ham Radio\\HAMSci Local Experiments\\WWV15_K1FR_20260727_103022_narrowband_drf'
reader = drf.DigitalRFReader(dataset_path)

# 2. Get all available channels in the dataset
channels = reader.get_channels()
print(f"Available channels: {channels}")
for channel in channels:
    print(f"Channel: {channel}", reader.get_properties(channel),"\r\n")
print_to_File(filename, f"Available channels: {channels}\r\n")
# print_to_File("drf header.txt", f"Available channels: {channels}")
# print_to_File("drf header.txt", f"Available channels: {channels}")
# Let's inspect the first available channel
channel = channels[0]

# 3. Get channel properties (e.g., sample rate, center frequency)
properties = reader.get_properties(channel)
print_to_File(filename, f"\nProperties for channel {channel}:\r\n")
for key, value in properties.items():
    print_to_File(filename, f"  {key}: {value}\r\n")

# Inspect metadata dictionary for subchannel-specific metadata entries.
# A recording that was interrupted before the capture script reached its
# metadata-write step has no dmd_properties.h5 in the metadata folder --
# DigitalMetadataReader() raises OSError for that (a missing-file error,
# not a "no records in range" condition), so it needs its own guard rather
# than the bounds check below.
meta_reader = None
try:
    meta_reader = drf.DigitalMetadataReader("D:\\Data\\Ham Radio\\HAMSci Local Experiments\\WWV15_K1FR_20260727_103022_narrowband_drf\\ch0\\metadata")
except OSError as e:
    print(f"No metadata available for this dataset ({e}). "
          f"Likely an interrupted recording -- RF data may still be usable.")

if meta_reader is not None:
    bounds = meta_reader.get_bounds()
    if bounds[0] is not None:
        # Read the latest metadata entry
        latest_sample = bounds[1]
        metadata_dict = meta_reader.read(latest_sample, latest_sample)
        print("Subchannel Metadata Dictionary:")
        for key, val in metadata_dict[latest_sample].items():
            print(f"  {key}: {val}")
    else:
        print("No metadata records found in range.")

# 4. Get the time/sample bounds of the data
start_index, end_index = reader.get_bounds(channel)
print_to_File(filename, f"\nData Bounds (Sample Indices):\r\n")
print_to_File(filename, f"  Start Index: {start_index}\r\n")
print_to_File(filename, f"  End Index:   {end_index}\r\n")

if meta_reader is not None:
    # Get the time bounds for which metadata exists (in seconds since epoch)
    start_time, end_time = meta_reader.get_bounds()

    # Read all metadata entries within a specific time window
    metadata_dict = meta_reader.read(start_time, end_time)

    for timestamp, meta_fields in metadata_dict.items():
        print_to_File(filename, f"Time {timestamp}: {meta_fields}\r\n")


