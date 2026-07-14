import digital_rf as drf

filename = "drf header.txt"

def print_to_File(filename, content):
    with open(filename, "a") as f:
        f.write(content)

# 1. Initialize the reader with the path to your dataset
dataset_path = 'D:\\Data\\Ham Radio\\HAMSci Local Experiments\\WWV10_K1FR_20260713_195209_narrowband_drf'
reader = drf.DigitalRFReader(dataset_path)

# 2. Get all available channels in the dataset
channels = reader.get_channels()
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

# 4. Get the time/sample bounds of the data
start_index, end_index = reader.get_bounds(channel)
print_to_File(filename, f"\nData Bounds (Sample Indices):\r\n")
print_to_File(filename, f"  Start Index: {start_index}\r\n")
print_to_File(filename, f"  End Index:   {end_index}\r\n")

# Initialize the metadata reader directly pointing to the metadata folder
metadata_path = 'D:\\Data\\Ham Radio\\HAMSci Local Experiments\\WWV10_K1FR_20260713_195209_narrowband_drf\\ch0\\metadata'
meta_reader = drf.DigitalMetadataReader(metadata_path)

# Get the time bounds for which metadata exists (in seconds since epoch)
start_time, end_time = meta_reader.get_bounds()

# Read all metadata entries within a specific time window
metadata_dict = meta_reader.read(start_time, end_time)

for timestamp, meta_fields in metadata_dict.items():
    print_to_File(filename, f"Time {timestamp}: {meta_fields}\r\n")


