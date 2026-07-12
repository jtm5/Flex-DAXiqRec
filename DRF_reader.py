# Read Digital RF file
# Author: W. Engelke, AB4EJ, University of Alabama

import matplotlib.pyplot as plt
import numpy as np
import digital_rf as drf
from datetime import datetime
import datetime
import math
import os, tempfile
import maidenhead as mh 
import sys, getopt, os


# Get supplied argument(s)
# Remove 1st argument from the
# list of command line arguments
argumentList = sys.argv[1:]
dataDir = ''

# Options - these are valid options
options = "hf:"
 
# Long options
long_options = ["Help", "theDate", "file"]

try:
    # Parsing argument
    arguments, values = getopt.getopt(argumentList, options, long_options)
     
    # checking each argument
    for currentArgument, currentValue in arguments:
 
        if currentArgument in ("-h", "--Help"):
            print ("-d YYYY-MM-DD -f filewname")

        elif currentArgument in ("-f", "--file"):
            print ("the file:",currentValue)
            dataDir = currentValue

except getopt.error as err:
    # output error, and return with an error code
    print (str(err))

if dataDir == '':
    print("Enter DRF dataset to be processed (full path):")
    dataDir = input()

plt.style.use('_mpl-gallery-nogrid')
maidenheadGrid = 'EN91' # this is just a default
# plot
fig, ax = plt.subplots()

metadata_dir = dataDir + '\\ch0\\metadata'

print("Looking for metadata at:" + metadata_dir)

do = drf.DigitalRFReader(dataDir)
s, e = do.get_bounds('ch0')

#print("Data avail. starting ", datetime.datetime.fromtimestamp(s/10))
#print("   thru ", datetime.datetime.fromtimestamp(e/10))

#print("Plot spectrum for what date?  (YYYY-MM-DD)")
#t =input()

def parse_timestamp_token(token):
    for fmt in ('%Y-%m-%dT%H-%M-%S', '%Y-%m-%dT%H-%M'):
        try:
            return datetime.datetime.strptime(token, fmt)
        except ValueError:
            pass
    return None


def get_request_time(path, start_sample):
    """Resolve request time from path, ch0 folders, or DRF start bound."""
    norm_path = os.path.normpath(path)

    # 1) If the user passed a timestamped folder directly, use it.
    for part in reversed(norm_path.split(os.sep)):
        parsed = parse_timestamp_token(part)
        if parsed is not None:
            return parsed, part, "path"

    # 2) Otherwise inspect ch0 for timestamped folders.
    ch0_dir = os.path.join(norm_path, 'ch0')
    if os.path.isdir(ch0_dir):
        try:
            for part in sorted(os.listdir(ch0_dir), reverse=True):
                parsed = parse_timestamp_token(part)
                if parsed is not None:
                    return parsed, part, "ch0 folder"
        except OSError:
            pass

    # 3) Final fallback: use dataset start bound.
    fallback = datetime.datetime.utcfromtimestamp(start_sample / 10)
    return fallback, fallback.strftime('%Y-%m-%dT%H-%M-%S'), "DRF bounds"


requestTime, t, source = get_request_time(dataDir, s)
print("request date:" + t + " (source: " + source + ")")
# s from do.get_bounds() is already the correct sample index (unix_time * sample_rate)
# for narrowband 10 sps DRFs; do not overwrite it with the hour-boundary folder timestamp
print("time stamp ",s)


freqList = [0]
theLatitude = 0
theLongitude = 0

# get Metadata, if it exists
try:
    dmr = drf.DigitalMetadataReader(metadata_dir)
    print("metadata init okay")

    first_sample, last_sample = dmr.get_bounds()
    print("metadata bounds are %i to %i" % (first_sample, last_sample))

    # Metadata sample indices may use a different timescale than RF data indices.
    # Print raw bounds and only attempt UTC conversion when values are reasonable.
    try:
        if first_sample < 325036800000 and last_sample < 325036800000:
            from datetime import datetime # refresh this
            print("This is ",datetime.utcfromtimestamp(first_sample/10).strftime('%Y-%m-%d %H:%M:%S'),
                  "to", datetime.utcfromtimestamp(last_sample/10).strftime('%Y-%m-%d %H:%M:%S'),"UTC")
    except Exception as ex:
        print("metadata time conversion skipped:", ex)

    start_idx = int(np.uint64(first_sample))
    print('computed start_idx = ',start_idx)

    fields = dmr.get_fields()
    fields_set = set(fields)
    print("Available fields are <%s>" % (str(fields)))

    def first_value(value):
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        if arr.ndim == 0:
            return arr.item()
        return arr.reshape(-1)[0].item()

    if "tx_station" in fields_set:
        data_dict = dmr.read(start_idx, start_idx + 2, "tx_station")
        for key in data_dict.keys():
            tx_val = first_value(data_dict[key])
            if tx_val is not None:
                TX_STATION = tx_val
                print("TX Station: ", TX_STATION)
    else:
        print("Metadata field not present: tx_station")

    if "rx_station" in fields_set:
        data_dict = dmr.read(start_idx, start_idx + 2, "rx_station")
        for key in data_dict.keys():
            rx_val = first_value(data_dict[key])
            if rx_val is not None:
                RX_STATION = rx_val
                print("RX Station: ", RX_STATION)
    else:
        print("Metadata field not present: rx_station")
    if "center_frequencies" in fields_set:
        data_dict = dmr.read(start_idx, start_idx + 2, "center_frequencies")
        for key in data_dict.keys():
            freq_val = first_value(data_dict[key])
            if freq_val is not None:
                freqList = [freq_val]
                print("freq = ", freq_val)
    else:
        print("Metadata field not present: center_frequencies")
 
    if "center_frequencies" in fields_set:
        data_dict = dmr.read(start_idx, start_idx + 2, "center_frequencies")
        for key in data_dict.keys():
            freq_val = first_value(data_dict[key])
            if freq_val is not None:
                freqList = [freq_val]
                print("freq = ", freq_val)
    else:
        print("Metadata field not present: center_frequencies")

    if "lat" in fields_set:
        data_dict = dmr.read(start_idx, start_idx + 2, "lat")
        for key in data_dict.keys():
            lat_val = first_value(data_dict[key])
            if lat_val is not None:
                theLatitude = lat_val
                print("Latitude: ", theLatitude)
    else:
        print("Metadata field not present: lat")

    long_field = None
    if "long" in fields_set:
        long_field = "long"
    elif "lon" in fields_set:
        long_field = "lon"
    elif "longitude" in fields_set:
        long_field = "longitude"

    if long_field is not None:
        data_dict = dmr.read(start_idx, start_idx + 2, long_field)
        for key in data_dict.keys():
            lon_val = first_value(data_dict[key])
            if lon_val is not None:
                theLongitude = lon_val
                print("Longitude: ", theLongitude)
    else:
        print("Metadata longitude field not present (checked: long, lon, longitude)")

    if np.size(theLatitude) > 0 and np.size(theLongitude) > 0:
        maidenheadGrid = mh.to_maiden(theLatitude, theLongitude, 3)

except Exception as ex:
    print("Metadata read issue at " + metadata_dir + ": " + str(ex))


gain = 1.5
hr1 = np.arange(1024, dtype='f')
#print("numpy array type = ",type(hr1[0]))
zeros = np.zeros(1024, dtype='f')

print('Read data... this may take a few minutes...')
offset = 0
rows = []
good_reads = 0
gap_reads = 0
# Get data from DRF dataset and build the 2D Q array
for i in range(1439): # 1439 gives 1440 bins
    try:
        data = do.read_vector(s + offset, 1024, 'ch0')
        # At this point, the variable 'data' contains 1024 complex 32-bit float values
        # from the DRF dataset, starting at Unix time = s + offset
        rows.append(np.abs(data).astype('f'))
        good_reads += 1

    except IOError: # tried to read DRF data but did not find requsted time slice
        # Keep output shape stable across missing reads.
        rows.append(zeros.copy())
        gap_reads += 1

    # in narrow case, there are 10 samples/sec, so 600 samples = 1 minute
    offset = offset + 600  #  note overlap of the 1024 bins
    # if you want no overlap, set the above to offset = offset + 1024
    if (i % 100 == 0): # progress indicator, marching dots
        print(".",end='') # this works the same as when DRF used to store data

if len(rows) > 0:
    Q = np.vstack(rows)
else:
    Q = np.zeros((0, 1024), dtype='f')

print("\nDone reading DRF data.")
print("Q shape:", Q.shape, "good reads:", good_reads, "gaps filled:", gap_reads)
 

 
