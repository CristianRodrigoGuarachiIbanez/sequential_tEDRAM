
import h5py
from numpy import asarray, uint8, float64
from pickle import dump, HIGHEST_PROTOCOL
#filename = r"/scratch/facs_data/AffectNet/AffectNet_train_data_keras.hdf5"
filename = "../src/training_data/binocular_image_data.h5"
with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    print(f['binocular_image'].items())
    #feature_key = list(f.keys())[1]
    #label_key = list(f.keys())[2]
    #location_key = list(f.keys())[3]
    left_img = list(f.keys())[0]
    # Get the data
    #print(location_key)
    #features = asarray(list(f[feature_key]), dtype=uint8)
    #labels = asarray(list(f[label_key]), dtype=uint8)
    #locations = asarray(list(f[location_key]), dtype=float64)
    left_images = asarray(list(f[left_img]["left_img"]), dtype=uint8)
#print(locations.shape)
print(left_images.shape)
"""
print(features.shape, labels.shape)
with h5py.File("/scratch/gucr/tEDRAM2/training_data/AffectNet_train_data_keras.h5", "w") as w:
    w.create_dataset("emotions", data=features, compression='gzip', compression_opts=9)
    print("...dataset saved")

with open("/scratch/gucr/tEDRAM2/training_data/emotions_labels.txt", "wb") as d:
    dump(labels, d, protocol=HIGHEST_PROTOCOL)
    print("... text format saved ")
"""