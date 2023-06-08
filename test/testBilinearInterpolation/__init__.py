import tensorflow as tf
from tensorflow import keras

from bilinearInt import Slice
from bilinearInt import BilinearInterpolation

import matplotlib.pyplot as mpl

import numpy as np
import os
import sys

batch_size = 64
num_classes = 10

#gpu_id = sys.argv[1]

# if gpu_id not in ('0','1'):
#     print("Abort")
#     exit()
# else:
#     print("Using GPU", gpu_id)
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# Prepare a dataset.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_ordered = np.zeros((0,) + x_train.shape[1:])
y_ordered = np.zeros((0,) + y_train.shape[1:])
for i in np.arange(10):
    x_ordered = np.vstack((x_ordered, x_train[y_train==i]))
    y_ordered = np.hstack((y_ordered, y_train[y_train==i]))


x_train = x_ordered.reshape((60000//2, 2, 28, 28))
y_train = y_ordered.reshape((60000//2, 2))


dataset = tf.data.Dataset.from_tensor_slices((x_train[..., None], y_train[:, 0]))

dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

print("\n Dataset:\n")
for data, labels in dataset:
   print(data.shape)  # (64, 200, 200, 3)
   print(data.dtype)  # float32
   print(labels.shape)  # (64,)
   print(labels.dtype)  # int32
   break
print()

# initiate
input_image = keras.Input(shape=(2, 28, 28, 1), name='input')

sel1 = Slice([0, 0, 0, 0, 0], [-1, 1, -1, -1, -1], name='sel1')
sel2 = Slice([0, 1, 0, 0, 0], [-1, 1, -1, -1, -1], name='sel2')

conv_att_1 = keras.layers.Conv2D(8, (5,5), padding='valid', name='conv_att_1')
conv_att_2 = keras.layers.Conv2D(8, (5,5), padding='valid', name='conv_att_2')
pool_att = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool_att')
avg_att = keras.layers.GlobalAveragePooling2D(name='avg_att')
dense_att = keras.layers.Dense(6, activation="relu", name='dense_att')
reshape_att = tf.keras.layers.Reshape((2, 3), name='reshape_att')

bilin = BilinearInterpolation(height=10, width=10)
#bilin = BilinearInterpolation((10, 10), 1.0)

conv1 = keras.layers.Conv2D(8, (3,3), padding='valid', name='conv1')
conv2 = keras.layers.Conv2D(8, (3,3), padding='valid', name='conv2')
pool = keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool')
dense = keras.layers.Dense(num_classes, activation="relu", name='dense')
avg = keras.layers.GlobalAveragePooling2D(name='avg')


# build
input_image1 = sel1(input_image)
input_image2 = sel2(input_image)

x = conv_att_1(input_image1)
x = conv_att_2(x)
x = pool_att(x)
x = avg_att(x)
att = dense_att(x)
att = reshape_att(att)
x = bilin([input_image1, att])

x = conv1(x)
x = conv2(x)
x = pool(x)
output1 = avg(x)

x = conv_att_1(input_image2)
x = conv_att_2(x)
x = pool_att(x)
x = avg_att(x)
att = dense_att(x)
att = reshape_att(att)
x = bilin([input_image2, att])

x = conv1(x)
x = conv2(x)
x = pool(x)
output2 = avg(x)

x = tf.keras.layers.concatenate([output1, output2])
output = dense(x)

model = keras.Model(inputs=input_image, outputs=output)

model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

model.fit(dataset, epochs=10)

# print("\n Show layers\n")
# for layer in model.layers:
#     if layer.get_config()['name']=='conv1':
#         for i in np.arange(8):
#             mpl.imshow(layer.get_weights()[0][..., i])
#             mpl.show()
