from model_keras import *

import os
import argparse
parser = argparse.ArgumentParser(description='Trains a deep network on FACS-annotated datasets.')
parser.add_argument("--gpu", help="Specifies the GPU id.", type=int, action="store", nargs='?', default=-1, const=0, dest="gpu_device")
parser.add_argument("--em_weights", help=".", type=int, action="store", default=1, dest="em_weights")
parser.add_argument("--bn", help=".", type=int, action="store", default=1, dest="bn")
parser.add_argument("--mode", help=".", type=int, action="store", default=1, dest="mode")
parser.add_argument("--steps", help=".", type=int, action="store", default=1, dest="steps")
parser.add_argument("--model", help=".", type=str, action="store", default=1, dest="model_path")
parser.add_argument("--input", help=".", type=str, action="store", default=1, dest="input_path")

options, unknown = parser.parse_known_args()

# Select the correct GPU
if options.gpu_device == -1:
    print('You need to select a gpu. Ex: python Train.py --gpu=2')
    exit()
else:
    print('Using GPU', options.gpu_device)
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu_device)


edram = edram_model(steps=options.steps, preset_emission_weights=options.em_weights, bn=options.bn, output_mode=options.mode)

edram.summary()
edram.get_layer('edram_cell').summary()

i = [np.reshape(np.random.rand(20000), (2,100,100,1)),np.reshape([1,0,0,0,1,0,1,0,0,0,1,0],(2,6)), np.zeros((2,512)), np.zeros((2,512)), np.zeros((2,512)), np.zeros((2,512))]
i[0][0,13,71,0] = 1000

output = edram.predict(i)

print(output[0].shape)
print(output[0])
print(output[1].shape)
print(output[1])


if False:

    model = silly_lstm_model()
    #model = silly_lstm_model_2()
    model.summary()

    model.layers[3].set_weights(model.layers[7].get_weights())

    i = np.random.rand(4)
    i = [np.reshape([i,i],(1,2,4)), np.reshape(i,(1,1,4)), np.zeros((1,2)), np.zeros((1,2))]

    o = model.predict(i)

    for i in o:
        print(i)
