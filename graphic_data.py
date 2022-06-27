
from __future__ import print_function
from numpy import ndarray, zeros, ones, array, asarray, vstack, dot, mean as Mean, square, std as Std, var, empty, transpose, nan_to_num, float64, int64, load, uint8, float32, newaxis
from cython_modules.history_handler.valuesRecoverer import LossValuesRecoverer
from cython_modules.visualizer import *

path = r"/scratch/gucr2/tEDRAM2/outputs/"
load_path = path + input("FILE:")
path_loss = load_path + "/history/" #output7_10/default/history/" # r"/scratch/gucr2/tEDRAM2/outputs/output56_25_10/default/history/"
print(" class loss: ")
path = path_loss + "classifications_loss.npy"
class_loss_data = LossValuesRecoverer(path.encode('UTF-8'))
print(" loss data: ")
path = path_loss + "loss.npy"
loss_data = LossValuesRecoverer(path.encode('UTF-8'))
print(" class categorical acc: ")
path = path_loss+"classifications_categorical_accuracy.npy"
class_cat_acc = LossValuesRecoverer(path.encode('UTF-8'))
print(" history: ")
history = load(path_loss+ "history.npy", allow_pickle=True)
print(" loc loss: ")
path = path_loss+"localisations_loss.npy"
loc_loss = LossValuesRecoverer(path.encode('UTF-8'))
print(" val cla: ")
path = path_loss+"val_classifications_categorical_accuracy.npy"
val_class = LossValuesRecoverer(path.encode('UTF-8'))
print(" val cla loss: ")
path = path_loss+"val_classifications_loss.npy"
val_cla_loss = LossValuesRecoverer(path.encode('UTF-8'))
print(" val loc loss: ")
path = path_loss + "val_localisations_loss.npy"
val_loc_loss = LossValuesRecoverer(path.encode('UTF-8'))
print(" val loss: ")
path = path_loss + "val_loss.npy"
val_loss = LossValuesRecoverer(path.encode('UTF-8'))
####################################################################
########################## plot ####################################
####################################################################

X = loss_data.data()
Y = val_loss.data()
visualise_plot(X, "Trainings- bzw. Validierungsverlust" , "Epoche", "Mittlerer quadratischer Fehler ", "./loss1.png", Y )
#grpcs = TerminalGraphics(100,10,(0,1),(0,1))
#grpcs.plot_config(X,Y,c=25, l="loss vs val loss ")
X = loc_loss.data()
Y = val_loc_loss.data()
visualise_plot(X, "Lokalisierungsverlust: tEDRAM + Aug." , "Epoche", "Mittlerer quadratischer Fehler ", "./loss2.png", Y)
#grpcs.plot_config(X, Y, c=50, l="loc loss vs val loc loss")

X = class_loss_data.data()
Y = val_cla_loss.data()
visualise_plot(X, "Klassifizierungsverlust: tEDRAM + Aug.", "Epoche", "Kategoriale Kreuzentropie", "./loss3.png", Y)

X = class_cat_acc.data()
Y = val_class.data()
visualise_plot(X, "Klassifizierungsgenauigkeit: tEDRAM + Aug.", "Epoche", "Kategoriale Kreuzentropie", "./loss4.png", Y)
