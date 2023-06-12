from typing import Dict
from h5py import File
from numpy import ndarray, asarray, newaxis
from pickle import load



def load_dataset(dataset_id: int, config: Dict):
    # train on CDS?
    # train on high-res input?
    if  dataset_id == 1:
        return config['input_shape_scene']  # (73969,120,160,1)-> (10,120,160,1)
    elif  dataset_id == 2:
        return config['disparity_maps_s']  # (73969,120,160,7)-> (10, 7, 120,160) (10,120,160,7),
    elif  dataset_id == 3:
        return config['disparity_maps_56']  # (73969,120,160,56)-> (10, 56, 120,160)  (10,120,160,56)
    elif dataset_id == 4:
        return config['disparity_maps']
    elif dataset_id == 5:
        return config["input_shape_binocular"] #(73969,120,160) -> (10,7,120,160),
    elif dataset_id == 6:
        return config["input_affectnet"]
    else:
        print('[Info] the shapes of the sequences (?,10,120,160,1), (?,10,7,8,120,160,1) and (?,10,7,120,160,1) are availables!')
        print('select as dataset id 1 and 2 correspondently')


class LOADER:
    data: File
    label: ndarray
    train_images: ndarray
    train_labels: ndarray
    train_locations: ndarray
    test_images: ndarray
    test_labels: ndarray
    test_locations: ndarray
    n_train: int
    n_test: int

    def get_train_images(self):
        return self.train_images

    def get_train_labels(self) -> ndarray:
        return self.train_labels

    def get_train_locations(self) -> ndarray:
        return self.train_locations

    def get_test_images(self) -> ndarray:
        return self.test_images

    def get_test_labels(self) -> ndarray:
        return self.test_labels

    def get_test_locations(self) -> ndarray:
        return self.test_locations

    def get_n_train(self) -> int:
        return self.n_train

    def get_n_test(self) -> int:
        return self.n_test


    def load_image_data(self, data_path:str):
        # declare variables
        # initialize the declared image and label data
        # load the image data

        try:
            self.data = File(data_path, 'r')  ##### HIER MUSS MAN DIE GROUPS AUFRUFEN
            print("keys", self.data.keys())
        except Exception as e:
            print("[Error]", e)
            exit()

    def load_label_data(self, labels_path:str):
        # ------------- load labels
        print("\n[Info] opening:", labels_path)
        try:
            file =  open(labels_path, 'rb')
            self.label = load(file)
            print('length of the labels data:', asarray(self.label).shape);
            file.close()

        except Exception as e:
            print("[Error] troubles at loading the data from pickle:",e);
            exit()


    def split_dataset(self, group:str, dataset:str, dataset_id:int):
        self.n_train: int = self.data[group][dataset].shape[0] - 22671
        self.n_test: int = self.data[group][dataset].shape[0] - self.n_train  # hier ist 22671
        # print(self.n_train, self.n_test)
        if (self.n_train > 0) and (self.n_test > 0):
            # train data
            self.train_images = self.data[group][dataset][:self.n_train]
            # self.train_images = train_images[:,:,:,:,:,newaxis]  # (73969, 7, 8, 120, 160)
            self.train_labels = asarray(self.label[:self.n_train])  #  --- 0. left_hand, 1. right_hand, 2.left_forearm, 3. right_forearm
            self.train_locations = None
            # test data
            self.test_images = self.data[group][dataset][self.n_train:]
            # self.test_images = test_images[:,:,:,:,:,newaxis]
            self.test_labels = asarray(self.label[self.n_train:])  # --- 0. left_hand, 1. right_hand, 2.left_forearm, 3. right_forearm
            self.test_locations = None
        else:
            print('[dataset id {}] the size of n_train and n_test was not set!'.format(dataset_id))

    def split_dataset_nongroup(self, dataset: str, dataset_id: int):
        self.n_train:int = self.data[dataset].shape[0] - 22671
        self.n_test:int  = self.data[dataset].shape[0] - self.n_train  #  hier ist 22671
        if (self.n_train > 0) and (self.n_test > 0):
            # train data
            self.train_images = self.data[dataset][:self.n_train]  #  (73969, 7, 8, 120, 160)
            self.train_labels = asarray(self.label[:self.n_train])  # --- 0. left_hand, 1. right_hand, 2.left_forearm, 3. right_forearm
            self.train_locations = None
            # test data
            self.test_images = self.data[dataset][self.n_train:]
            # self.test_images = test_images[:, :, :, :, :, newaxis]
            self.test_labels = asarray(self.label[self.n_train:])  # --- 0. left_hand, 1. right_hand, 2.left_forearm, 3. right_forearm
            self.test_locations = None
        else:
            print('[dataset id {}] the size of n_train and n_test was not set!'.format(dataset_id))
