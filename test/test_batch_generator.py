import stEDRAM.batch_generator as bg
import numpy as np
import unittest

dataset_size = 100
batch_size = 5
init_state_size = (512, 512)
n_steps = 5 # or seq_size
n_classes = 6
features: np.ndarray = np.zeros((100, 120, 160, 1))
labels = np.ones((100, 10, 6))
locations = None # np.empty((batch_size, 6))

augment = False
scale = 1
normalize = False
mean = 0
std = 0
mode = 0
mode2 = 0
mode3 = 0
model_id = 1
glimpse_size = (26, 26)
zoom = 1


class TestBatchGenerator(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestBatchGenerator, self).__init__(*args, **kwargs)
        self.batch_generator_ = bg.batch_generator(dataset_size, batch_size, n_steps, n_classes, init_state_size, features, labels, locations, augment,
                               scale, normalize, mean, std, mode, mode2, mode3, model_id, glimpse_size, zoom)
        self.test_ = next(self.batch_generator_)
        self.input_tests_ = self.multiple_input_tests(5)

    def multiple_input_tests(self, size):
        count = 0
        tests_ = []
        while count > size:
            tmp = next(self.batch_generator_)[0].get("input_image")
            tests_.append(tmp.size ==(batch_size, n_steps, 120, 160, 1))
        return tests_

    def multiple_output_tests(self, size):
        count = 0
        tests_ = []
        while count > size:
            tmp = next(self.batch_generator_)[1].get("classifications")
            tests_.append(tmp.size ==(batch_size, n_steps, 120, 160, 1))
        return tests_

    def test_multiple_input_images(self):
        return self.assertEqual(all(self.input_tests_), True, "all True")


    def test_input_image_shape(self):
        # test1 = next(self.batch_generator_)
        img = self.test_[0].get("input_image")
        return self.assertEqual(img.shape, (batch_size, n_steps, 120, 160, 1), "the dimensions of input images set")

    def test_input_matrix_(self):
        # test1 = next(self.batch_generator_)
        img = self.test_[0].get("input_matrix")
        return self.assertEqual(img.shape, (batch_size, n_classes), "the dims of input matrix")

    def test_initial_hidden_state_1(self):
        # test = next(self.batch_generator_)
        img = self.test_[0].get("initial_hidden_state_1")
        return self.assertEqual(img.shape, (batch_size, init_state_size[0]), "dim of initial_hidden_state_1")

    def test_initial_cell_state_1(self):
        # test = next(self.batch_generator_)
        img = self.test_[0].get("initial_cell_state_2")
        return self.assertEqual(img.shape, (batch_size, init_state_size[1]), "dim of initial_cell_state_2")

    def test_bias_26(self):
        # test = next(self.batch_generator_)
        img = self.test_[0].get("b26")
        return self.assertEqual(img.shape, (batch_size, 26, 26, 1) , "dims of the B1")

    def test_bias_24(self):
        # test = next(self.batch_generator_)
        img = self.test_[0].get("b24")
        return self.assertEqual(img.shape, (batch_size, 24, 24, 1), "dims of the B2")

    def test_bias_12(self):
        # test = next(self.batch_generator_)
        img = self.test_[0].get("b12")
        return self.assertEqual(img.shape, (batch_size, 12, 12, 1), "dims of the B3")



if __name__ == "__main__":

    # L = bg.batch_generator(dataset_size, batch_size, n_steps, n_classes, init_state_size, features, labels, locations, augment, scale, normalize, mean, std, mode, mode2, mode3, model_id, glimpse_size, zoom)
    #
    # output = next(L)
    # img = output[0].get("input_image")
    # print(img.shape,(batch_size, n_steps, 120, 160, 1))
    unittest.main()

