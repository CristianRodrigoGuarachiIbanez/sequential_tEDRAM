

from stEDRAM.sequenceConstructor import *
import unittest


dataset_size = 200 # training or evaluation dateset size
imgs_size = (dataset_size*2, 120, 160, 1)

img_size = (120, 160, 1)
n_step = 5
classes = 6
batch_size = 5
seq_size = 5
labels_size = (dataset_size*2, classes)
class TestSequenceConstructor(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSequenceConstructor, self).__init__(*args, **kwargs)

        self.indices_ = SequenceConstructor(dataset_size, n_step, classes)  # 6
        self.data_ = zeros(imgs_size)
        self.labels_ = zeros(labels_size)
        self.imgs_ = self.indices_.samples(self.data_, start=0, end=n_step)

    def test_random_matrix(self):
        return self.assertEqual(self.indices_.getMatrix().shape, (dataset_size//classes, n_step))

    def test_extracted_sample_images_size(self):
        return self.assertEqual(self.imgs_.shape, (batch_size, seq_size, img_size[0], img_size[1], img_size[2]), "the size of the sampled input images batch_size x seq size x dim")

    def test_extracted_sample_labels_size(self):
        labels = self.indices_.labels(labels=self.labels_, start=0, end=n_step)
        return self.assertEqual(labels.shape, (batch_size, seq_size, classes))

if __name__ == "__main__":
    unittest.main()
    #
    # s = SequenceConstructor(dataset_size, n_step, classes)
    # print("RANDOM  MATRIX", s.getMatrix().shape)
    # data_ = zeros(imgs_size)
    # d = s.samples(data_, 0, n_step)
    # labels_ = zeros(labels_size)
    # l = s.labels(labels=labels_, start=0, end=n_step)
    # print(labels_.shape)


