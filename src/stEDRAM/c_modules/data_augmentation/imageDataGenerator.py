import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from typing import Tuple, List
from random import randint
from numpy import ndarray, asarray, float64, uint8
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
#from .imgDataGenerator.imageDataGenerator import ImgAugmentation
import cv2


class ImageDataGenerator(object):
    image:ndarray
    _images:List[List]
    def __init__(self, images, rotation_range, horizontal_flip, vertical_flip, shear_range,zoom_range,
                          noise_range, bright_range):
        self._images = list()
        # print("shape: ",self.images.shape)
        if (images.size != 0):
            self.d = images.shape[0]
            self.d1 = images.shape[1]
            self.d2 = images.shape[2]
            self.select_algorithm(images, rotation_range, horizontal_flip, vertical_flip, shear_range, zoom_range, noise_range,
                                  bright_range)

            # print("shape", self.d, self.d1, self.d2)
    def get_img_array(self):
        return asarray(self._images, dtype=float64)
    @staticmethod
    def rand_num():
        return randint(1,7)
    @staticmethod
    def bounding_box( image):
        return BoundingBoxesOnImage([BoundingBox(x1=10, x2=520, y1=10, y2=300)], shape=image.shape)
    def select_algorithm(self, images, rotation_range, horizontal_flip, vertical_flip, shear_range,zoom_range,
                          noise_range, bright_range):

        for i in range(self.d):
            rand_num = self.rand_num()
            seq=[]
            for j in range(self.d1):
                sm=[]
                for k in range(self.d2):
                    image = asarray(images[i, j, k], dtype=uint8)
                    self.image = self._select_algorithm(image, rand_num, rotation_range, horizontal_flip, vertical_flip, shear_range,
                                           zoom_range, noise_range, bright_range)
                    #print(i, len(self.image), len(self.image[0]))
                    sm.append(asarray(self.image, dtype=float64))
                seq.append(asarray(sm, dtype=float))
            self._images.append(seq)

    def _select_algorithm(self, img, rand_num,rotation_range, horizontal_flip, vertical_flip, shear_range,zoom_range,
                          noise_range, bright_range):
        #print("rand number", rand_num)
        if (rand_num == 1):
            return self._rotation(img, -50, rotation_range)
        elif (rand_num == 2):
            return self._flipping(img, horizontal_flip)
        elif (rand_num == 3):
            return self._shearing(img, 0, shear_range)
        elif (rand_num == 4):
            return self._flipup(img, vertical_flip)
        elif (rand_num == 5):
            return self._cropping(img, 0.0, zoom_range)
        elif (rand_num == 6):
            return self._add_noise(img, noise_range, noise_range * 2)
        elif (rand_num == 7):
            return self._brightness(img, bright_range)

    def _rotation(self, image, e_1, e_2):
        rotate = iaa.Affine(rotate=(e_1, e_2));
        return rotate.augment_image(image)
    def _flipping(self, image, p):
        flip_hr = iaa.Fliplr(p=p)
        return flip_hr.augment_image(image)
    def _shearing(self, image, s_1,s_2):
        shear = iaa.Affine(shear=(s_1, s_2))
        #print("[Error]", shear.augment_image(image))
        return shear.augment_image(image)
        #print(self.image.shape)
    def _flipup(self, image, p ):
        flip_vr = iaa.Flipud(p=p)
        return flip_vr.augment_image(image)
    def _cropping(self, image, per_1, per_2):
        crop = iaa.Crop(percent=(per_1, per_2))  # crop image
        return crop.augment_image(image)
    def _brightness(self, image, gamma):
        contrast = iaa.GammaContrast(gamma=gamma)
        return contrast.augment_image(image)
    def _add_noise(self, image, elem_1, elem_2):
        gaussian_noise = iaa.AdditiveGaussianNoise(elem_1, elem_2)
        return gaussian_noise.augment_image(image)


# path = r"spatial_transformer.png"
#
# image = imageio.imread(path)
# image = asarray(image)[:,:,0:1]
# image = cv2.resize(image, (120,160))
# print(ia.is_np_array(image))
# shear = iaa.Affine(shear=(0,40))
# shear_image=shear.augment_image(image)
# ia.imshow(shear_image)


if __name__ == '__main__':
   pass