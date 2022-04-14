#from concat_mat.matConcat import MatConcat
from cython_modules.concat_images.concat_np.npConcat import NPConcat
from h5py import File
from dataset_tools.h5Writer import  H5Writer
from numpy import ones,zeros,uint8, float64, concatenate,asarray
#o = ones((9000,160,120), dtype=uint8)
#z = zeros((9000,160,120), dtype=uint8)
#zo = concatenate((o,z), axis=0)
#print(zo.shape)

#img = MatConcat(o,z)
#img2 = NPConcat(o,z)
#print(img2.get_images().shape)


if __name__ == '__main__':

    path:str = r"training_data/binocular_image_data.h5"
    arr = File(path, 'r')
    left_img= arr['binocular_image']['left_img']
    right_img= arr['binocular_image']['right_img']
    img = NPConcat(asarray(left_img, dtype=uint8), asarray(right_img,dtype=uint8))
    print(img.get_images())
    binocular_images = img.get_images().shape


    path = r"./training_data/binocular_images.h5"
    writer: H5Writer = H5Writer(path)
    datasets = ["binocular_images"]  # "disparity_arrays_s"
    data = [ asarray(binocular_images, dtype=uint8)]  # asarray(self.disparity_array, dtype=uint8)
    writer.saveImgDataIntoGroup(data, "binocular", datasetNames=datasets)
