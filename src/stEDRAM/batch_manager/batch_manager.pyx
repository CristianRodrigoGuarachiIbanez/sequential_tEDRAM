from lib.batch_manager import *
# from sequenceConstructor cimport SequenceConstructor
from libc.string cimport memcpy
from libcpp cimport bool
from numpy cimport uint8_t, float32_t,ndarray as ndarray_t
from numpy import ndarray, dstack, ascontiguousarray, uint8, float32, asarray, hstack, reshape
from cython cimport boundscheck, wraparound

typedef unsigned char

cdef class BatchManager:
    cdef:
        SequenceConstructor*indices_ptr_
        PyImageDataGenerator*imagesAug_ptr_
        uint8_t[:,:,:,:] input_images_
        float32_t[:,:] A_, S1_, S2_
        float32_t[:,:,:,:] B1_, B2_, B3_, B4_, B5_, B6_
        int[:,:,:] Y_cla_
        float32_t[:,:,:] Y_loc_


    cdef __cinit__(self, uint8_t[:,:,:,:]&features, int scale, int dataset_size, int n_steps, int scale, int start, int end ):
         self.indices_ptr_ = new SequenceConstructor(dataset_size, n_steps, n_steps);

    cdef  SequenceConstructor miniBatchInput(self, uint8_t[:,:,:,:]&features, int scale, int start, int end):
         if scale != 1 and scale != 0:
            return self.indices_ptr_.samples(features, start, end)/scale
        if normalize is True:
            return (self.indices_ptr_.samples(features, start, end)-mean)/std
        else:
            return self.indices_ptr_.samples(features, start, end)

    cdef uint8_t[:,:] transformationMatrix(self, int batch_size):
        # transformation matrix with zoom paramters set to 1
        A = zeros((batch_size, 6), dtype='float32')
        A[:, (0,4)] = 1
        return A

    cdef uint8_t[:,:] locationMatrix(self, uint8_t[:,:]&locations, int zoom, int batch_size):
        uint8_t[:,:] Y_loc
        if zoom == 1:
            # location matrix could be not defined
            if locations.size == 0:
                Y_loc = zeros((batch_size, 6), dtype='float32')
                return Y_loc[:, (0, 4)] = zoom
            else:
                return Y_loc = indices.samples(locations, start, end);
        else:
            # localization matrix is definitively not defined N x 10 x 6
            if locations.size == 0:
                Y_loc = zeros((batch_size, 6), dtype='float32')
                return Y_loc[:, (0, 4)] = zoom

    cdef void augmentation(self, uin8_t[:,:,:,:,:]&I):
        # transposed matrix N x 10 x 120 x 160 x 56 -> N x 10 x 56 x 120 x 160 will be augmented according to the last dim

        if I.shape[-1] == 100: # N x 10 x 1 x 100 x 100
            imagesAug_ptr_ = new PyImageDataGenerator(I, 30.0, 88, 98, 2.0, 2, 70, 40.0)
        else:
            imagesAug_ptr_ = new PyImageDataGenerator(I, 30.0, 90, 100, 2.0, 2, 70, 40.0)

        return transpose(imagesAug_ptr_.getAugmentedImages(), (0, 1, 3, 4, 2))

    cdef void mainLoop(self, uint8_t[:,:,:,:]&features, uint8_t[:,:]&locations, int[2] init_state_size, int pre_minibatch_size, int scale, int start, int end,
                       int[2] glimpse_size, int zoom, int mode, bool augment,  ):

        cdef:
             int i

        state_size_1 = init_state_size[0]
        state_size_2 = init_state_size[1]
        for i in range(pre_minibatch_size):

            self.input_images_ = self.miniBatchInput(features, scale, start, end)
            # transformation matrix with zoom paramters set to 1
            self.A_ = self.transformationMatrix(batch_size)

            # initial RNN states
            self.S1_ = zeros((batch_size, state_size_1), dtype='float32')
            self.S2_ = zeros((batch_size, state_size_2), dtype='float32')

            # Biases
            if glimpse_size[0] == 26 and glimpse_size[1] == 26:
                self.B1_ = ones((batch_size, 26, 26, 1), dtype='float32')
                self.B2_ = ones((batch_size, 24, 24, 1), dtype='float32')
                self.B3_ = ones((batch_size, 12, 12, 1), dtype='float32')
                self.B4_ = ones((batch_size, 8, 8, 1), dtype='float32')
                self.B5_ = ones((batch_size, 6, 6, 1), dtype='float32')
                self.B6_ = ones((batch_size, 4, 4, 1), dtype='float32')
            else:
                self.B1_ = ones((batch_size, 16, 16, 1), dtype='float32')
                self.B2_ = ones((batch_size, 16, 16, 1), dtype='float32')
                self.B3_ = ones((batch_size, 8, 8, 1), dtype='float32')
                self.B4_ = ones((batch_size, 8, 8, 1), dtype='float32')
                self.B5_ = ones((batch_size, 6, 6, 1), dtype='float32')
                self.B6_ = ones((batch_size, 4, 4, 1), dtype='float32')


            # target outputs
            # labels
            self.Y_cla_ = self.indices_ptr_.labels(labels, start, end)  # sequenz_size x batch_size x categories -> 40 x 2 x 6

            self.Y_loc_ = self.locationMatrix(locations, zoom, batch_size)

            # when using all outputs for training
            if mode is True:
                self.Y_loc_ = reshape(self.Y_loc_, (batch_size, 1, 6))
                self.Y_loc_ = hstack([Y_loc for _ in range(0, n_steps+mode2)])

                if (n_steps > 1 and not mode3) and (Y_cla.ndim == 2):
                    Y_cla = reshape(Y_cla, (batch_size,1,Y_cla.shape[1]))
                    Y_cla = hstack([Y_cla for _ in range(0, n_steps)])


            if augment is True:


        else:
            if I.shape[2] == 56 and I.shape[3] == 120: # N x 10 x 56 x 120 x 160 -> N x 10 x 120 x 160 x 56
                I = transpose(I, (0, 1, 3, 4, 2))

            elif I.shape[2] == 7 and I.shape[3] == 120: # N x 10 x 7 x 120 x 160 -> N x 10 x 120 x 160 x 156
                I = transpose(I, (0, 1, 3, 4, 2))

            elif I.shape[2] == 100 and I.shape[3] == 100:
                pass


   if I.shape[-1] < 160: # N x 10 x 120 x 160 x 56 -> N x 10 x 56 x 120 x 160 oder N x 10 x 100 x 100 x 1 -> N x 10 x 1 x 100 x 100
            print( "[Debug]", I.shape)
            I = transpose(I, (0, 1, 4, 2, 3))

         if I.dtype != uint8:
            I = asarray(I, dtype=uint8)