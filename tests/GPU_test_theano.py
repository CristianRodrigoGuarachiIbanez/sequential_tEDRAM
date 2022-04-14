"""
    This tests wether we can use the GPU (source: deeplearning.net/software/theano/tutorial/using_gpu.html)

    Use like this:

    $ THEANO_FLAGS=device=cpu python gpu_tutorial1.py

    [Elemwise{exp,no_inplace}(<TensorType(float64, vector)>)]
    Looping 1000 times took 2.271284 seconds
    Result is [ 1.23178032  1.61879341  1.52278065 ...,  2.20771815  2.29967753
      1.62323285]
    Used the cpu

    $ THEANO_FLAGS=device=cuda0 python gpu_tutorial1.py

    Using cuDNN version 5105 on context None
    Mapped name None to device cuda0: GeForce GTX 750 Ti (0000:07:00.0)
    [GpuElemwise{exp,no_inplace}(<GpuArrayType<None>(float64, (False,))>), HostFromGpu(gpuarray)(GpuElemwise{exp,no_inplace}.0)]
    Looping 1000 times took 1.697514 seconds
    Result is [ 1.23178032  1.61879341  1.52278065 ...,  2.20771815  2.29967753
      1.62323285]
    Used the gpu

"""


from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()

for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')