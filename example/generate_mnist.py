import matplotlib.pyplot as plt
import math
import numpy as np
import cv2

import gluoncv
import mxnet as mx

batch_size = 9
rand_shape = (batch_size, 100)
data_shape = (batch_size, 1, 28, 28)
context = mx.gpu()

model_prefix = "./model/generator"
model_epoch = 49

mod = mx.module.Module.load( prefix=model_prefix, epoch=model_epoch, data_names=['code'] )
# sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, model_epoch)

print( mod.symbol.list_arguments() )
print( mod.params_initialized )
# mod need to be initialized and binded
if not mod.binded:
    mod.bind(data_shapes=[("code", rand_shape)])


rnd_data = mx.random.normal(0, 1.0, shape=rand_shape)
curr_batch = mx.io.DataBatch(
                [mx.nd.zeros(rand_shape, ctx=context )], None)

curr_batch.data[0] = rnd_data
mod.forward( curr_batch, is_train=False )
output = mod.get_outputs()
output = output[0].asnumpy()

output = output.transpose( (0,2,3,1) )
output = output.squeeze( axis=3 )
print( output.shape )

_, h, w = output.shape
side_len = int( math.sqrt( batch_size-1 ) ) + 1
result = np.zeros( (h*side_len, w*side_len), dtype=np.float )
for ii in range( batch_size ):
    r = ii // side_len
    c = ii % side_len
    result[ r*h:(r+1)*h, c*w:(c+1)*w ] = output[ii]

result = (result - np.min(result)) / (np.max(result) - np.min(result))
#cv2.imshow("result", result)
#cv2.waitKey()
plt.imshow( result )
plt.show()
