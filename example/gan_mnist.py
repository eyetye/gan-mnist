import logging
import numpy as np
import mxnet as mx
import sys

sys.path.append(".")

from mxgan import module, generator, encoder, viz

def ferr(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return np.abs(label - (pred > 0.5)).sum() / label.shape[0]

lr = 0.0005
beta1 = 0.5
batch_size = 500
# generate batch_size samples, each dimension is 1X100
rand_shape = (batch_size, 100)
num_epoch = 50 # 00
data_shape = (batch_size, 1, 28, 28)
context = mx.gpu()

save_prefix = "./model/generator"

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')
sym_gen = generator.dcgan28x28(oshape=data_shape, ngf=32, final_act="sigmoid")

gmod = module.GANModule(
    sym_gen,
    symbol_encoder=encoder.lenet(),
    context=context,
    data_shape=data_shape,
    code_shape=rand_shape)

gmod.init_params(mx.init.Xavier(factor_type="in", magnitude=2.34))

gmod.init_optimizer(
    optimizer="adam",
    optimizer_params={
        "learning_rate": lr,
        "wd": 0.,
        "beta1": beta1,
})

data_dir = '../dataset/mnist/'
train = mx.io.MNISTIter(
    image = data_dir + "train-images-idx3-ubyte",
    label = data_dir + "train-labels-idx1-ubyte",
    input_shape = data_shape[1:],
    batch_size = batch_size,
    shuffle = True)

metric_acc = mx.metric.CustomMetric(ferr)

for epoch in range(num_epoch):
    train.reset()
    metric_acc.reset()
    for t, batch in enumerate(train):
        gmod.update(batch)
        gmod.temp_label[:] = 0.0
        metric_acc.update([gmod.temp_label], gmod.outputs_fake)
        gmod.temp_label[:] = 1.0
        metric_acc.update([gmod.temp_label], gmod.outputs_real)

        if t % 100 == 0:
            logging.info("epoch: %d, iter %d, metric=%s", epoch, t, metric_acc.get())

            viz.imshow("gout", gmod.temp_outG[0].asnumpy(), 2)
            diff = gmod.temp_diffD[0].asnumpy()
            diff = (diff - diff.mean()) / diff.std() + 0.5
            viz.imshow("diff", diff)
            viz.imshow("data", batch.data[0].asnumpy(), 2)
    if (epoch+1) % 10 == 0:        
        gmod.modG.save_checkpoint(save_prefix, epoch)




