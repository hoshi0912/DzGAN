import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

def lindim(shape, scale, n):
    w, h = shape
    return (w // scale) * (h // scale) * n


def convdim(shape, scale, n):
    w, h = shape
    return (n, w // scale, h // scale)




class Generator(chainer.Chain):
    def __init__(self, n_hidden, n_class, bottom_width=3, ch=512, wscale=0.02):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.ch = ch
        self.bottom_width = bottom_width

        with self.init_scope():
            self.fc1 = L.Linear(n_hidden, 1024)
            self.fc1_bn = L.BatchNormalization(1024)

            self.fc2 = L.Linear(1024, lindim((28, 28), 4, 128))
            self.fc2_bn = L.BatchNormalization(lindim((28, 28), 4, 128))

            self.dc1 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1)
            self.dc1_bn = L.BatchNormalization(64)

            self.dc2 = L.Deconvolution2D(64, 1, 4, stride=2, pad=1)

    def make_hidden(self, batchsize, n_class, lavels, xp):
        #temp = xp.array(batchsize*[self.n_hidden*[1*[1*[0]]]]).astype(numpy.float32)
        temp = xp.array(batchsize * [self.n_hidden*[0]]).astype(xp.float32)
        for i in range(batchsize):
            temp[i] = xp.random.uniform(lavels[i]*2/n_class-1, lavels[i]*2/n_class+(2/n_class)-1, self.n_hidden).astype(xp.float32)
        return temp

    def make_visualize_hidden(self, batchsize,n_sigma):
        temp = numpy.array(batchsize * [self.n_hidden*[0]]).astype(numpy.float32)
        #class
        for i in range(batchsize):
            temp[i] = numpy.random.normal((i * 2 / batchsize - 1) + (1 / batchsize), (n_sigma), size=(self.n_hidden)).astype(numpy.float32)
        return temp

    def __call__(self, z):
        h = F.relu(self.fc1_bn(self.fc1(z)))
        h = F.relu(self.fc2_bn(self.fc2(h)))
        h = F.reshape(h, (z.shape[0],) + convdim((28, 28), 4, 128))
        h = F.relu(self.dc1_bn(self.dc1(h)))
        h = F.sigmoid(self.dc2(h))
        return h


class Discriminator(chainer.Chain):
    def __init__(self, bottom_width=3, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(1, 64, 4, stride=2, pad=1)

            self.c2 = L.Convolution2D(64, 128, 4, stride=2, pad=1)
            self.c2_bn = L.BatchNormalization(128)

            self.fc1 = L.Linear(lindim((28, 28), 4, 128), 1024)
            self.fc1_bn = L.BatchNormalization(1024)

            # Real/Fake prediction
            self.fc_d = L.Linear(1024, 2)

            # Mutual information reconstruction
            self.fc_mi1 = L.Linear(1024, 128)
            self.fc_mi1_bn = L.BatchNormalization(128)

            self.fc_mi2 = L.Linear(128, 10)


    def __call__(self, x):
        h = F.leaky_relu(self.c1(x), slope=0.2)
        h = F.leaky_relu(self.c2_bn(self.c2(h)), slope=0.2)
        h = F.leaky_relu(self.fc1_bn(self.fc1(h)), slope=0.2)

        d = self.fc_d(h)

        mi = F.leaky_relu(self.fc_mi1_bn(self.fc_mi1(h)), slope=0.2)
        mi = self.fc_mi2(mi)
        return d, mi
