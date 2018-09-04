import chainer
import chainer.functions as F
from chainer import Variable
from operator import itemgetter
import numpy as np

class DCGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.n_hidden = kwargs.pop('n_hidden')
        self.n_sigma = kwargs.pop('n_sigma')
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real, y_class, labels):
        batchsize = len(y_fake)

        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        L4 = F.softmax_cross_entropy(y_class,labels)/batchsize
        loss = L1 + L2 + L4# + L3 + L4
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake, x_class, labels):
        batchsize = len(y_fake)

        L1 = F.sum(F.softplus(-y_fake)) / batchsize
        L2 = F.softmax_cross_entropy(x_class, labels)
        loss = L1 + L2
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        images = [batch[i][0] for i in range(batchsize)]
        labels = [batch[i][1] for i in range(batchsize)]

        x_real = Variable(self.converter(images, self.device)) / 255.
        x_label = Variable(self.converter(labels, self.device))
        xp = chainer.cuda.get_array_module(x_real.data)

        xp_label = chainer.cuda.get_array_module(x_label.data)
        labels = xp_label.asarray(labels)
        n_class = 10

        temp = xp.array(batchsize * [self.n_hidden * [0]]).astype(xp.float32)
        for i in range(batchsize):
            temp[i] = xp.random.normal((labels[i] * 2 / n_class - 1) + (1 / n_class), (self.n_sigma), size=(self.n_hidden)).astype(xp.float32)

        gen, dis = self.gen, self.dis
        y_real, y_class = dis(x_real)

        x_fake = gen(temp)
        y_fake, x_class = dis(x_fake)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real, y_class, labels)
        gen_optimizer.update(self.loss_gen, gen, y_fake, x_class, labels)
        #dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        #gen_optimizer.update(self.loss_gen, gen, y_fake)
