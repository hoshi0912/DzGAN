import argparse
import os
import numpy as np
import chainer
from chainer import training
from chainer.training import extensions

from net_mnist import Discriminator
from net_mnist import Generator
from updater import DCGANUpdater
from visualize import out_generated_image




def main():
    parser = argparse.ArgumentParser(description='Chainer: DZGAN MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', type=str, default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--disresume', '-d', type=str, default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--genresume', '-gen', type=str, default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--n_class', '-c', type=int, default=0,
                        help='class num')
    parser.add_argument('--n_sigma', '-s', type=float, default=0,
                        help='sigma param')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('')
    #add command temp
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    path = str(args.out)+'/lastcommand.txt'
    with open(path, mode='w') as f:
        f.write(str(args))
    f.close()

    # Set up a neural network to train

    gen = Generator(n_hidden=args.n_hidden, n_class=args.n_class)
    dis = Discriminator()

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer_gen(model, alpha=2e-4, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        #optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    def make_optimizer_dis(model, alpha=2e-4, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        #optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer

    opt_gen = make_optimizer_gen(gen)
    opt_dis = make_optimizer_dis(dis)

    # Load the MNIST dataset
    train, _ = chainer.datasets.get_mnist(withlabel=True, ndim=3, scale=255.) # ndim=3 : (ch,width,height)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    # Set up a trainer
    updater = DCGANUpdater(models=(gen, dis), iterator=train_iter, optimizer={'gen':opt_gen, 'dis':opt_dis}, device=args.gpu, n_hidden=args.n_hidden, n_sigma=args.n_sigma)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    epoch_interval = (1, 'epoch')
    visual_interval = (1, 'epoch')
    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    # trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    # trainer.extend(extensions.snapshot_object(gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    # trainer.extend(extensions.snapshot_object(dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
    trainer.extend(extensions.snapshot_object(gen, 'gen_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
    trainer.extend(extensions.snapshot_object(dis, 'dis_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'gen/loss', 'dis/loss',]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(out_generated_image(gen, dis, 10, 20, args.seed, args.out,args.n_sigma), trigger=visual_interval)
    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    if args.disresume:
        chainer.serializers.load_npz(args.disresume, dis)

    if args.genresume:
        chainer.serializers.load_npz(args.genresume, gen)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
