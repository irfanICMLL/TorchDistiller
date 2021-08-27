import argparse
import torch
import time
import logging
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# dataset
DATASET = 'city'
NUM_CLASSES = 19
DATA_DIRECTORY = '/workdir/cityscapes'
DATA_LIST_TRAIN_PATH = './dataset/list/cityscapes/train.lst'
DATA_LIST_VAL_PATH = './dataset/list/cityscapes/val.lst'
DATA_LIST_TEST_PATH = './dataset/list/cityscapes/test.lst'
IGNORE_LABEL = 255
INPUT_SIZE = '512,512'

# init params
T_CKPT = './ckpt/teacher_city.pth'
S_CKPT = './ckpt/resnet18-imagenet.pth'

# training params
BATCH_SIZE = 8
NUM_STEPS = 40000
MOMENTUM = 0.9
POWER = 0.9
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 0.0005

# save params
SAVE_CKPT_START = 40000
SAVE_CKPT_EVERY = 1000

def log_init(log_dir, name='log'):
    time_cur = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_dir + '/' + name + '_' + str(time_cur) + '.log',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description='train')
        parser.add_argument('--local_rank', default=0, type=int)
        # dataset
        parser.add_argument('--data_set', type=str, default=DATASET, help='The name of the dataset.')
        parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, help='Number of classes to predict.')
        parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY, help="Path to the directory containing the dataset.")
        parser.add_argument("--data-list", type=str, default=DATA_LIST_TRAIN_PATH, help="Path to the file listing the images in the dataset.")
        parser.add_argument("--data-listval", type=str, default=DATA_LIST_VAL_PATH, help="Path to the file listing the images in the dataset.")
        parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL, help="The index of the label to ignore during the training.")
        parser.add_argument("--input-size", type=str, default=INPUT_SIZE, help="Comma-separated string with height and width of images.")
        parser.add_argument("--random-mirror", action="store_true", help="Whether to randomly mirror the inputs during the training.")
        parser.add_argument("--random-scale", action="store_true", help="Whether to randomly scale the inputs during the training.")

        # init params
        parser.add_argument('--T_ckpt_path', type=str, default=T_CKPT, help='teacher ckpt path')
        parser.add_argument('--S_resume', type=str2bool, default='False', help='is or not use student ckpt')
        parser.add_argument('--S_ckpt_path', type=str, default='', help='student ckpt path')
        parser.add_argument('--D_resume', type=str2bool, default='False', help='is or not use discriminator ckpt')
        parser.add_argument('--D_ckpt_path', type=str, default='', help='discriminator ckpt path')
        parser.add_argument("--is-student-load-imgnet", type=str2bool, default='True', help="is student load imgnet")
        parser.add_argument("--student-pretrain-model-imgnet", type=str, default=S_CKPT, help="student pretrain model on imgnet")

        # training params
        parser.add_argument("--gpu", type=str, default='None', help="Choose gpu device.")
        parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Number of images sent to the network in one step.")
        parser.add_argument("--num-steps", type=int, default=NUM_STEPS, help="Number of training steps.")
        parser.add_argument("--momentum", type=float, default=MOMENTUM, help="Momentum component of the optimiser.")
        parser.add_argument("--power", type=float, default=POWER, help="Decay parameter to compute the learning rate.")
        parser.add_argument("--lr-g", type=float, default=LEARNING_RATE, help="learning rate for G")
        parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--last-step", type=int, default=0, help="last train step.")

        # distiller

        parser.add_argument("--kd", type=str2bool, default='True')
        parser.add_argument("--lambda-kd", type=float, default=10.0, help="lambda_kd")


        parser.add_argument("--cwd", type=str2bool, default='True')
        parser.add_argument("--cwd-feat", type=str2bool, default='False')
        parser.add_argument("--temperature", type=float, default=1.0, help="normalize temperature")
        parser.add_argument("--lambda-cwd", type=float, default=1.0, help="lambda_kd")
        parser.add_argument("--norm-type", type=str, default='none', help="kd normalize setting")
        parser.add_argument("--divergence", type=str, default='kl', help="kd divergence setting")

        parser.add_argument("--adv", type=str2bool, default='False')
        parser.add_argument("--lambda-adv", type=float, default=0.001, help="lambda_adv")
        parser.add_argument("--preprocess-GAN-mode", type=int, default=1, help="preprocess-GAN-mode should be tanh or bn")
        parser.add_argument("--adv-loss-type", type=str, default='wgan-gp', help="adversarial loss setting")
        parser.add_argument("--imsize-for-adv", type=int, default=65, help="imsize for addv")
        parser.add_argument("--adv-conv-dim", type=int, default=64, help="conv dim in adv")
        parser.add_argument("--lambda-gp", type=float, default=10.0, help="lambda_gp")
        parser.add_argument("--lambda-d", type=float, default=0.1, help="lambda_d")
        parser.add_argument("--lr-d", type=float, default=4e-4, help="learning rate for D")

        parser.add_argument("--ifv", type=str2bool, default='False')
        parser.add_argument('--lambda-ifv', type=float, default=200.0, help='lambda_ifv')

        # save params
        parser.add_argument("--save-name", type=str, default='exp')
        parser.add_argument("--save-dir", type=str, default='ckpt', help="Where to save models.")
        parser.add_argument("--save-ckpt-start", type=int, default=SAVE_CKPT_START)
        parser.add_argument("--save-ckpt-every", type=int, default=SAVE_CKPT_EVERY)

        args = parser.parse_args()

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_path = args.save_dir + '/' + args.save_name
        log_init(args.save_path, args.data_set)

        for key, val in args._get_kwargs():
            logging.info(key+' : '+str(val))

        return args

class ValOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description='Val')
        parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY)
        parser.add_argument("--data-list", type=str, default=DATA_LIST_VAL_PATH)
        parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
        parser.add_argument("--restore-from", type=str, default='')
        parser.add_argument("--gpu", type=str, default='None')

        args = parser.parse_args()

        for key, val in args._get_kwargs():
            print(key+' : '+str(val))

        return args

class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description='Test')
        parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY)
        parser.add_argument("--data-list", type=str, default=DATA_LIST_TEST_PATH)
        parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
        parser.add_argument("--restore-from", type=str, default='')
        parser.add_argument("--gpu", type=str, default='None')

        args = parser.parse_args()

        for key, val in args._get_kwargs():
            print(key+' : '+str(val))

        return args
