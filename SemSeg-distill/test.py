import os
import torch
from options import TestOptions
from torch.utils import data
from dataset.datasets import CSTestSet
from networks.pspnet import Res_pspnet, BasicBlock, Bottleneck
from utils.evaluate import evaluate_main
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    args = TestOptions().initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    testloader = data.DataLoader(CSTestSet(args.data_dir, args.data_list, crop_size=(1024, 2048)), 
                                 batch_size=1, shuffle=False, pin_memory=True)
    student = Res_pspnet(BasicBlock, [2, 2, 2, 2], num_classes = args.num_classes)
    student.load_state_dict(torch.load(args.restore_from))
    evaluate_main(student, testloader, '512,512', args.num_classes, True, 1, 'test')
