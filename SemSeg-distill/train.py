from options import TrainOptions
from torch.utils import data
from dataset.datasets import CSTrainValSet
from networks.kd_model import NetModel
from utils.evaluate import evaluate_main
import os
import warnings
warnings.filterwarnings("ignore")
import logging

# for reproducibility
import random
import numpy as np
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

args = TrainOptions().initialize()
# device
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# data
h, w = map(int, args.input_size.split(','))
trainloader = data.DataLoader(CSTrainValSet(args.data_dir, args.data_list, max_iters=args.num_steps*args.batch_size, 
                                            crop_size=(h, w), scale=args.random_scale, mirror=args.random_mirror), 
                                            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
valloader = data.DataLoader(CSTrainValSet(args.data_dir, args.data_listval, crop_size=(1024, 2048), scale=False, mirror=False),
                            batch_size=1, shuffle=False, pin_memory=True)
# model
model = NetModel(args)
# train
for step, data in enumerate(trainloader, args.last_step):
    model.adjust_learning_rate(args.lr_g, model.G_solver, step)
    model.adjust_learning_rate(args.lr_d, model.D_solver, step)
    model.set_input(data)
    model.optimize_parameters()
    model.print_info(step)
    if ((step + 1) >= args.save_ckpt_start) and ((step + 1 - args.save_ckpt_start) % args.save_ckpt_every == 0):
        model.save_ckpt(step)
        mean_IU, IU_array = evaluate_main(model.student, valloader, '512,512', args.num_classes, True, 1, 'val')
        logging.info('mean_IU: {:.6f}  IU_array: \n{}'.format(mean_IU, IU_array))
