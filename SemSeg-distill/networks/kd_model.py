import os
import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.backends.cudnn as cudnn
from libs import InPlaceABN, InPlaceABNSync
from networks.pspnet import Res_pspnet, BasicBlock, Bottleneck
from networks.sagan_models import Discriminator
from utils.criterion import CriterionDSN,CriterionCWD, CriterionKD, CriterionAdv, CriterionAdvForG, CriterionAdditionalGP, CriterionIFV

def load_S_model(args, model):
    logging.info("------------")
    if args.is_student_load_imgnet:
        if os.path.isfile(args.student_pretrain_model_imgnet):
            saved_state_dict = torch.load(args.student_pretrain_model_imgnet)
            new_params=model.state_dict()
            saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in new_params}
            new_params.update(saved_state_dict)
            model.load_state_dict(new_params)
            logging.info("=> load" + str(args.student_pretrain_model_imgnet))
        else:
            logging.info("=> the pretrain model on imgnet '{}' does not exit".format(args.student_pretrain_model_imgnet))
    if args.S_resume:
        if os.path.isfile(args.S_ckpt_path):
            checkpoint = torch.load(args.S_ckpt_path)
            args.last_step = checkpoint['step'] if 'step' in checkpoint else None
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("=> loaded checkpoint '{}' \n (step:{} \n )".format(args.S_ckpt_path, args.last_step))
        else:
            logging.info("=> student checkpoint '{}' does not exit".format(args.S_ckpt_path))
    logging.info("------------")

def load_T_model(args, model):
    logging.info("------------")
    if os.path.isfile(args.T_ckpt_path):
        model.load_state_dict(torch.load(args.T_ckpt_path))
        logging.info("=> load" + str(args.T_ckpt_path))
    else:
        logging.info("=> teacher checkpoint '{}' does not exit".format(args.T_ckpt_path))
    logging.info("------------")

def load_D_model(args, model):
    logging.info("------------")
    if args.D_resume:
        if os.path.isfile(args.D_ckpt_path):
            checkpoint = torch.load(args.D_ckpt_path)
            args.last_step = checkpoint['step'] if 'step' in checkpoint else None
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("=> loaded checkpoint '{}' \n (step:{} \n )".format(args.D_ckpt_path, args.last_step))
        else:
            logging.info("=> checkpoint '{}' does not exit".format(args.D_ckpt_path))
    else:
        logging.info("=> train d from scratch")
    logging.info("------------")

def print_model_parm_nums(model, string):
    b = []
    for param in model.parameters():
        b.append(param.numel())
    logging.info(string + ': Number of params: %.2fM', sum(b) / 1e6)

def to_tuple_str(str_first, gpu_num, str_ind):
    if gpu_num > 1:
        tmp = '(' 
        for cpu_ind in range(gpu_num):
            tmp += '(' + str_first + '[' + str(cpu_ind) + ']' + str_ind +',)'  
            if cpu_ind != gpu_num-1: tmp +=  ', '
        tmp += ')'
    else:
        tmp = str_first + str_ind  
    return tmp

class NetModel():
    def name(self):
        return 'kd_seg'

    def __init__(self, args):
        self.args = args
        student = Res_pspnet(BasicBlock, [2, 2, 2, 2], num_classes = args.num_classes)
        load_S_model(args, student)
        print_model_parm_nums(student, 'student_model')
        student.cuda()
        self.student = student

        teacher = Res_pspnet(Bottleneck, [3, 4, 23, 3], num_classes = args.num_classes)
        load_T_model(args, teacher)
        print_model_parm_nums(teacher, 'teacher_model')
        teacher.cuda()
        self.teacher = teacher

        D_model = Discriminator(args.preprocess_GAN_mode, args.num_classes, args.batch_size, args.imsize_for_adv, args.adv_conv_dim)
        load_D_model(args, D_model)
        print_model_parm_nums(D_model, 'D_model')
        logging.info("------------")
        D_model.cuda()
        self.D_model = D_model

        self.G_solver = optim.SGD([{'params': filter(lambda p: p.requires_grad, student.parameters()), 'initial_lr': args.lr_g}], args.lr_g, momentum=args.momentum, weight_decay=args.weight_decay)
        self.D_solver = optim.Adam(filter(lambda p: p.requires_grad, D_model.parameters()), args.lr_d, [0.9, 0.99])
        # self.D_solver = optim.SGD([{'params': filter(lambda p: p.requires_grad, D_model.parameters()), 'initial_lr': args.lr_d}], args.lr_d, momentum=args.momentum, weight_decay=args.weight_decay)

        self.criterion_dsn = CriterionDSN().cuda()
        if args.kd:
            self.criterion_kd = CriterionKD().cuda()
        if args.adv:
            self.criterion_adv = CriterionAdv(args.adv_loss_type).cuda()
            if args.adv_loss_type == 'wgan-gp': self.criterion_AdditionalGP = CriterionAdditionalGP(D_model, args.lambda_gp).cuda()
            self.criterion_adv_for_G = CriterionAdvForG(args.adv_loss_type).cuda()
        if args.cwd:
            self.criterion_cwd = CriterionCWD(args.norm_type,args.divergence,args.temperature).cuda()
            if args.cwd_feat:
                self.linear = torch.nn.Sequential(
        	      torch.nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                InPlaceABNSync(512),
                torch.nn.ReLU(inplace=False)
          			).cuda()
                # self.linear = torch.nn.Conv2d(128,512, kernel_size=1, stride=1, padding=0, dilation=1, bias=True).cuda()
                # self.linear = torch.nn.Conv2d(128,512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False).cuda()
                self.G_solver.add_param_group({'params':self.linear.parameters(), 'initial_lr': args.lr_g})
        self.G_loss, self.D_loss = 0.0, 0.0
        self.mc_G_loss, self.kd_G_loss, self.adv_G_loss, self.cwd_G_loss = 0.0, 0.0, 0.0, 0.0

        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_input(self, data):
        images, labels, _, _ = data
        self.images = images.cuda()
        self.labels = labels.long().cuda()

    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr*((1-float(iter)/max_iter)**(power))
            
    def adjust_learning_rate(self, base_lr, optimizer, i_iter):
        args = self.args
        lr = self.lr_poly(base_lr, i_iter, args.num_steps, args.power)
        optimizer.param_groups[0]['lr'] = lr
        return lr

    def segmentation_forward(self):
        with torch.no_grad():
            self.preds_T = self.teacher.eval()(self.images)
        self.preds_S = self.student.train()(self.images)

    def segmentation_backward(self):
        args = self.args
        temp = self.criterion_dsn(self.preds_S, self.labels)
        self.mc_G_loss = temp.item()
        g_loss = temp
        if args.kd:
            temp = args.lambda_kd*self.criterion_kd(self.preds_S, self.preds_T)
            self.kd_G_loss = temp.item()
            g_loss = g_loss + temp
        if args.adv:
            temp = args.lambda_adv*self.criterion_adv_for_G(self.D_model(self.preds_S[0]))
            self.adv_G_loss = temp.item()
            g_loss = g_loss + temp
        if args.cwd:
            if args.cwd_feat:
                temp = args.lambda_cwd*self.criterion_cwd(self.linear(self.preds_S[2]), self.preds_T[2])
            else:
                temp = args.lambda_cwd*self.criterion_cwd(self.preds_S[0], self.preds_T[0])
            self.cwd_G_loss = temp.item()
            g_loss = g_loss + temp
        g_loss.backward()
        self.G_loss = g_loss.item()

    def discriminator_forward_backward(self):
        args = self.args
        d_loss = args.lambda_d*self.criterion_adv(self.D_model(self.preds_S[0].detach()), self.D_model(self.preds_T[0].detach()))
        if args.adv_loss_type == 'wgan-gp': d_loss += args.lambda_d*self.criterion_AdditionalGP(self.preds_S, self.preds_T)
        d_loss.backward()
        self.D_loss = d_loss.item()

    def optimize_parameters(self):
        self.segmentation_forward()
        self.G_solver.zero_grad()
        self.segmentation_backward()
        self.G_solver.step()
        if self.args.adv:
            self.D_solver.zero_grad()
            self.discriminator_forward_backward()
            self.D_solver.step()

    def print_info(self, step):
        logging.info('step:{:5d} G_lr:{:.6f} G_loss:{:.5f}(mc:{:.5f} kd:{:.5f} adv:{:.5f} cwd:{:5f}) D_lr:{:.6f} D_loss:{:.5f}'.format(
                        step, self.G_solver.param_groups[-1]['lr'], self.G_loss,
                        self.mc_G_loss, self.kd_G_loss, self.adv_G_loss, self.cwd_G_loss,
                        self.D_solver.param_groups[-1]['lr'], self.D_loss))

    def save_ckpt(self, step):
        args = self.args
        logging.info('saving ckpt: '+args.save_path+'/'+args.data_set+'_'+str(step)+'_G.pth')
        torch.save(self.student.state_dict(), args.save_path+'/'+args.data_set+'_'+str(step)+'_G.pth')
        if self.args.adv:
            logging.info('saving ckpt: '+args.save_path+'/'+args.data_set+'_'+str(step)+'_D.pth')
            torch.save(self.D_model.state_dict(), args.save_path+'/'+args.data_set+'_'+str(step)+'_D.pth')

    def __del__(self):
        pass
