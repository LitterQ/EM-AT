import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *
#from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import utils
import math

from utils import softCrossEntropy
from utils import one_hot_tensor, label_smoothing
import ot
import pickle
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


class Attack_None(nn.Module):
    def __init__(self, basic_net, config):
        super(Attack_None, self).__init__()
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.basic_net = basic_net
        print(config)

    def forward(self, inputs, targets, attack=None, batch_idx=-1):
        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()
        x = inputs.clone()
        for j in range(3):
            x.requires_grad_()
            logits_pred, fea = self.basic_net(x)
            self.basic_net.zero_grad()

            inp_grad = torch.autograd.grad([fea.sum()], [x])[0]
            x_energy = x - 2/255 * 2 * torch.sign(inp_grad) * 4
            #x_energy = torch.min(torch.max(x_energy, x - 8/255 * 2),
                              #x + 8/255*2)
            #x_energy = torch.clamp(x_energy, -1.0, 1.0)

        outputs = self.basic_net(x_energy.detach())[0]
        #outputs, _ = self.basic_net(inputs)
        return outputs, None


class Attack_PGD(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_PGD, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.loss_func = torch.nn.CrossEntropyLoss(
            reduction='none') if 'loss_func' not in config.keys(
            ) else config['loss_func']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']

        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net.eval()
        logits_pred_nat = aux_net(inputs)[0]
        targets_prob = F.softmax(logits_pred_nat.float(), dim=1)

        num_classes = targets_prob.size(1)

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_org = x.detach()
        loss_array = np.zeros((inputs.size(0), self.num_steps))

        for i in range(self.num_steps):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x)[0]
            loss = self.loss_func(logits, y_tensor_adv)
            loss = loss.mean()
            aux_net.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * self.step_size * torch.sign(
                x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        x_pgd = x.clone()
        x.detach()
        for j in range(3):
            x.requires_grad_()
            logits_pred, fea = self.basic_net(x)
            self.basic_net.zero_grad()

            inp_grad = torch.autograd.grad([fea.sum()], [x])[0]
            #x = x - self.step_size * torch.sign(inp_grad) * 2
            x = x - 2 / 255 * 2 * torch.sign(inp_grad) * 4
            #x = torch.min(torch.max(x, x_pgd - self.epsilon),
                              #x_pgd + self.epsilon)
            #x = torch.clamp(x, -1.0, 1.0)

        logits_pert = self.basic_net(x.detach())[0]

        return logits_pert, targets_prob.detach()


class Attack_FeaScatter(nn.Module):
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_FeaScatter, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']
        self.mse_loss = loss = nn.MSELoss()

        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs, _ = self.basic_net(inputs)
            return outputs, None
        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))

        aux_net.eval()
        batch_size = inputs.size(0)
        m = batch_size
        n = batch_size

        logits = aux_net(inputs)[0]
        num_classes = logits.size(1)

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()

        x_org = x.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pred_nat, fea_nat = aux_net(inputs)

        num_classes = logits_pred_nat.size(1)
        y_gt = one_hot_tensor(targets, num_classes, device)

        loss_ce = softCrossEntropy()

        iter_num = self.num_steps

        indices = np.random.permutation(x.size(0))
        mse = 0.0

        for i in range(iter_num):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)

            logits_pred, fea = aux_net.adapt_for(x)
            #print(logits_pred.shape, fea.shape)
            #ot_loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,
                                                  #logits_pred, None, None,
                                                  #0.01, m, n)

            #inp_grad = torch.autograd.grad([fea], [x])[0]
            #x = x + inp_grad
            ot_loss = self.loss_func(logits_pred, targets)
            ot_loss = ot_loss.mean()

            aux_net.zero_grad()
            adv_loss = ot_loss
            adv_loss.backward(retain_graph=True)
            x_adv = x.data + self.step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

            x_mse = x.clone()
            for j in range(1):
                x_mse.requires_grad_()
                logits_pred_energy, fea_energy = self.basic_net(x_mse)
                self.basic_net.zero_grad()

                inp_grad = torch.autograd.grad([fea_energy.sum()], [x_mse])[0]
                x_energy = x - self.step_size * 4 * torch.sign(inp_grad)
                x_energy = torch.min(torch.max(x_energy, x_mse - self.epsilon),
                              x_mse + self.epsilon)
                x_energy = torch.clamp(x_energy, -1.0, 1.0)

                mse = mse + self.mse_loss(x_energy, inputs)

        logits_pred_2, _ = self.basic_net(x_energy)
        logits_pred_3, _ = self.basic_net(x)
        logits_pred_natural, _ = self.basic_net(inputs)
        y_sm = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)

        adv_loss = loss_ce(logits_pred_2, y_sm.detach())
        nat_loss = loss_ce(logits_pred_natural, y_sm.detach())
        #strong_loss = loss_ce(logits_pred_3, y_sm.detach())
        total_loss = mse + adv_loss + nat_loss #+ 2 * strong_loss

        return logits_pred, total_loss
