import torch
from matplotlib import pyplot as plt
import os


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    print("Learning Rate Change: init:{:0.4f}, now:{:0.4f}".format(init_lr, decay*init_lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = float(init_lr * decay)


class AvgMeter(object):
    def __init__(self, num=0):
        self.iou = 0
        self.val = []
        self.losses = []
        self.count = []
        self.num = num
        self.reset()

    def reset(self):
        self.count = 0
        self.losses = []
        self.val = []
        self.iou = []

    def update(self, val, n=1, dtype=0):
        self.count += n
        if dtype == 0:
            self.losses.append(val)
        elif dtype == 1:
            self.iou.append(val)
        else:
            self.val.append(val)

    def show(self, dtype=0):
        if dtype == 0:
            return torch.mean(torch.stack(self.losses))
        elif dtype == 1:
            return torch.mean(torch.stack(self.iou))
        return torch.mean(torch.stack(self.val))

    def show_test_result(self, dtype=0):
        if dtype == 0:
            return float(sum(self.iou)/len(self.iou))
        if dtype == 1:
            return float(sum(self.val)/len(self.val))

    def save(self, path, name):
        path = os.path.join(path, name)
        x = [i for i in range(1,len(self.losses)+1)]
        plt.figure()
        plt.subplot(121)
        plt.plot(x,self.losses, 'r-',label='loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        plt.subplot(122)
        plt.plot(x,self.iou, 'b-', label='iou')
        plt.plot(x,self.val, 'g-', label='val-dice')
        plt.xlabel('Epoch')
        plt.ylabel('value')
        plt.legend()
        plt.grid()
        
        plt.savefig(path)
        plt.close()
