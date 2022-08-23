import os
import gc
import torch
from torch.autograd import Variable
from datetime import datetime
from data.dataloader import get_loader, BrainDataset
from data.dataset import Dataset
import torch.utils.data as datas
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from utils.metrics import dice, m_iou
import torch.nn.functional as F
from Net.CM_MLP import CM_MLP
from utils.lookahead import Lookahead

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7, 8'
device = torch.device('cuda', 2)
best_train = 0.0


def structure_loss(pred, mask):
    """
    :param pred: Prediction
    :param mask: Ground Truth
    :return: Loss and IoU
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce=None, reduction='mean')
    pred = torch.sigmoid(pred)
    inter = (pred * mask)
    union = (pred + mask)
    wiou = 1 - ((inter * weit).sum(dim=(2, 3)) + 1) / (
                (union * weit).sum(dim=(2, 3)) - (inter * weit).sum(dim=(2, 3)) + 1)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    inter = (pred * mask)
    union = (pred + mask)
    iou = (inter.sum(dim=(2, 3)) + 1) / ((union).sum(dim=(2, 3)) - inter.sum(dim=(2, 3)) + 1)

    return (wbce+wiou).mean(), iou.mean()


def valid(model, val_data_loader, classes):
    model.eval()
    test_loader = val_data_loader
    b = 0.0
    c = 0.0
    print('[test_size]', len(test_loader))
    with torch.no_grad():
        for i, data in enumerate(test_loader, start=1):
            image, gt = data
            image = image.cuda()
            res4, res3, res2, res1 = model(image)
            res = res4
            res = F.upsample(res, size=(gt.shape[2], gt.shape[3]), mode='bilinear', align_corners=False)
            res = res.sigmoid()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            b += dice(res, gt)
            c += m_iou(res, gt)

    return b / len(test_loader), c / len(test_loader)


def train(data_loader, model, optimizer, epoch, t_size, clip, batch_size, t_epoch, val_dataloader, classes, param):
    model.train()
    global best_train
    b_loss_record, loss_record1, loss_record2, loss_record3, loss_record4, b_iou_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, data in enumerate(data_loader, start=1):
        optimizer.zero_grad()
        images, labels = data
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1 = model(images)

        loss4, iou4 = structure_loss(lateral_map_4, labels)
        loss3, iou3 = structure_loss(lateral_map_3, labels)
        loss2, iou2 = structure_loss(lateral_map_2, labels)
        loss1, iou1 = structure_loss(lateral_map_1, labels)

        loss = loss1 + loss2 + loss3 + loss4
        iou = (iou1 + iou2 + iou3 + iou4) / 4
        # ------ backward ------
        loss.backward()
        clip_gradient(optimizer, clip)
        optimizer.step()

        # ------ recording loss ------
        loss_record4.update(loss4.data, batch_size, dtype=0)
        loss_record3.update(loss3.data, batch_size, dtype=0)
        loss_record2.update(loss2.data, batch_size, dtype=0)
        loss_record1.update(loss1.data, batch_size, dtype=0)
        b_iou_record.update(iou.data, batch_size, dtype=1)
        b_loss_record.update(loss.data, batch_size, dtype=0)

        print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}],loss:{:0.4f}, iou:{:0.4f}'
              ' lateral-4: wiou4: {:0.4f}], lateral-3: wiou3: {:0.4f}], lateral-2: wiou2:{:0.4f}], lateral-1: wiou1: {:0.4}]'.
              format(datetime.now(), epoch, t_epoch, i, total_step, b_loss_record.show(dtype=0),
                     b_iou_record.show(dtype=1),
                     loss_record4.show(dtype=0), loss_record3.show(dtype=0), loss_record2.show(dtype=0),
                     loss_record1.show(dtype=0))
              )
    record.update(b_loss_record.show(dtype=0), dtype=0)
    record.update(b_iou_record.show(dtype=1), dtype=1)
    '''Reset all recoder'''
    b_iou_record.reset()
    loss_record4.reset()
    loss_record3.reset()
    loss_record2.reset()
    loss_record1.reset()
    b_loss_record.reset()

    save_path = param['model_save_path']
    model_path = param['model_path']
    os.makedirs(save_path, exist_ok=True)

    '''Validation'''
    meandice, mIOU = valid(model, val_dataloader, classes)
    print('Epoch [{:03d}/{:03d}], mean dice:{:0.4f}, mean IOU:{:0.4f}'.format(epoch, t_epoch, meandice, mIOU))
    record.update(meandice, dtype=2)

    '''Log'''
    log_dir = param['log_dir']
    best_log = param['best_log']
    if not os.path.isfile(best_log):
        f = open(best_log, 'a')
        f.write('0')
        f.close()
    fp = open(log_dir, 'a')
    fp.write(str(meandice) + '\n')
    fp.close()

    '''Saving snapshot when current dice larger than before'''
    if meandice > best_train:
        fp = open(best_log, 'w')
        fp.write(str(meandice))
        fp.close()
        fp = open(best_log, 'r')
        best_train = meandice
        fp.close()
        torch.save(model.state_dict(), model_path)
        print('[Saving best Snapshot:](%d / %d)' % (epoch, t_epoch), ' Mean dice:', meandice, '[best:]', best_train)


if __name__ == '__main__':
    torch.cuda.set_device(device)
    torch.backends.cudnn.enabled = False
    batch_size = 6
    fold = 5
    Epoch = 60
    clip = 0.5
    learning_rate = 1e-4
    decay_rate = 0.1
    decay_epoch = 15
    train_size = 512
    ill_param = {
        'data_path': '/data/illness',
        'fig_path': './result/illness_fig',
        'fig_path_arg': './result/illness_fig/',
        'log_base_path': './log/illness',
        'log_dir': './log/illness/log.txt',
        'best_log': './log/illness/best.txt',
        'model_save_path': './result/illness/',
        'model_path': './result/illness/caraNet.pth',
        'result_path': './result/illness/figure/',
        'debug_path': './log/debug/illness',
        'classes': 2
    }

    ill_region_param = {
        'data_path': '/data/illness_region',
        'fig_path': './result/illness_region_fig',
        'fig_path_arg': './result/illness_region_fig/',
        'log_base_path': './log/illness_region',
        'log_dir': './log/illness_region/log.txt',
        'best_log': './log/illness_region/best.txt',
        'model_save_path': './result/illness_region/',
        'model_path': './result/illness_region/caraNet.pth',
        'result_path': './result/illness_region/figure/',
        'debug_path': './log/debug/illness_region',
        'classes': 15
    }

    skull_param = {
        'data_path': '/data/skull_SH_new/',
        'fig_path': './result/compare/train/mynet/skull_fig',
        'fig_path_arg': './result/compare/train/mynet/skull_fig/',
        'log_base_path': './log/skull',
        'log_dir': './log/skull/log.txt',
        'best_log': './log/skull/best.txt',
        'model_save_path': './result/skull/',
        'model_path': './result/skull/caraNet.pth',
        'result_path': './result/skull/figure/',
        'debug_path': './log/debug/Skull',
        'classes': 1
    }

    skull_param_out = {
        'data_path': '/data/skull_SH_EP_new',
        'fig_path': './result/skull_SH_EP_fig',
        'fig_path_arg': './result/skull_SH_EP_fig/',
        'log_base_path': './log/skull_SE',
        'log_dir': './log/skull_SE/log_SE.txt',
        'best_log': './log/skull_SE/best_SE.txt',
        'model_save_path': './result/skull_SE/',
        'model_path': './result/skull_SE/caraNet_SE.pth',
        'result_path': './result/skull_SE/figure_out/',
        'debug_path': './log/debug/SH_EP',
        'classes': 2
    }

    kvasir_param = {
        'data_path': '/data/kvasir/',
        'fig_path': './result/compare/train/mynet/kvasir_fig',
        'fig_path_arg': './result/compare/train/mynet/kvasir_fig/',
        'log_base_path': './log/kvasir',
        'log_dir': './log/kvasir/log.txt',
        'best_log': './log/kvasir/best.txt',
        'model_save_path': './result/kvasir/',
        'model_path': './result/kvasir/caraNet.pth',
        'result_path': './result/kvasir/figure/',
        'debug_path': './log/debug/kvasir',
        'classes': 1
    }

    cvc_param = {
        'data_path': '/data/CVC-Clinic/',
        'fig_path': './result/compare/train/mynet/CVC-Clinic_fig/',
        'fig_path_arg': './result/compare/train/mynet/CVC-Clinic_fig/',
        'log_base_path': './log/CVC-Clinic',
        'log_dir': './log/CVC-Clinic/log.txt',
        'best_log': './log/CVC-Clinic/best.txt',
        'model_save_path': './result/CVC-Clinic/',
        'model_path': './result/CVC-Clinic/caraNet.pth',
        'result_path': './result/CVC-Clinic/figure/',
        'debug_path': './log/debug/CVC-Clinic',
        'classes': 1
    }

    polyp_param = {
        'data_path': '/data/train_set/TrainSet/',
        'fig_path': './result/compare/train/mynet/Polyp_fig/',
        'fig_path_arg': './result/compare/train/mynet/Polyp_fig/',
        'log_base_path': './log/Polyp',
        'log_dir': './log/Polyp/log.txt',
        'best_log': './log/Polyp/best.txt',
        'model_save_path': './result/Polyp/',
        'model_path': './result/Polyp/mynet.pth',
        'result_path': './result/Polyp/figure/',
        'debug_path': './log/debug/Polyp',
        'classes': 1
    }

    glas_param = {
        'data_path': '/data/GLAS/train/',
        'fig_path': './result/glas_fig',
        'fig_path_arg': './result/glas_fig/',
        'log_base_path': './log/glas',
        'log_dir': './log/glas/log.txt',
        'best_log': './log/glas/best.txt',
        'model_save_path': './result/glas/',
        'model_path': './result/glas/caraNet.pth',
        'result_path': './result/glas/figure/',
        'debug_path': './log/debug/glas',
        'classes': 1
    }

    bowl_param = {
        'data_path': '/data/2018_data_science_bowl/train/',
        'fig_path': './result/bowl_fig',
        'fig_path_arg': './result/bowl_fig/',
        'log_base_path': './log/bowl',
        'log_dir': './log/bowl/log.txt',
        'best_log': './log/bowl/best.txt',
        'model_save_path': './result/bowl/',
        'model_path': './result/bowl/caraNet.pth',
        'result_path': './result/bowl/figure/',
        'debug_path': './log/debug/bowl',
        'classes': 1
    }

    param = skull_param

    classes = param['classes']
    model = CM_MLP(classes=classes).cuda()
    fig_path = param['fig_path']
    fig_path_args = param['fig_path_arg']
    log_base_path = param['log_base_path']

    if not os.path.exists(log_base_path):
        os.makedirs(log_base_path, exist_ok=True)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path, exist_ok=True)

    optimizer = Lookahead(torch.optim.Adam(model.parameters(), learning_rate, eps=1e-5), la_steps=3, la_alpha=0.5)

    data_root = param['data_path']
    dataset = Dataset(data_root, True)
    t_train_list = dataset.get_all_list()
    test_loader = get_loader(data_root=data_root, batch_size=batch_size, dtype='test', size=train_size, augmentations=False)
    record = AvgMeter()

    for epoch in range(1, Epoch+1):
        length = len(t_train_list) // fold
        start = length * (epoch % fold)
        val_list = [t_train_list[x % len(t_train_list)] for x in range(start, start + length)]
        train_list = list(set(t_train_list).difference(set(val_list)))
        train_dataset = BrainDataset(train_list, 'train', train_size, dataset.get_path(), True)
        val_dataset = BrainDataset(val_list, 'val', train_size, dataset.get_path(), False)
        train_loader = datas.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        total_step = len(train_loader)
        val_loader = datas.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )

        adjust_lr(optimizer, learning_rate, epoch, decay_rate, decay_epoch)
        torch.cuda.empty_cache()
        gc.collect()
        train(train_loader, model, optimizer, epoch, train_size, clip, batch_size, Epoch, val_loader, classes, param)
        record.save(fig_path_args, 'args.jpg')
