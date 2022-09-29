import torch
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import datetime
from model.ResNet_models_combine import Saliency_feat_endecoder
from data import get_loader
from utils import adjust_lr, AvgMeter
from scipy import misc
from utils import l2_regularisation


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=12, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
parser.add_argument('-beta1_gen', type=float, default=0.5,help='beta of Adam for generator')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--feat_channel', type=int, default=64, help='reduced channel of saliency feat')

opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
generator = Saliency_feat_endecoder(channel=opt.feat_channel)
generator.cuda()

generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)

## load data
image_root = './RGB/'
gt_root = './GT/'
depth_root = './depth/'

train_loader = get_loader(image_root, gt_root, depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

## define loss

CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [0.75,1,1.25]  # multi-scale training

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))


    pred  = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou  = 1-(inter+1)/(union-inter+1)

    return (wbce+wiou).mean()

def visualize_mi_rgb(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_rgb_mi.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_mi_depth(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_depth_mi.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

## visualize predictions and gt
def visualize_rgb_init(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_rgb_int.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_depth_init(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_depth_int.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_rgb_ref(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_rgb_ref.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_depth_ref(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_depth_ref.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_final_rgbd(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_rgbd.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_uncertainty_prior_init(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_prior_int.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

## linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed

def visualize_all_pred(pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8):
    for kk in range(pred1.shape[0]):
        pred1_kk, pred2_kk, pred3_kk, pred4_kk, pred5_kk, pred6_kk, pred7_kk, pred8_kk = pred1[kk, :, :, :], pred2[kk, :, :, :], pred3[kk, :, :, :], pred4[kk, :, :, :], pred5[kk, :, :, :], pred6[kk, :, :, :], pred7[kk, :, :, :], pred8[kk, :, :, :]
        pred1_kk = (pred1_kk.detach().cpu().numpy().squeeze()*255.0).astype(np.uint8)
        pred2_kk = (pred2_kk.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred3_kk = (pred3_kk.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred4_kk = (pred4_kk.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred5_kk = (pred5_kk.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred6_kk = (pred6_kk.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred7_kk = (pred7_kk.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)
        pred8_kk = (pred8_kk.detach().cpu().numpy().squeeze() * 255.0).astype(np.uint8)

        cat_img = cv2.hconcat([pred1_kk, pred2_kk, pred3_kk, pred4_kk, pred5_kk, pred6_kk, pred7_kk, pred8_kk])
        save_path = './temp/'
        name = '{:02d}_gt_initR_refR_intD_refD_mR_mD_Fused.png'.format(kk)
        cv2.imwrite(save_path + name, cat_img)

print("Let's Play!")
for epoch in range(1, opt.epoch+1):
    generator.train()
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            images, gts, depths = pack
            # print(index_batch)
            images = Variable(images)
            gts = Variable(gts)
            depths = Variable(depths)
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()

            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear',
                                          align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                depths = F.upsample(depths, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            init_rgb, ref_rgb, init_depth, ref_depth, mi_rgb, mi_depth, fuse_sal, latent_loss = generator.forward(images,depths)

            sal_rgb_loss = structure_loss(init_rgb, gts) + structure_loss(ref_rgb, gts) + structure_loss(mi_rgb, gts)
            sal_depth_loss = structure_loss(init_depth, gts) + structure_loss(ref_depth, gts) + structure_loss(mi_depth, gts)
            sal_final_rgbd = structure_loss(fuse_sal, gts)


            anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
            latent_loss = 0.1*anneal_reg*latent_loss
            sal_loss = sal_rgb_loss+sal_depth_loss+sal_final_rgbd + latent_loss
            sal_loss.backward()
            generator_optimizer.step()
            visualize_all_pred(gts,torch.sigmoid(init_rgb),torch.sigmoid(ref_rgb),torch.sigmoid(init_depth),torch.sigmoid(ref_depth),torch.sigmoid(mi_rgb),torch.sigmoid(mi_depth),torch.sigmoid(fuse_sal))

            if rate == 1:
                loss_record.update(sal_loss.data, opt.batchsize)


        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))
            # print(anneal_reg)


    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

    save_path = 'models/'


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % opt.epoch == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
