import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from model.ResNet import B2_ResNet
from utils import init_weights,init_weights_orthogonal_normal
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable
from torch.nn import Parameter, Softmax
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from model.HolisticAttention import HA
import math
CE = torch.nn.BCELoss(reduction='sum')
cos_sim = torch.nn.CosineSimilarity(dim=1,eps=1e-8)
from model.batchrenorm import BatchRenorm2d

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Mutual_info_reg, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        self.channel = channels

        self.fc1_rgb1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        self.fc2_rgb1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        self.fc1_depth1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        self.fc2_depth1 = nn.Linear(channels * 1 * 16 * 16, latent_size)

        self.fc1_rgb2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        self.fc2_rgb2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        self.fc1_depth2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        self.fc2_depth2 = nn.Linear(channels * 1 * 22 * 22, latent_size)

        self.fc1_rgb3 = nn.Linear(channels * 1 * 28 * 28, latent_size)
        self.fc2_rgb3 = nn.Linear(channels * 1 * 28 * 28, latent_size)
        self.fc1_depth3 = nn.Linear(channels * 1 * 28 * 28, latent_size)
        self.fc2_depth3 = nn.Linear(channels * 1 * 28 * 28, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, depth_feat):
        rgb_feat = self.layer3(self.leakyrelu(self.bn1(self.layer1(rgb_feat))))
        depth_feat = self.layer4(self.leakyrelu(self.bn2(self.layer2(depth_feat))))
        # print(rgb_feat.size())
        # print(depth_feat.size())
        if rgb_feat.shape[2] == 16:
            rgb_feat = rgb_feat.view(-1, self.channel * 1 * 16 * 16)
            depth_feat = depth_feat.view(-1, self.channel * 1 * 16 * 16)

            mu_rgb = self.fc1_rgb1(rgb_feat)
            logvar_rgb = self.fc2_rgb1(rgb_feat)
            mu_depth = self.fc1_depth1(depth_feat)
            logvar_depth = self.fc2_depth1(depth_feat)
        elif rgb_feat.shape[2] == 22:
            rgb_feat = rgb_feat.view(-1, self.channel * 1 * 22 * 22)
            depth_feat = depth_feat.view(-1, self.channel * 1 * 22 * 22)
            mu_rgb = self.fc1_rgb2(rgb_feat)
            logvar_rgb = self.fc2_rgb2(rgb_feat)
            mu_depth = self.fc1_depth2(depth_feat)
            logvar_depth = self.fc2_depth2(depth_feat)
        else:
            rgb_feat = rgb_feat.view(-1, self.channel * 1 * 28 * 28)
            depth_feat = depth_feat.view(-1, self.channel * 1 * 28 * 28)
            mu_rgb = self.fc1_rgb3(rgb_feat)
            logvar_rgb = self.fc2_rgb3(rgb_feat)
            mu_depth = self.fc1_depth3(depth_feat)
            logvar_depth = self.fc2_depth3(depth_feat)

        mu_depth = self.tanh(mu_depth)
        mu_rgb = self.tanh(mu_rgb)
        logvar_depth = self.tanh(logvar_depth)
        logvar_rgb = self.tanh(logvar_rgb)
        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        z_depth = self.reparametrize(mu_depth, logvar_depth)
        dist_depth = Independent(Normal(loc=mu_depth, scale=torch.exp(logvar_depth)), 1)
        bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
            self.kl_divergence(dist_depth, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_depth)
        ce_rgb_depth = CE(z_rgb_norm,z_depth_norm.detach())
        ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
        latent_loss = ce_rgb_depth+ce_depth_rgb-bi_di_kld
        # latent_loss = torch.abs(cos_sim(z_rgb,z_depth)).sum()

        return latent_loss, z_rgb, z_depth



class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)



class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res



class Saliency_feat_decoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel,latent_dim):
        super(Saliency_feat_decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 2048)
        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel)
        self.layer7 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 256)
        self.conv2 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 512)
        self.conv3 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 1024)
        self.conv4 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 2048)

        self.spatial_axes = [2, 3]

        self.racb_43 = RCAB(channel * 2)
        self.racb_432 = RCAB(channel * 3)
        self.racb_4321 = RCAB(channel * 4)

        self.conv43 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 2*channel)
        self.conv432 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 3*channel)
        self.conv4321 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 4*channel)

        self.layer_depth = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 3, channel * 4)

        self.rcab_z1 = RCAB(channel + latent_dim)
        self.conv_z1 = BasicConv2d(channel+latent_dim,channel,3,padding=1)

        self.rcab_z2 = RCAB(channel + latent_dim)
        self.conv_z2 = BasicConv2d(channel + latent_dim, channel, 3, padding=1)

        self.rcab_z3 = RCAB(channel + latent_dim)
        self.conv_z3 = BasicConv2d(channel + latent_dim, channel, 3, padding=1)

        self.rcab_z4 = RCAB(channel + latent_dim)
        self.conv_z4 = BasicConv2d(channel + latent_dim, channel, 3, padding=1)

        self.br1 = BatchRenorm2d(channel)
        self.br2 = BatchRenorm2d(channel)
        self.br3 = BatchRenorm2d(channel)
        self.br4 = BatchRenorm2d(channel)


    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, x1,x2,x3,x4,z1=None,z2=None,z3=None,z4=None):
        conv1_feat = self.br1(self.conv1(x1))
        conv2_feat = self.br2(self.conv2(x2))
        conv3_feat = self.br3(self.conv3(x3))
        conv4_feat = self.br4(self.conv4(x4))

        if z1!=None:
            z1 = torch.unsqueeze(z1, 2)
            z1 = self.tile(z1, 2, conv1_feat.shape[self.spatial_axes[0]])
            z1 = torch.unsqueeze(z1, 3)
            z1 = self.tile(z1, 3, conv1_feat.shape[self.spatial_axes[1]])

            z2 = torch.unsqueeze(z2, 2)
            z2 = self.tile(z2, 2, conv2_feat.shape[self.spatial_axes[0]])
            z2 = torch.unsqueeze(z2, 3)
            z2 = self.tile(z2, 3, conv2_feat.shape[self.spatial_axes[1]])

            z3 = torch.unsqueeze(z3, 2)
            z3 = self.tile(z3, 2, conv3_feat.shape[self.spatial_axes[0]])
            z3 = torch.unsqueeze(z3, 3)
            z3 = self.tile(z3, 3, conv3_feat.shape[self.spatial_axes[1]])

            z4 = torch.unsqueeze(z4, 2)
            z4 = self.tile(z4, 2, conv4_feat.shape[self.spatial_axes[0]])
            z4 = torch.unsqueeze(z4, 3)
            z4 = self.tile(z4, 3, conv4_feat.shape[self.spatial_axes[1]])

            conv1_feat = torch.cat((conv1_feat,z1),1)
            conv1_feat = self.rcab_z1(conv1_feat)
            conv1_feat = self.conv_z1(conv1_feat)

            conv2_feat = torch.cat((conv2_feat, z2), 1)
            conv2_feat = self.rcab_z2(conv2_feat)
            conv2_feat = self.conv_z2(conv2_feat)

            conv3_feat = torch.cat((conv3_feat, z3), 1)
            conv3_feat = self.rcab_z3(conv3_feat)
            conv3_feat = self.conv_z3(conv3_feat)

            conv4_feat = torch.cat((conv4_feat, z4), 1)
            conv4_feat = self.rcab_z4(conv4_feat)
            conv4_feat = self.conv_z4(conv4_feat)

        conv4_feat = self.upsample2(conv4_feat)

        conv43 = torch.cat((conv4_feat, conv3_feat), 1)
        conv43 = self.racb_43(conv43)
        conv43 = self.conv43(conv43)

        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432(conv432)
        conv432 = self.conv432(conv432)

        conv432 = self.upsample2(conv432)
        conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 1)
        conv4321 = self.racb_4321(conv4321)
        conv4321 = self.conv4321(conv4321)

        sal_init = self.layer6(conv4321)

        return sal_init


class Saliency_feat_endecoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel):
        super(Saliency_feat_endecoder, self).__init__()
        self.resnet_rgb = B2_ResNet()
        self.resnet_depth = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.latent_dim = 6
        self.conv_depth1 = BasicConv2d(6, 3, kernel_size=3, padding=1)
        self.sal_decoder1 = Saliency_feat_decoder(channel, self.latent_dim)
        self.sal_decoder2 = Saliency_feat_decoder(channel, self.latent_dim)
        self.sal_decoder3 = Saliency_feat_decoder(channel, self.latent_dim)
        self.sal_decoder4 = Saliency_feat_decoder(channel, self.latent_dim)
        self.sal_decoder5 = Saliency_feat_decoder(channel, self.latent_dim)
        self.sal_decoder6 = Saliency_feat_decoder(channel, self.latent_dim)

        self.HA = HA()
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample025 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.upsample0125 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)


        self.convx1_depth = nn.Conv2d(in_channels=256, out_channels=channel, kernel_size=3, padding=1)
        self.convx2_depth = nn.Conv2d(in_channels=512, out_channels=channel, kernel_size=3, padding=1)
        self.convx3_depth = nn.Conv2d(in_channels=1024, out_channels=channel, kernel_size=3, padding=1)
        self.convx4_depth = nn.Conv2d(in_channels=2048, out_channels=channel, kernel_size=3, padding=1)

        self.convx1_rgb = nn.Conv2d(in_channels=256, out_channels=channel, kernel_size=3, padding=1)
        self.convx2_rgb = nn.Conv2d(in_channels=512, out_channels=channel, kernel_size=3, padding=1)
        self.convx3_rgb = nn.Conv2d(in_channels=1024, out_channels=channel, kernel_size=3, padding=1)
        self.convx4_rgb = nn.Conv2d(in_channels=2048, out_channels=channel, kernel_size=3, padding=1)

        self.mi_level1 = Mutual_info_reg(channel,channel,self.latent_dim)
        self.mi_level2 = Mutual_info_reg(channel, channel, self.latent_dim)
        self.mi_level3 = Mutual_info_reg(channel, channel, self.latent_dim)
        self.mi_level4 = Mutual_info_reg(channel, channel, self.latent_dim)

        self.spatial_axes = [2, 3]
        self.final_clc = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1)
        self.rcab_rgb_feat = RCAB(channel*4)
        self.rcab_depth_feat = RCAB(channel*4)




        if self.training:
            self.initialize_weights()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, x,depth=None):
        raw_x = x
        x = self.resnet_rgb.conv1(x)
        x = self.resnet_rgb.bn1(x)
        x = self.resnet_rgb.relu(x)
        x = self.resnet_rgb.maxpool(x)
        x1_rgb = self.resnet_rgb.layer1(x)  # 256 x 64 x 64
        x2_rgb = self.resnet_rgb.layer2(x1_rgb)  # 512 x 32 x 32
        x3_rgb = self.resnet_rgb.layer3_1(x2_rgb)  # 1024 x 16 x 16
        x4_rgb = self.resnet_rgb.layer4_1(x3_rgb)  # 2048 x 8 x 8

        sal_init_rgb = self.sal_decoder1(x1_rgb, x2_rgb, x3_rgb, x4_rgb)
        x2_2_rgb = self.HA(self.upsample05(sal_init_rgb).sigmoid(), x2_rgb)
        x3_2_rgb = self.resnet_rgb.layer3_2(x2_2_rgb)  # 1024 x 16 x 16
        x4_2_rgb = self.resnet_rgb.layer4_2(x3_2_rgb)  # 2048 x 8 x 8
        sal_ref_rgb = self.sal_decoder2(x1_rgb, x2_2_rgb, x3_2_rgb, x4_2_rgb)

        if depth==None:
            return self.upsample4(sal_init_rgb), self.upsample4(sal_ref_rgb)
        else:
            x = depth
            x = self.resnet_depth.conv1(x)
            x = self.resnet_depth.bn1(x)
            x = self.resnet_depth.relu(x)
            x = self.resnet_depth.maxpool(x)
            x1_depth = self.resnet_depth.layer1(x)  # 256 x 64 x 64
            x2_depth = self.resnet_depth.layer2(x1_depth) # 512 x 32 x 32
            x3_depth = self.resnet_depth.layer3_1(x2_depth)  # 1024 x 16 x 16
            x4_depth = self.resnet_depth.layer4_1(x3_depth)  # 2048 x 8 x 8

            sal_init_depth = self.sal_decoder3(x1_depth, x2_depth, x3_depth, x4_depth)
            x2_2_depth = self.HA(self.upsample05(sal_init_depth).sigmoid(), x2_depth)
            x3_2_depth = self.resnet_depth.layer3_2(x2_2_depth)  # 1024 x 16 x 16
            x4_2_depth = self.resnet_depth.layer4_2(x3_2_depth)  # 2048 x 8 x 8
            sal_ref_depth = self.sal_decoder4(x1_depth, x2_2_depth, x3_2_depth, x4_2_depth)


            lat_loss1, z1_rgb, z1_depth = self.mi_level1(self.convx1_rgb(x1_rgb), self.convx1_depth(x1_depth))
            lat_loss2, z2_rgb, z2_depth = self.mi_level2(self.upsample2(self.convx2_rgb(x2_2_rgb)), self.upsample2(self.convx2_depth(x2_2_depth)))
            lat_loss3, z3_rgb, z3_depth = self.mi_level3(self.upsample4(self.convx3_rgb(x3_2_rgb)), self.upsample4(self.convx3_depth(x3_2_depth)))
            lat_loss4, z4_rgb, z4_depth = self.mi_level4(self.upsample8(self.convx4_rgb(x4_2_rgb)), self.upsample8(self.convx4_depth(x4_2_depth)))

            lat_loss = lat_loss1+lat_loss2+lat_loss3+lat_loss4

            sal_mi_rgb = self.sal_decoder5(x1_rgb, x2_2_rgb, x3_2_rgb, x4_2_rgb, z1_depth,z2_depth,z3_depth,z4_depth)

            sal_mi_depth = self.sal_decoder6(x1_depth, x2_2_depth, x3_2_depth, x4_2_depth, z1_rgb,z2_rgb,z3_rgb,z4_rgb)

            final_sal = torch.cat((sal_ref_rgb,sal_ref_depth,sal_mi_rgb,sal_mi_depth),1)
            final_sal = self.final_clc(final_sal)

            return self.upsample4(sal_init_rgb), self.upsample4(sal_ref_rgb), self.upsample4(sal_init_depth), self.upsample4(
                sal_ref_depth), self.upsample4(sal_mi_rgb), self.upsample4(sal_mi_depth), self.upsample4(final_sal), lat_loss



    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet_rgb.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_rgb.state_dict().keys())
        self.resnet_rgb.load_state_dict(all_params)
        self.resnet_depth.load_state_dict(all_params)
