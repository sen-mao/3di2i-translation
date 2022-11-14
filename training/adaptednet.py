# Copyright (c) Computer school, NKU(Nankai University).

import torch
import torch.nn as nn
import random

from torch_utils import persistence
from training.networks import Conv2dLayer
from training.utils import norm
from training.adaptedblock import AdaptedBlock

@persistence.persistent_class
class AdaptedNet(torch.nn.Module):
    # We adapt network contained in cyclegan (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master).
    def __init__(self, input_nc=512, output_nc=64, ngf=64, net='resnet_9blocks', norm='instance', no_dropout=True, init_type='normal', **unused):
        super().__init__()
        self.adaptedblock = AdaptedBlock(input_nc=input_nc, output_nc=output_nc, ngf=ngf, netG=net, use_dropout=not no_dropout, init_type=init_type)
        self.to_rgb = Conv2dLayer(in_channels=output_nc, out_channels=3, kernel_size=1, activation='linear')

    def forward(self, b64_x):
        fake_x_nerf = self.adaptedblock(b64_x)
        fake_img_nerf = self.to_rgb(fake_x_nerf)
        return norm(fake_x_nerf), norm(fake_img_nerf)

# ---------- CCPL ---------- #
vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
)

@persistence.persistent_class
class Normalize(torch.nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

@persistence.persistent_class
class CCPL(torch.nn.Module):
    def __init__(self, encoder, num_s, num_l, tau):
        super().__init__()
        self.num_s = num_s
        self.num_l = num_l
        self.tau   = tau
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.end_layer = 1
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.l1loss             = torch.nn.L1Loss()
        self.mlp = nn.ModuleList([nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128)])

    def NeighborSample(self, feat, layer, num_s, sample_ids=None):
        b, c, h, w = feat.size()
        if sample_ids is None:
            sample_ids = []
            while len(sample_ids) < num_s:
                h_id = random.randint(0, h-3) # upper left corner
                w_id = random.randint(0, w-3)
                if [h_id, w_id] not in sample_ids:
                    sample_ids += [[h_id, w_id]]
            sample_ids = torch.tensor(sample_ids)
        h_ids = sample_ids[:,0]
        w_ids = sample_ids[:,1]
        ft = torch.ones((b,c,8*num_s)).to(feat.device) # b, c, 32
        for i in range(num_s):
            f_c = feat[:,:,h_ids[i]+1,w_ids[i]+1].view(b,c,1) # centor
            f = feat[:,:,h_ids[i]:h_ids[i]+3,w_ids[i]:w_ids[i]+3].flatten(2, 3) - f_c
            ft[:,:,8*i:8*i+8] = torch.cat([f[:,:,:4], f[:,:,5:]], 2)
        ft = ft.permute(0,2,1) # b, (8*num_s), c
        # for i in range(3):
        #     ft = self.mlp[3*layer+i](ft)
        ft = Normalize(2)(ft.permute(0,2,1))
        return ft, sample_ids

    ## PatchNCELoss code from: https://github.com/taesungp/contrastive-unpaired-translation
    def PatchNCELoss(self, f_q, f_k, tau=0.07):
        # batch size, channel size, and number of sample locations
        B, C, S = f_q.shape
        # calculate v * v+: BxSx1
        l_pos = (f_k * f_q).sum(dim=1)[:, :, None]
        # calculate v * v-: BxSxS
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k)
        # The diagonal entries are not negatives. Remove them.
        identity_matrix = torch.eye(S,dtype=torch.bool)[None, :, :].to(f_q.device)
        l_neg.masked_fill_(identity_matrix, -float('inf'))
        # calculate logits: (B)x(S)x(S+1)
        logits = torch.cat((l_pos, l_neg), dim=2) / tau
        # return PatchNCE loss
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * S, dtype=torch.long).to(f_q.device)
        return self.cross_entropy_loss(predictions, targets)

    def forward_ccpl(self, feats_q, feats_k, num_s, start_layer, end_layer, tau=0.07):
        loss_ccp = 0.0
        if not isinstance(feats_q, list): feats_q = [feats_q]
        if not isinstance(feats_k, list): feats_k = [feats_k]
        for i in range(start_layer, end_layer):
            f_q, sample_ids = self.NeighborSample(feats_q[i], i, num_s)
            f_k, _ = self.NeighborSample(feats_k[i], i, num_s, sample_ids)
            # loss_ccp += self.PatchNCELoss(f_q, f_k, tau)
            loss_ccp += self.l1loss(f_q, f_k)
        return loss_ccp

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(self.end_layer):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def forward(self, content, gimage, is_feature=False):
        if is_feature:
            content_feats, g_t_feats = content, gimage
        else:
            content_feats = self.encode_with_intermediate(content)
            g_t_feats = self.encode_with_intermediate(gimage)

        end_layer = self.end_layer

        start_layer = end_layer - self.num_l
        loss_ccp = self.forward_ccpl(g_t_feats, content_feats, self.num_s, start_layer, end_layer, self.tau)

        return loss_ccp
