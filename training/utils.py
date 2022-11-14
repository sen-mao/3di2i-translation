import numpy as np
import torch
import math
import re
import os
import time

def proc_img(img):
    return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

def upper_dir(dir):
    return '/'.join(dir.split('/')[:-1])

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 15, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 8, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels, _ = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

def image_grid(img, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)
    return img

def norm(x):
        return x / (1e-7 + x.norm(dim=-1, keepdim=True))


from apps.inversion import VGG16_for_Perceptual
from training.data_utils import save_image_grid

def caluclate_loss(synth_img, img, perceptual_net, img_p, MSE_Loss, upsample2d):
    # calculate MSE Loss
    mse_loss = MSE_Loss(synth_img, img)  # (lamda_mse/N)*||G(w)-I||^2

    # calculate Perceptual Loss
    real_0, real_1, real_2, real_3 = perceptual_net(img_p)
    synth_p = upsample2d(synth_img)  # (1, 3, 256, 256)
    synth_0, synth_1, synth_2, synth_3 = perceptual_net(synth_p)

    perceptual_loss = 0
    perceptual_loss += MSE_Loss(synth_0, real_0)
    perceptual_loss += MSE_Loss(synth_1, real_1)
    perceptual_loss += MSE_Loss(synth_2, real_2)
    perceptual_loss += MSE_Loss(synth_3, real_3)

    return mse_loss, perceptual_loss

def encode_image(img_num, imgs, img_res, num_steps, G, D, Adapted_net, device):
    """
    https://github.com/pacifinapacific/StyleGAN_LatentEditor/blob/master/encode_image.py
    """
    print(f'| ---------- predict w ---------- |')
    grid_size = (int(math.sqrt(img_num)), int(math.sqrt(img_num)))
    b64_x = D(imgs, step=2)
    fake_x_nerf = Adapted_net(b64_x.detach().to(dtype=torch.float32)).detach()
    assert fake_x_nerf is not None

    ws_nerf = torch.zeros([img_num, 10, G.z_dim], requires_grad=True, device=device)
    ws = torch.zeros([img_num, 7, G.z_dim], requires_grad=True, device=device)  # rand better than randn initialization
    styles = [ws_nerf, ws]
    assert styles != [None, None]
    # z = torch.randn([img_num, G.z_dim], requires_grad=False, device=device)
    # ws_z = G.mapping(z)[1]
    # ws = torch.zeros([img_num, 9, G.z_dim], requires_grad=True, device=device)
    # ws.data = ws_z.data

    imgs_p = imgs.clone()  # for perceptual loss
    upsample2d = torch.nn.Upsample(scale_factor=256 / img_res, mode='bilinear')
    imgs_p = upsample2d(imgs_p)

    MSE_Loss = torch.nn.L1Loss(reduction="mean")
    perceptual_net = VGG16_for_Perceptual(n_layers=[2, 4, 14, 21]).to(device)
    optimizer = torch.optim.Adam({ws_nerf, ws}, lr=0.01, betas=(0.9,0.999), eps=1e-8)
    start_time = time.time()
    for i in range(num_steps):
        optimizer.zero_grad()

        synth_imgs = G.get_final_output_adapted(styles=styles, fake_x_nerf=fake_x_nerf)
        if i % 100 == 0:
            save_image_grid(synth_imgs.cpu().detach().numpy(), os.path.join('./results/', f'synth_imgs_{i}.png'), drange=[-1, 1], grid_size=grid_size)
        # synth_imgs = (synth_imgs + 1.0) / 2.0
        mse_loss, perceptual_loss = caluclate_loss(synth_imgs, imgs, perceptual_net, imgs_p, MSE_Loss, upsample2d)
        loss = 0.1*mse_loss + perceptual_loss
        loss.backward()

        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        loss_p = perceptual_loss.detach().cpu().numpy()
        loss_m = mse_loss.detach().cpu().numpy()
        if i % 50 == 0:
            print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}".format(i, loss_np, loss_m, loss_p))
    print(f'| ---------- iteration time: {(time.time() - start_time) / 60 :.3f} minutes ---------- |')
    return ws_nerf, ws