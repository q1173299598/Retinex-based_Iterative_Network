import os
import sys
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDGAN import CIDNet
from net.CIDGAN import Discriminator  # 假设Discriminator是你的判别器实现
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm

opt = option().parse_args()

class BrightnessPreservationLoss(nn.Module):
    def __init__(self):
        super(BrightnessPreservationLoss, self).__init__()

    def forward(self, original_image, generated_image):
        mean_brightness_original = torch.mean(original_image)
        mean_brightness_generated = torch.mean(generated_image)
        brightness_loss = torch.abs(mean_brightness_generated - mean_brightness_original)
        return brightness_loss

class ContrastPreservationLoss(nn.Module):
    def __init__(self):
        super(ContrastPreservationLoss, self).__init__()
        kernel = np.array([[1, -1]])
        self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda()

    def forward(self, original_image, generated_image):
        # 针对每个通道分别计算
        contrast_loss = 0
        for i in range(original_image.shape[1]):  # 遍历通道
            original_channel = original_image[:, i:i+1, :, :]
            generated_channel = generated_image[:, i:i+1, :, :]

            original_dx = F.conv2d(original_channel, self.kernel, padding=1)
            generated_dx = F.conv2d(generated_channel, self.kernel, padding=1)

            original_contrast = torch.std(original_dx)
            generated_contrast = torch.std(generated_dx)

            contrast_loss += torch.abs(generated_contrast - original_contrast)

        return contrast_loss

    
def seed_torch():
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_init():
    seed_torch()
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

def train(epoch):
    model.train()
    D_brightness.train()
    D_contrast.train()
    D_detail.train()
    loss_print = 0
    pic_cnt = 0
    loss_last_10 = 0
    pic_last_10 = 0
    train_len = len(training_data_loader)
    iter = 0
    torch.autograd.set_detect_anomaly(True)
    for batch in tqdm(training_data_loader):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()

        # CIDNet 前向传播
        output_rgb = model(im1)
        gt_rgb = im2
        output_hvi = model.HVIT(output_rgb)
        gt_hvi = model.HVIT(gt_rgb)

        # CIDNet 损失计算
        loss_hvi = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        loss_cidnet = loss_rgb + opt.HVI_weight * loss_hvi
        
        
        # 判别器损失
        # 判别器损失计算
        # 生成真实标签（全1）和伪造标签（全0）
        real_output_brightness = D_brightness(gt_rgb)
        fake_output_brightness = D_brightness(output_rgb.detach())
        real_labels = torch.ones_like(real_output_brightness).cuda()
        fake_labels = torch.zeros_like(fake_output_brightness).cuda()

        real_output_brightness = D_brightness(gt_rgb)
        fake_output_brightness = D_brightness(output_rgb.detach())
        real_loss_brightness = criterion_D(real_output_brightness, real_labels)
        fake_loss_brightness = criterion_D(fake_output_brightness, fake_labels)
        loss_D_brightness = (real_loss_brightness + fake_loss_brightness) * 0.5

        real_output_contrast = D_contrast(gt_rgb)
        fake_output_contrast = D_contrast(output_rgb.detach())
        real_loss_contrast = criterion_D(real_output_contrast, real_labels)
        fake_loss_contrast = criterion_D(fake_output_contrast, fake_labels)
        loss_D_contrast = (real_loss_contrast + fake_loss_contrast) * 0.5

        real_output_detail = D_detail(gt_rgb)
        fake_output_detail = D_detail(output_rgb.detach())
        real_loss_detail = criterion_D(real_output_detail, real_labels)
        fake_loss_detail = criterion_D(fake_output_detail, fake_labels)
        loss_D_detail = (real_loss_detail + fake_loss_detail) * 0.5

        # 总判别器损失
        ########################################################################################33
        #total_D_loss = loss_D_brightness + loss_D_contrast + loss_D_detail
        total_D_loss = loss_D_brightness

        # 判别器反向传播和优化
        optimizer_D_brightness.zero_grad()
        optimizer_D_contrast.zero_grad()
        optimizer_D_detail.zero_grad()
        total_D_loss.backward()
        optimizer_D_brightness.step()
        optimizer_D_contrast.step()
        optimizer_D_detail.step()

        # 生成器损失
        optimizer.zero_grad()
        output_rgb = model(im1)
        gt_rgb = im2
        loss_brightness = L1_loss(output_rgb, gt_rgb) + criterion_D(D_brightness(output_rgb), torch.ones_like(real_output_brightness))
        loss_contrast = D_loss(output_rgb, gt_rgb) + criterion_D(D_contrast(output_rgb), torch.ones_like(real_output_contrast))
        loss_detail = E_loss(output_rgb, gt_rgb) + criterion_D(D_detail(output_rgb), torch.ones_like(real_output_detail))
        G_loss = loss_brightness + loss_contrast + loss_detail
        total_G_loss = loss_cidnet + G_loss

        # 生成器反向传播和优化
        optimizer.zero_grad()##
        total_G_loss.backward()
        optimizer.step()

        # 记录损失
        loss = loss_cidnet + G_loss + total_D_loss
        loss_print = loss_print + loss.item()
        loss_last_10 = loss_last_10 + loss.item()
        pic_cnt += 1
        pic_last_10 += 1
        iter += 1
        if iter == train_len:
            print("===> Epoch[{}]: Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                loss_last_10/pic_last_10, optimizer.param_groups[0]['lr']))
            loss_last_10 = 0
            pic_last_10 = 0
            output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0))
            gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):
                os.mkdir(opt.val_folder+'training')
            output_img.save(opt.val_folder+'training/test.png')
            gt_img.save(opt.val_folder+'training/gt.png')
    return loss_print, pic_cnt

def checkpoint(epoch):
    os.makedirs(opt.weights_folder, exist_ok=True)
    model_out_path = os.path.join(opt.weights_folder, "epoch_{}.pth".format(epoch))
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path

def load_datasets():
    print('===> Loading datasets')
    if opt.lol_v1 or opt.lol_blur or opt.lolv2_real or opt.lolv2_syn or opt.SID or opt.SICE_mix or opt.SICE_grad:
        if opt.lol_v1:
            train_set = get_lol_training_set(opt.data_train_lol_v1, size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
            test_set = get_eval_set(opt.data_val_lol_v1)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        if opt.lol_blur:
            train_set = get_training_set_blur(opt.data_train_lol_blur, size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
            test_set = get_eval_set(opt.data_val_lol_blur)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        if opt.lolv2_real:
            train_set = get_lol_v2_training_set(opt.data_train_lolv2_real, size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
            test_set = get_eval_set(opt.data_val_lolv2_real)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        if opt.lolv2_syn:
            train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn, size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
            test_set = get_eval_set(opt.data_val_lolv2_syn)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

        if opt.SID:
            train_set = get_SID_training_set(opt.data_train_SID, size=opt.cropSize)
            training_data_loader = DataLoader(datasets(train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True))
            test_set = get_eval_set(opt.data_val_SID)
            testing_data_loader = DataLoader(datasets(test_set, num_workers=opt.threads, batch_size=1, shuffle=False))

        if opt.SICE_mix:
            train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
            training_data_loader = DataLoader(datasets(train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True))
            test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
            testing_data_loader = DataLoader(datasets(test_set, num_workers=opt.threads, batch_size=1, shuffle=False))

        if opt.SICE_grad:
            train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
            training_data_loader = DataLoader(datasets(train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True))
            test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
            testing_data_loader = DataLoader(datasets(test_set, num_workers=opt.threads, batch_size=1, shuffle=False))
    else:
        raise Exception("should choose a dataset")
    return training_data_loader, testing_data_loader

def build_model():
    print('===> Building model ')
    model = CIDNet().cuda()
    D_brightness = Discriminator().cuda()
    D_contrast = Discriminator().cuda()
    D_detail = Discriminator().cuda()
    if opt.start_epoch > 0:
        pth = os.path.join(opt.weights_folder, f"epoch_{opt.start_epoch}.pth")
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
    return model, D_brightness, D_contrast, D_detail

def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    optimizer_D_brightness = optim.Adam(D_brightness.parameters(), lr=opt.lr)
    optimizer_D_contrast = optim.Adam(D_contrast.parameters(), lr=opt.lr)
    optimizer_D_detail = optim.Adam(D_detail.parameters(), lr=opt.lr)
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[(opt.nEpochs//4)-opt.warmup_epochs, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[opt.nEpochs//4, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
    else:
        raise Exception("should choose a scheduler")
    return optimizer, scheduler, optimizer_D_brightness, optimizer_D_contrast, optimizer_D_detail

def init_loss():
    L1_weight   = opt.L1_weight
    D_weight    = opt.D_weight
    E_weight    = opt.E_weight
    P_weight    = 1.0

    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').cuda()
    #L1_loss = BrightnessPreservationLoss().cuda()
    D_loss = SSIM(weight=D_weight).cuda()
    #D_loss = ContrastPreservationLoss().cuda()
    E_loss = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').cuda()
    criterion_D = torch.nn.BCELoss().cuda()  # GAN判别器损失函数
    return L1_loss, P_loss, E_loss, D_loss, criterion_D

def selected_eval_config():
    norm_size = True
    if opt.lol_v1:
        return 'LOLv1', opt.data_valgt_lol_v1, norm_size, True, False
    if opt.lolv2_real:
        return 'LOLv2_real', opt.data_valgt_lolv2_real, norm_size, False, True
    if opt.lolv2_syn:
        return 'LOLv2_syn', opt.data_valgt_lolv2_syn, norm_size, False, False
    if opt.lol_blur:
        return 'LOL_blur', opt.data_valgt_lol_blur, norm_size, False, False
    if opt.SID:
        return 'SID', opt.data_valgt_SID, norm_size, False, False
    if opt.SICE_mix:
        return 'SICE_mix', opt.data_valgt_SICE_mix, False, False, False
    if opt.SICE_grad:
        return 'SICE_grad', opt.data_valgt_SICE_grad, False, False, False
    raise Exception("should choose a dataset")

def load_eval_dataset():
    print('===> Loading eval dataset')
    if opt.lol_v1:
        test_set = get_eval_set(opt.data_val_lol_v1)
    elif opt.lol_blur:
        test_set = get_eval_set(opt.data_val_lol_blur)
    elif opt.lolv2_real:
        test_set = get_eval_set(opt.data_val_lolv2_real)
    elif opt.lolv2_syn:
        test_set = get_eval_set(opt.data_val_lolv2_syn)
    elif opt.SID:
        test_set = get_eval_set(opt.data_val_SID)
    elif opt.SICE_mix:
        test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
    elif opt.SICE_grad:
        test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
    else:
        raise Exception("should choose a dataset")
    return DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

def run_eval_only():
    if not os.path.exists(opt.eval_weight):
        raise FileNotFoundError("eval_weight not found: {}".format(opt.eval_weight))

    dataset_name, label_dir, norm_size, is_lol, is_v2 = selected_eval_config()
    output_folder = os.path.join(opt.val_folder, dataset_name)
    if not output_folder.endswith(os.sep):
        output_folder += os.sep
    os.makedirs(opt.val_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    testing_data_loader = load_eval_dataset()
    eval_model = CIDNet().cuda()
    eval(eval_model, testing_data_loader, opt.eval_weight, output_folder,
         norm_size=norm_size, LOL=is_lol, v2=is_v2, alpha=0.8)

    im_dir = os.path.join(output_folder, '*.png')
    avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=opt.use_GT_mean)
    print("===> Eval weight: {}".format(opt.eval_weight))
    print("===> Dataset: {}".format(dataset_name))
    print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
    print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
    print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))

    metrics_dir = os.path.join(opt.val_folder, 'training')
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, 'eval_only_metrics.md')
    with open(metrics_path, "w") as f:
        f.write("dataset: " + dataset_name + "\n")
        f.write("weight: " + opt.eval_weight + "\n")
        f.write(f"use_GT_mean: {opt.use_GT_mean}\n")
        f.write("| PSNR | SSIM | LPIPS |\n")
        f.write("|----------------------|----------------------|----------------------|\n")
        f.write(f"| {avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} |\n")
    print("===> Metrics saved to {}".format(metrics_path))

if __name__ == '__main__':
    '''
    preparision
    '''
    train_init()
    if opt.eval_only:
        run_eval_only()
        sys.exit(0)

    training_data_loader, testing_data_loader = load_datasets()
    model, D_brightness, D_contrast, D_detail = build_model()
    optimizer, scheduler, optimizer_D_brightness, optimizer_D_contrast, optimizer_D_detail = make_scheduler()
    L1_loss, P_loss, E_loss, D_loss, criterion_D = init_loss()

    '''
    train
    '''
    psnr = []
    ssim = []
    lpips = []
    start_epoch = 0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):
        os.mkdir(opt.val_folder)

    for epoch in range(start_epoch+1, opt.nEpochs + start_epoch + 1):
        epoch_loss, pic_num = train(epoch)
        scheduler.step()
        ####################
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch)
            norm_size = True

            # LOL three subsets
            if opt.lol_v1:
                output_folder = 'LOLv1/'
                label_dir = opt.data_valgt_lol_v1
            if opt.lolv2_real:
                output_folder = 'LOLv2_real/'
                label_dir = opt.data_valgt_lolv2_real
            if opt.lolv2_syn:
                output_folder = 'LOLv2_syn/'
                label_dir = opt.data_valgt_lolv2_syn

            # LOL-blur dataset with low_blur and high_sharp_scaled
            if opt.lol_blur:
                output_folder = 'LOL_blur/'
                label_dir = opt.data_valgt_lol_blur

            if opt.SID:
                output_folder = 'SID/'
                label_dir = opt.data_valgt_SID
                npy = True
            if opt.SICE_mix:
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            if opt.SICE_grad:
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False

            output_path = os.path.join(opt.val_folder, output_folder)
            im_dir = os.path.join(output_path, '*.png')
            eval(model, testing_data_loader, model_out_path, output_path,
                 norm_size=norm_size, LOL=opt.lol_v1, v2=opt.lolv2_real, alpha=0.8)

            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)#################
            print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
            print(psnr)
            print(ssim)
            print(lpips)
        torch.cuda.empty_cache()

    metrics_dir = os.path.join(opt.val_folder, "training")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "metrics.md"), "w") as f:
        f.write("dataset: " + output_folder + "\n")
        f.write(f"lr: {opt.lr}\n")
        f.write(f"batch size: {opt.batchSize}\n")
        f.write(f"crop size: {opt.cropSize}\n")
        f.write(f"HVI_weight: {opt.HVI_weight}\n")
        f.write(f"L1_weight: {opt.L1_weight}\n")
        f.write(f"D_weight: {opt.D_weight}\n")
        f.write(f"E_weight: {opt.E_weight}\n")
        f.write(f"P_weight: {opt.P_weight}\n")
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")
        f.write("|----------------------|----------------------|----------------------|----------------------|\n")
        for i in range(len(psnr)):
            f.write(f"| {(i+1)*10} | {psnr[i]:.4f} | {ssim[i]:.4f} | {lpips[i]:.4f} |\n")
