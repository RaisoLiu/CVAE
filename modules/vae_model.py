import gc
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader import Dataset_Dance

from .kl_annealing import kl_annealing
from .modules import (
    Decoder_Fusion,
    Gaussian_Predictor,
    Generator,
    Label_Encoder,
    RGB_Encoder,
)


def Generate_PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args

        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)

        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor = Gaussian_Predictor(
            args.F_dim + args.L_dim, args.N_dim
        )
        self.Decoder_Fusion = Decoder_Fusion(
            args.F_dim + args.L_dim + args.N_dim, args.D_out_dim
        )

        # Generative model
        self.Generator = Generator(input_nc=args.D_out_dim, output_nc=3)

        self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=self.args.milestones, gamma=self.args.gamma
        )
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 1

        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde

        self.train_vi_len = args.train_vi_len
        self.val_vi_len = args.val_vi_len
        self.batch_size = args.batch_size
        self.writer = tensorboard.SummaryWriter(
            f"{self.args.save_root}/kl-type_{args.kl_anneal_type}_tfr_{args.tfr}_teacher-decay_{args.tfr_d_step}"
        )

        # AMP scaler
        self.scaler = GradScaler("cuda")

    def forward(self, img, label):
        pass

    def training_stage(self):
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False

            train_losses = []
            for img, label in (pbar := tqdm(train_loader, ncols=120)):

                img = img.to(self.args.device)
                label = label.to(self.args.device)
                if adapt_TeacherForcing:
                    loss = self.training_one_step_with_teacher_forcing(img, label)
                else:
                    loss = self.training_one_step_without_teacher_forcing(img, label)
                train_losses.append(loss.detach().cpu())

                beta = self.kl_annealing.get_beta()

                if adapt_TeacherForcing:
                    self.tqdm_bar(
                        "train [TeacherForcing: ON, {:.1f}], beta: {}".format(
                            self.tfr, beta
                        ),
                        pbar,
                        loss.detach().cpu(),
                        lr=self.scheduler.get_last_lr()[0],
                    )
                else:
                    self.tqdm_bar(
                        "train [TeacherForcing: OFF, {:.1f}], beta: {}".format(
                            self.tfr, beta
                        ),
                        pbar,
                        loss.detach().cpu(),
                        lr=self.scheduler.get_last_lr()[0],
                    )

            if self.current_epoch % self.args.per_save == 0:
                self.save(
                    os.path.join(
                        self.args.save_root, f"epoch-{self.current_epoch}.ckpt"
                    )
                )


            self.writer.add_scalar(
                "Loss/train", np.mean(train_losses), self.current_epoch
            )
            self.writer.add_scalar("beta", beta, self.current_epoch)
            self.writer.add_scalar("tfr", self.tfr, self.current_epoch)
            val_losses, PSNRS = self.eval()

            print(
                f"Epoch {self.current_epoch} train loss: {np.mean(train_losses):.6f}, val loss: {np.mean(val_losses):.6f}, PSNR: {np.mean(PSNRS):.4f}, beta: {beta:.4f}, tfr: {self.tfr:.2f}"
            )
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            gc.collect()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        PSNRS = []
        val_losses = []
        # 為第一個batch保存生成的圖像
        save_imgs = []
        first_batch = True

        for img, label in val_loader:
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, PSNR, generated_imgs = self.val_one_step(img, label)
            val_losses.append(loss.detach().cpu())
            PSNRS.append(PSNR)

            # 只保存第一個batch的結果
            if first_batch:
                save_imgs.extend([img.cpu() for img in generated_imgs])
                first_batch = False



        # 保存GIF
        if len(save_imgs) > 0 and self.current_epoch % self.args.per_save == 0:
            save_path = os.path.join(
                self.args.save_root, f"val_demo_epoch_{self.current_epoch}.gif"
            )
            self.make_gif(save_imgs, save_path)
            print(f"Saved validation demo GIF to {save_path}")


        self.writer.add_scalar("Loss/val", np.mean(val_losses), self.current_epoch)
        self.writer.add_scalar("PSNR/val", np.mean(PSNRS), self.current_epoch)
        return np.mean(val_losses), np.mean(PSNRS)

    # def training_one_step_without_teacher_forcing(self, img, label):
    #     batch_size, time_step, channel, height, width = img.shape
    #     beta = self.kl_annealing.get_beta()

    #     with autocast("cuda"):
    #         no_head_label_emb, _, _ = self._process_frame_features(img, label)

    #         prev_frame = img[:, 0].reshape(-1, channel, height, width)
    #         prev_frame_emb = self.frame_transformation(prev_frame)
    #         prev_z = torch.randn(batch_size, self.args.N_dim, height, width).to(
    #             self.args.device
    #         )

    #         pred_no_head_img_list, mu_list, logvar_list = [], [], []
    #         no_head_label_emb = no_head_label_emb.view(batch_size, time_step-1, self.args.L_dim, height, width)
    #         for i in range(time_step-1):
    #             img_hat = self._generate_next_frame(
    #                 prev_frame_emb, no_head_label_emb[:, i], prev_z
    #             )
    #             pred_no_head_img_list.append(img_hat.detach())

    #             prev_frame_emb = self.frame_transformation(img_hat)
    #             z, mu, logvar = self.Gaussian_Predictor(
    #                 prev_frame_emb, no_head_label_emb[:, i]
    #             )
    #             prev_z = z
    #             mu_list.append(mu)
    #             logvar_list.append(logvar)

    #         pred_no_head_img_list = torch.stack(pred_no_head_img_list, dim=1)
    #         mse = self.mse_criterion(
    #             pred_no_head_img_list.view(-1, channel, height, width),
    #             img[:, 1:].reshape(-1, channel, height, width),
    #         )

    #         mu = torch.stack(mu_list, dim=1)
    #         logvar = torch.stack(logvar_list, dim=1)
    #         kld = self.kl_criterion(mu, logvar)

    #         loss = self._update_training_metrics(mse, kld, beta)

    #     self.optim.zero_grad()
    #     self.scaler.scale(loss).backward()
    #     self.scaler.step(self.optim)
    #     self.scaler.update()
    #     return loss

    # def training_one_step_with_teacher_forcing(self, img, label):
    #     beta = self.kl_annealing.get_beta()

    #     with autocast("cuda"):
    #         no_head_label_emb, no_head_frame_emb, no_tail_frame_emb = (
    #             self._process_frame_features(img, label)
    #         )

    #         # Ensure dimensions match
    #         assert (
    #             no_head_frame_emb.dim() == no_head_label_emb.dim()
    #         ), f"Dimension mismatch: frame_emb {no_head_frame_emb.shape}, label_emb {no_head_label_emb.shape}"

    #         z, mu, logvar = self.Gaussian_Predictor(
    #             no_head_frame_emb, no_head_label_emb
    #         )
    #         img_hat = self._generate_next_frame(no_tail_frame_emb, no_head_label_emb, z)

    #         kld = self.kl_criterion(mu, logvar)
    #         mse = self.mse_criterion(img_hat, img[:, 1:].reshape(-1, *img.shape[2:]))
    #         loss = self._update_training_metrics(mse, kld, beta)

    #     self.optim.zero_grad()
    #     self.scaler.scale(loss).backward()
    #     self.scaler.step(self.optim)
    #     self.scaler.update()
    #     return loss

    def kl_criterion(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= self.batch_size
        return KLD

    # def val_one_step(self, img, label):
    #     batch_size, time_step, channel, height, width = img.shape
    #     beta = self.kl_annealing.get_beta()
    #     PSNR = []
    #     generated_frames = []

    #     with autocast("cuda"):
    #         no_head_label_emb, _, _ = self._process_frame_features(img, label)

    #         prev_frame = img[:, 0].reshape(-1, channel, height, width)
    #         prev_frame_emb = self.frame_transformation(prev_frame)
    #         prev_z = torch.randn(batch_size, self.args.N_dim, height, width).to(
    #             self.args.device
    #         )

    #         generated_frames.append(prev_frame[0].cpu())
    #         pred_no_head_img = []
    #         mu_list = []
    #         logvar_list = []

    #         for i in range(1, time_step):
    #             # Ensure label_emb has correct dimensions
    #             current_label_emb = no_head_label_emb[i - 1 : i].squeeze(0)
    #             if current_label_emb.dim() == 3:
    #                 current_label_emb = current_label_emb.unsqueeze(0)

    #             img_hat = self._generate_next_frame(
    #                 prev_frame_emb, current_label_emb, prev_z
    #             )
    #             pred_no_head_img.append(img_hat.detach())

    #             prev_frame = img_hat
    #             prev_frame_emb = self.frame_transformation(prev_frame)
    #             z, mu, logvar = self.Gaussian_Predictor(
    #                 prev_frame_emb, current_label_emb
    #             )
    #             prev_z = z
    #             mu_list.append(mu)
    #             logvar_list.append(logvar)

    #             PSNR.append(Generate_PSNR(img_hat, img[:, i]).item())
    #             generated_frames.append(img_hat[0].cpu())

    #         pred_no_head_img = torch.stack(pred_no_head_img, dim=1)
    #         mse = self.mse_criterion(
    #             pred_no_head_img.view(-1, channel, height, width),
    #             img[:, 1:].reshape(-1, channel, height, width),
    #         )

    #         mu = torch.stack(mu_list, dim=1)
    #         logvar = torch.stack(logvar_list, dim=1)
    #         kld = self.kl_criterion(mu, logvar)

    #         self.writer.add_scalar(
    #             "kld/val", kld.item(), self.current_epoch
    #         )
    #         self.writer.add_scalar(
    #             "mse/val", mse.item(), self.current_epoch
    #         )
    #         loss = mse + beta * kld

    #     return loss, np.mean(PSNR), generated_frames
    def training_one_step_without_teacher_forcing(self, img, label):
        # img_batch: (Batch_size, Time_step, Channel, Height, Width) = (2, 16, 3, 32, 64)
        batch_size, time_step, channel, height, width = img.shape
        f_dim, l_dim, n_dim = self.args.F_dim, self.args.L_dim, self.args.N_dim
        beta = self.kl_annealing.get_beta()

        no_head_label = label[:,1:].reshape(-1, channel, height, width)
        no_head_label_emb = self.label_transformation(no_head_label)
        no_head_label_emb = no_head_label_emb.view(batch_size, time_step-1, l_dim, height, width)

        prev_frame = img[:,0].reshape(-1, channel, height, width)
        prev_frame_emb = self.frame_transformation(prev_frame).detach()
        prev_z = torch.randn(batch_size, n_dim, height, width).to(self.args.device)
        pred_no_head_img, mu_list, logvar_list = [], [], []
        for i in range(1, time_step):
            decoded = self.Decoder_Fusion(prev_frame_emb, no_head_label_emb[:,i-1], prev_z)
            img_hat = self.Generator(decoded)
            pred_no_head_img.append(img_hat.detach())
            prev_frame = img_hat
            prev_frame_emb = self.frame_transformation(prev_frame).detach()
            z, mu, logvar = self.Gaussian_Predictor(prev_frame_emb, no_head_label_emb[:,i-1])
            prev_z = z.detach()
            mu_list.append(mu)
            logvar_list.append(logvar)

        

        pred_no_head_img = torch.stack(pred_no_head_img, dim=1)
        mu = torch.stack(mu_list, dim=1)
        logvar = torch.stack(logvar_list, dim=1)
        mse = self.mse_criterion(pred_no_head_img.view(-1, channel, height, width), img[:,1:].reshape(-1, channel, height, width))
        kld = self.kl_criterion(mu, logvar)
        loss = mse + beta * kld

        self.optim.zero_grad()
        loss.backward()
        self.optimizer_step()
        self.writer.add_scalar(
            "kld/train", kld.item(), self.current_epoch
        )
        self.writer.add_scalar(
            "mse/train", mse.item(), self.current_epoch
        )
        return loss

        

  

    def training_one_step_with_teacher_forcing(self, img, label):   
        batch_size, time_step, channel, height, width = img.shape
        f_dim, l_dim, n_dim = self.args.F_dim, self.args.L_dim, self.args.N_dim
        beta = self.kl_annealing.get_beta()
        

        frame_emb = self.frame_transformation(img.view(-1, channel, height, width))
        frame_emb = frame_emb.view(batch_size, time_step, f_dim, height, width)
        no_head_frame_emb = frame_emb[:,1:].reshape(-1, f_dim, height, width)
        no_tail_frame_emb = frame_emb[:,:-1].reshape(-1, f_dim, height, width)
        no_head_label = label[:,1:].reshape(-1, channel, height, width)
        no_head_label_emb = self.label_transformation(no_head_label)
        z, mu, logvar = self.Gaussian_Predictor(no_head_frame_emb, no_head_label_emb)
        
        decoded = self.Decoder_Fusion(no_tail_frame_emb, no_head_label_emb, z)
        img_hat = self.Generator(decoded)
    
        kld = self.kl_criterion(mu, logvar)
        mse = self.mse_criterion(img_hat, img[:,1:].reshape(-1, channel, height, width))
        loss = mse + beta * kld

        self.optim.zero_grad()
        loss.backward()
        self.optimizer_step()
        self.writer.add_scalar(
            "kld/train", kld.item(), self.current_epoch
        )
        self.writer.add_scalar(
            "mse/train", mse.item(), self.current_epoch
        )
        return loss


    def val_one_step(self, img, label):
        batch_size, time_step, channel, height, width = img.shape
        f_dim, l_dim, n_dim = self.args.F_dim, self.args.L_dim, self.args.N_dim
        beta = self.kl_annealing.get_beta()

        PSNR = []
        generated_frames = []

        no_head_label = label[:,1:].reshape(-1, channel, height, width)
        no_head_label_emb = self.label_transformation(no_head_label)
        no_head_label_emb = no_head_label_emb.view(batch_size, time_step-1, l_dim, height, width)
        prev_frame = img[:,0].reshape(-1, channel, height, width)
        prev_frame_emb = self.frame_transformation(prev_frame)
        prev_z = torch.randn(batch_size, n_dim, height, width).to(self.args.device)
        pred_no_head_img = []
        mu_list = []
        logvar_list = []
        
        # 將第一幀的 label 和預測圖像上下排列
        first_frame = torch.cat([label[0,0], prev_frame[0]], dim=1)  # 在高度維度上堆疊
        generated_frames.append(first_frame.cpu())

        for i in range(1, time_step):
            decoded = self.Decoder_Fusion(prev_frame_emb, no_head_label_emb[:,i-1], prev_z)
            img_hat = self.Generator(decoded)
            pred_no_head_img.append(img_hat)
            prev_frame = img_hat
            prev_frame_emb = self.frame_transformation(prev_frame)
            z, mu, logvar = self.Gaussian_Predictor(prev_frame_emb, no_head_label_emb[:,i-1])
            prev_z = z
            mu_list.append(mu)
            logvar_list.append(logvar)
            PSNR.append(Generate_PSNR(img_hat, img[:,i]).item())
            
            # 將 label 和預測圖像上下排列
            combined_frame = torch.cat([label[0,i], img_hat[0]], dim=1)  # 在高度維度上堆疊
            generated_frames.append(combined_frame.cpu())
        

        pred_no_head_img = torch.stack(pred_no_head_img, dim=1)        
        mu = torch.stack(mu_list, dim=1)
        logvar = torch.stack(logvar_list, dim=1)
        kld = self.kl_criterion(mu, logvar)
        mse = self.mse_criterion(pred_no_head_img.view(-1, channel, height, width), img[:,1:].reshape(-1, channel, height, width))
        loss = mse + beta * kld

        self.writer.add_scalar(
            "kld/val", kld.item(), self.current_epoch
        )
        self.writer.add_scalar(
            "mse/val", mse.item(), self.current_epoch
        )

        return loss, np.mean(PSNR), generated_frames

    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))

        new_list[0].save(
            img_name,
            format="GIF",
            append_images=new_list[1:],
            save_all=True,
            duration=60,
            loop=0,
        )

    def _get_image_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor(),
            ]
        )

    def _get_dataloader(self, mode, video_len, partial, batch_size=None):
        transform = self._get_image_transforms()
        dataset = Dataset_Dance(
            root=self.args.DR,
            transform=transform,
            mode=mode,
            video_len=video_len,
            partial=partial,
        )

        if batch_size is None:
            batch_size = self.batch_size if mode == "train" else 1

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.args.num_workers,
            drop_last=True,
            shuffle=(mode == "train"),
            pin_memory=True,
            persistent_workers=True,
        )

    def _process_frame_features(self, img, label):
        batch_size, time_step, channel, height, width = img.shape
        f_dim, l_dim, n_dim = self.args.F_dim, self.args.L_dim, self.args.N_dim

        # Process label features
        no_head_label = label[:, 1:].reshape(-1, channel, height, width)
        no_head_label_emb = self.label_transformation(no_head_label)
        no_head_label_emb = no_head_label_emb.view(
            batch_size, time_step - 1, l_dim, height, width
        )

        # Process frame features
        frame_emb = self.frame_transformation(img.view(-1, channel, height, width))
        frame_emb = frame_emb.view(batch_size, time_step, f_dim, height, width)
        no_head_frame_emb = frame_emb[:, 1:].reshape(-1, f_dim, height, width)
        no_tail_frame_emb = frame_emb[:, :-1].reshape(-1, f_dim, height, width)

        # Reshape label embeddings to match frame embeddings
        no_head_label_emb = no_head_label_emb.reshape(-1, l_dim, height, width)

        return no_head_label_emb, no_head_frame_emb, no_tail_frame_emb

    def _generate_next_frame(self, prev_frame_emb, label_emb, z):
        img_hat = self.Generator(self.Decoder_Fusion(prev_frame_emb, label_emb, z))
        return torch.clamp(img_hat, 0.0, 1.0)

    def _update_training_metrics(self, mse, kld, beta):
        self.writer.add_scalar("mse/train", mse.item(), self.current_epoch)
        self.writer.add_scalar("kld/train", kld.item(), self.current_epoch)
        return mse + beta * kld

    def train_dataloader(self):
        partial = self.args.fast_partial if self.args.fast_train else self.args.partial
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
        return self._get_dataloader("train", self.train_vi_len, partial)

    def val_dataloader(self):
        return self._get_dataloader("val", self.val_vi_len, 1.0)

    def teacher_forcing_ratio_update(self):
        if (
            self.current_epoch >= self.tfr_sde
            and self.current_epoch % self.tfr_sde == 0
        ):
            self.tfr -= self.tfr_d_step
            self.tfr = max(0, self.tfr)

    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(
            f"({mode}) Epoch {self.current_epoch}, lr:{lr}", refresh=False
        )
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

    def save(self, path):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "lr": self.scheduler.get_last_lr()[0],
                "tfr": self.tfr,
                "last_epoch": self.current_epoch,
            },
            path,
        )
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint["state_dict"], strict=True)
            # self.args.lr, self.tfr = checkpoint['lr'], checkpoint['tfr']
            # self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            # self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            # self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            # self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optim.step()
