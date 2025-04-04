import argparse
import glob
import os
from math import log10

import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import stack
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from modules import (
    Decoder_Fusion,
    Gaussian_Predictor,
    Generator,
    Label_Encoder,
    RGB_Encoder,
    VAE_Model,
)

TA_ = """
 ██████╗ ██████╗ ███╗   ██╗ ██████╗ ██████╗  █████╗ ████████╗██╗   ██╗██╗      █████╗ ████████╗██╗ ██████╗ ███╗   ██╗███████╗    ██╗██╗██╗
██╔════╝██╔═══██╗████╗  ██║██╔════╝ ██╔══██╗██╔══██╗╚══██╔══╝██║   ██║██║     ██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝    ██║██║██║
██║     ██║   ██║██╔██╗ ██║██║  ███╗██████╔╝███████║   ██║   ██║   ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║███████╗    ██║██║██║
██║     ██║   ██║██║╚██╗██║██║   ██║██╔══██╗██╔══██║   ██║   ██║   ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║╚════██║    ╚═╝╚═╝╚═╝
╚██████╗╚██████╔╝██║ ╚████║╚██████╔╝██║  ██║██║  ██║   ██║   ╚██████╔╝███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║███████║    ██╗██╗██╗
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝    ╚═╝╚═╝╚═╝                                                                                                                          
"""


def get_key(fp):
    filename = fp.split("/")[-1]
    filename = filename.split(".")[0].replace("frame", "")
    return int(filename)


from glob import glob

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as torchData
from torchvision.datasets.folder import default_loader as imgloader
from dataloader import Dataset_Dance

class Dataset_Dance(torchData):
    def __init__(self, root, transform, mode="test", video_len=7, partial=1.0):
        super().__init__()
        self.img_folder = []
        self.label_folder = []

        data_num = len(glob(os.path.join(root, f"test/test_img/*")))
        for i in range(data_num):
            self.img_folder.append(
                sorted(glob(os.path.join(root, f"test/test_img/{i}/*")), key=get_key)
            )
            self.label_folder.append(
                sorted(glob(os.path.join(root, f"test/test_label/{i}/*")), key=get_key)
            )

        self.transform = transform

    def __len__(self):
        return len(self.img_folder)

    def __getitem__(self, index):
        frame_seq = self.img_folder[index]
        label_seq = self.label_folder[index]

        imgs = []
        labels = []
        imgs.append(self.transform(imgloader(frame_seq[0])))
        for idx in range(len(label_seq)):
            labels.append(self.transform(imgloader(label_seq[idx])))
        return stack(imgs), stack(labels)


class Test_model(VAE_Model):
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

        self.Generator = Generator(input_nc=args.D_out_dim, output_nc=3)

        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0

        self.val_vi_len = args.val_vi_len
        self.batch_size = args.batch_size

    def forward(self, img, label):
        pass

    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        pred_seq_list = []
        print("Evaluating..., length of val_loader:", len(val_loader))
        for idx, (img, label) in enumerate(val_loader):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            pred_seq = self.val_one_step(img, label, idx)
            pred_seq_list.append(pred_seq)

        # submission.csv is the file you should submit to kaggle
        pred_to_int = (np.rint(torch.cat(pred_seq_list).numpy() * 255)).astype(int)
        df = pd.DataFrame(pred_to_int)
        df.insert(0, "id", range(0, len(df)))
        df.to_csv(
            os.path.join(self.args.save_root, f"submission.csv"),
            header=True,
            index=False,
        )

    def val_one_step(self, img, label, idx=0):
        batch_size, time_step, channel, height, width = label.shape
        f_dim, l_dim, n_dim = self.args.F_dim, self.args.L_dim, self.args.N_dim



        generated_frames = []

        no_head_label = label[:,1:].reshape(-1, channel, height, width)
        no_head_label_emb = self.label_transformation(no_head_label)
        no_head_label_emb = no_head_label_emb.view(batch_size, time_step-1, l_dim, height, width)
        prev_frame = img[:,0].reshape(-1, channel, height, width)
        prev_frame_emb = self.frame_transformation(prev_frame)
        prev_z = torch.randn(batch_size, n_dim, height, width).to(self.args.device)
        generated_frames.append(prev_frame[0].cpu())

        for i in tqdm(range(1, time_step)):
            decoded = self.Decoder_Fusion(prev_frame_emb, no_head_label_emb[:,i-1], prev_z)
            img_hat = self.Generator(decoded)
            prev_frame = img_hat
            prev_frame_emb = self.frame_transformation(prev_frame)
            z, _, _ = self.Gaussian_Predictor(prev_frame_emb, no_head_label_emb[:,i-1])
            prev_z = z
            generated_frames.append(img_hat[0].cpu())
        
    
        generated_frames = torch.stack(generated_frames, dim=0)        
        print('generated_frame:', generated_frames.shape)

        generated_frames = generated_frames.unsqueeze(0)
        assert generated_frames.shape == (
            1,
            630,
            3,
            32,
            64,
        ), f"The shape of output should be (1, 630, 3, 32, 64), but your output shape is {generated_frames.shape}"

        self.make_gif(
            generated_frames[0], os.path.join(self.args.save_root, f"pred_seq{idx}.gif")
        )

        # Reshape the generated frame to (630, 3 * 64 * 32)
        generated_frames = generated_frames.reshape(630, -1)

        return generated_frames

    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))

        new_list[0].save(
            img_name,
            format="GIF",
            append_images=new_list,
            save_all=True,
            duration=20,
            loop=0,
        )

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor(),
            ]
        )
        dataset = Dataset_Dance(
            root=self.args.DR, transform=transform, video_len=self.val_vi_len
        )
        print('Length of dataset:', len(dataset))
        val_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.args.num_workers,
            drop_last=True,
            shuffle=False,
        )
        return val_loader

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint["state_dict"], strict=True)


def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    model = Test_model(args).to(args.device)
    model.load_checkpoint()
    model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--optim", type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--no_sanity", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--make_gif", action="store_true")
    parser.add_argument("--DR", type=str, required=True, help="Your Dataset Path")
    parser.add_argument(
        "--save_root", type=str, required=True, help="The path to save your data"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--num_epoch", type=int, default=70, help="number of total epoch"
    )
    parser.add_argument(
        "--per_save", type=int, default=3, help="Save checkpoint every seted epoch"
    )
    parser.add_argument(
        "--partial",
        type=float,
        default=1.0,
        help="Part of the training dataset to be trained",
    )
    parser.add_argument(
        "--train_vi_len", type=int, default=16, help="Training video length"
    )
    parser.add_argument(
        "--val_vi_len", type=int, default=630, help="valdation video length"
    )
    parser.add_argument(
        "--frame_H", type=int, default=32, help="Height input image to be resize"
    )
    parser.add_argument(
        "--frame_W", type=int, default=64, help="Width input image to be resize"
    )

    # Module parameters setting
    parser.add_argument(
        "--F_dim", type=int, default=128, help="Dimension of feature human frame"
    )
    parser.add_argument(
        "--L_dim", type=int, default=32, help="Dimension of feature label frame"
    )
    parser.add_argument("--N_dim", type=int, default=12, help="Dimension of the Noise")
    parser.add_argument(
        "--D_out_dim",
        type=int,
        default=192,
        help="Dimension of the output in Decoder_Fusion",
    )

    # Teacher Forcing strategy
    parser.add_argument(
        "--tfr", type=float, default=1.0, help="The initial teacher forcing ratio"
    )
    parser.add_argument(
        "--tfr_sde",
        type=int,
        default=10,
        help="The epoch that teacher forcing ratio start to decay",
    )
    parser.add_argument(
        "--tfr_d_step",
        type=float,
        default=0.1,
        help="Decay step that teacher forcing ratio adopted",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="The path of your checkpoints"
    )

    # Training Strategy
    parser.add_argument("--fast_train", action="store_true")
    parser.add_argument(
        "--fast_partial",
        type=float,
        default=0.4,
        help="Use part of the training data to fasten the convergence",
    )
    parser.add_argument(
        "--fast_train_epoch",
        type=int,
        default=5,
        help="Number of epoch to use fast train mode",
    )

    # Kl annealing stratedy arguments
    parser.add_argument("--kl_anneal_type", type=str, default="Cyclical", help="")
    parser.add_argument("--kl_anneal_cycle", type=int, default=10, help="")
    parser.add_argument("--kl_anneal_ratio", type=float, default=1, help="")

    args = parser.parse_args()

    main(args)
