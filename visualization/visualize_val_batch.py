import argparse
import os

import imageio
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import Dataset_Dance


def make_gif(images_list, img_name):
    new_list = []
    for img in images_list:
        new_list.append(transforms.ToPILImage()(img))

    new_list[0].save(
        img_name,
        format="GIF",
        append_images=new_list,
        save_all=True,
        duration=40,
        loop=0,
    )


def main(args):
    # 設定轉換
    transform = transforms.Compose(
        [transforms.Resize((args.frame_H, args.frame_W)), transforms.ToTensor()]
    )

    # 載入驗證資料集
    dataset = Dataset_Dance(
        root=args.DR,
        transform=transform,
        mode="val",
        video_len=args.val_vi_len,
        partial=1.0,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    # 取得第一個 batch
    for batch_idx, (img, label) in enumerate(val_loader):
        if batch_idx == 0:  # 只處理第一個 batch
            # 準備儲存目錄
            os.makedirs(args.save_root, exist_ok=True)

            # 處理原始幀
            frames = []
            for i in range(img.shape[1]):  # 遍歷時間步
                frame = img[0, i]  # 取得第一個樣本的第 i 幀
                frames.append(frame)

            # 處理標籤幀
            label_frames = []
            for i in range(label.shape[1]):  # 遍歷時間步
                label_frame = label[0, i]  # 取得第一個樣本的第 i 幀標籤
                label_frames.append(label_frame)

            # 儲存 GIF
            make_gif(frames, os.path.join(args.save_root, "validation_frames.gif"))
            make_gif(
                label_frames, os.path.join(args.save_root, "validation_labels.gif")
            )

            print(f"已儲存 GIF 到 {args.save_root}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--DR", type=str, required=True, help="資料集路徑")
    parser.add_argument("--save_root", type=str, required=True, help="儲存結果的路徑")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_vi_len", type=int, default=630, help="驗證影片長度")
    parser.add_argument("--frame_H", type=int, default=32, help="輸入圖像的高度")
    parser.add_argument("--frame_W", type=int, default=64, help="輸入圖像的寬度")

    args = parser.parse_args()
    main(args)
