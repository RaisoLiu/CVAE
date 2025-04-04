import os
import sys

from PIL import Image


def merge_gifs(gif1_path, gif2_path, output_path):
    """
    將兩個 GIF 檔案上下合併成一個新的 GIF 檔案

    Args:
        gif1_path (str): 第一個 GIF 檔案的路徑
        gif2_path (str): 第二個 GIF 檔案的路徑
        output_path (str): 輸出 GIF 檔案的路徑
    """
    # 開啟兩個 GIF 檔案
    gif1 = Image.open(gif1_path)
    gif2 = Image.open(gif2_path)

    # 確保兩個 GIF 的寬度相同
    if gif1.width != gif2.width:
        print("警告：兩個 GIF 的寬度不同，將進行調整")
        # 調整第二個 GIF 的寬度以匹配第一個
        gif2 = gif2.resize((gif1.width, int(gif2.height * (gif1.width / gif2.width))))

    # 創建新的空白圖像，高度為兩個 GIF 的總和
    new_height = gif1.height + gif2.height
    new_width = gif1.width

    # 獲取兩個 GIF 的幀數
    frames1 = []
    frames2 = []

    try:
        while True:
            frames1.append(gif1.copy())
            gif1.seek(len(frames1))
    except EOFError:
        pass

    try:
        while True:
            frames2.append(gif2.copy())
            gif2.seek(len(frames2))
    except EOFError:
        pass

    # 確保兩個 GIF 的幀數相同
    min_frames = min(len(frames1), len(frames2))
    frames1 = frames1[:min_frames]
    frames2 = frames2[:min_frames]

    # 合併每一幀
    merged_frames = []
    for frame1, frame2 in zip(frames1, frames2):
        new_frame = Image.new("RGB", (new_width, new_height))
        new_frame.paste(frame1, (0, 0))
        new_frame.paste(frame2, (0, gif1.height))
        merged_frames.append(new_frame)

    # 保存合併後的 GIF
    merged_frames[0].save(
        output_path,
        save_all=True,
        append_images=merged_frames[1:],
        duration=gif1.info.get("duration", 100),
        loop=0,
    )

    print(f"已成功合併 GIF 並保存到: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("使用方法: python merge_gifs.py <gif1_path> <gif2_path> <output_path>")
        sys.exit(1)

    gif1_path = sys.argv[1]
    gif2_path = sys.argv[2]
    output_path = sys.argv[3]

    merge_gifs(gif1_path, gif2_path, output_path)
