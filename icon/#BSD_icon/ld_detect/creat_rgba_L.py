#!/usr/bin/env python3
from PIL import Image, ImageDraw
import math, os

def create_left_triangle_and_right_rect(
        size=(512, 256),       # (width, height)
        padding=20,            # 圖形與邊緣的間距
        rect_width=60,         # 右側長方形寬度 (像素)
        triangle_color=(128,128,128,255),  # RGBA 填色（灰色）
        out_file="triangle_left_rect_right.png"):
    W, H = size
    pad = padding

    if rect_width <= 0:
        raise ValueError("rect_width must be > 0")

    # 可用垂直高度給三角形（等邊三角形邊長 s = 可用高度）
    max_s_by_height = max(0, H - 2 * pad)
    if max_s_by_height <= 0:
        raise ValueError("畫布太小或 padding 太大，無法繪製")

    # 預設三角形邊長 s，及對應的水平寬 tri_horiz = (√3/2) * s
    s = max_s_by_height
    tri_horiz = (math.sqrt(3) / 2.0) * s

    # 計算三角形可用水平空間（左右兩側留 pad，右側還要保留 rect_width 給矩形）
    available_for_triangle = W - pad - rect_width - pad
    if available_for_triangle <= 0:
        raise ValueError("畫布太窄，無法同時放下三角形與右側矩形（請減小 rect_width 或 padding）")

    # 若預設 tri_horiz 超過可用寬度，縮放三角形（保持等邊形比例）
    if tri_horiz > available_for_triangle:
        tri_horiz = available_for_triangle
        s = (2.0 / math.sqrt(3)) * tri_horiz
        # 再次確保 s 不超過可垂直高度
        if s > max_s_by_height:
            s = max_s_by_height
            tri_horiz = (math.sqrt(3) / 2.0) * s

    # 三角形的垂直半高
    tri_half = s / 2.0
    mid_y = H / 2.0

    # 三角形頂點（尖端朝左）
    tip = (pad, mid_y)  # 尖端在左側靠 pad
    top_base = (pad + tri_horiz, mid_y - tri_half)
    bottom_base = (pad + tri_horiz, mid_y + tri_half)
    triangle = [top_base, tip, bottom_base]

    # 右側長方形座標（從 W - pad - rect_width 到 W - pad）
    rect_left = W - pad - rect_width
    rect_top = pad
    rect_right = W - pad
    rect_bottom = H - pad
    # 若矩形意外地跟三角形重疊（極端小畫布），矩形仍照給定寬度畫出；前面已盡量避免
    rect = [(rect_left, rect_top), (rect_right, rect_bottom)]

    # 建透明底圖並繪製（按照要求：先畫三角形，再畫右側長方形）
    img = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(img)

    # 1) 先畫三角形（填滿灰色）
    triangle_int = [(int(round(x)), int(round(y))) for (x,y) in triangle]
    draw.polygon(triangle_int, fill=triangle_color)

    # 2) 再畫右側長方形（填滿灰色）
    rect_int = [(int(round(rect_left)), int(round(rect_top))), (int(round(rect_right)), int(round(rect_bottom)))]
    draw.rectangle(rect_int, fill=triangle_color)

    # 儲存
    img.save(out_file)
    print(f"Saved: {os.path.abspath(out_file)} (size={W}x{H})")
    return img

if __name__ == "__main__":
    # 範例：512x256 圖片，右側長方形寬度 60 px
    create_left_triangle_and_right_rect(
        size=(600, 512),
        padding=24,
        rect_width=80,
        triangle_color=(200,200,150,255),
        out_file="rect_triangle_L.png"
    )
