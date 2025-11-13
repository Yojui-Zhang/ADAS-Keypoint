#!/usr/bin/env python3
from PIL import Image, ImageDraw
import math
import os

def create_rect_plus_right_equilateral_triangle(
        size=(300, 200),      # (width, height)
        padding=20,           # 圖形四周留白
        rect_width=10,        # 左側長方形寬度 (像素)
        triangle_color=(128,128,128,255),  # RGBA 灰色
        out_file="rect_triangle.png"):
    W, H = size
    pad = padding

    if rect_width <= 0:
        raise ValueError("rect_width must be > 0")

    # 可用垂直高度給三角形（base 的長度 s = tri_h），初始 tri_h 為 H - 2*pad
    tri_h = max(0, H - 2*pad)
    if tri_h <= 0:
        raise ValueError("畫布太小或 padding 太大，無法繪製")

    # 計算等邊三角形的水平寬度 h = (sqrt(3)/2) * s
    s = tri_h
    tri_horiz = (math.sqrt(3) / 2.0) * s

    # 計算所需總寬度：pad + rect_width + tri_horiz + pad
    needed_width = pad + rect_width + tri_horiz + pad
    if needed_width > W:
        # 縮放 tri_h 以適配畫布寬度（保持三角形為等邊）
        available_for_triangle = W - pad - rect_width - pad
        if available_for_triangle <= 0:
            raise ValueError("畫布太窄，無法容納指定的 rect_width 與 padding")
        # 由 tri_horiz_new = available_for_triangle, 得到 s_new = 2/√3 * tri_horiz_new
        tri_horiz = available_for_triangle
        s = (2.0 / math.sqrt(3)) * tri_horiz
        # 確保 s 不超過可垂直高度
        max_s_by_height = H - 2*pad
        if s > max_s_by_height:
            s = max_s_by_height
            tri_horiz = (math.sqrt(3) / 2.0) * s

    # 現在 tri 高度 s、水平寬 tri_horiz 都已計算好
    tri_height = s
    tri_half = tri_height / 2.0
    # 基線 x 座標（base 的 x）
    base_x = pad + rect_width + 60
    # 三角形頂點座標（頂點在右側）
    mid_y = H / 2.0
    top_vertex = (base_x, mid_y - tri_half)
    bottom_vertex = (base_x, mid_y + tri_half)
    tip_vertex = (base_x + tri_horiz, mid_y)

    # 長方形區域（左側），從 pad 到 pad + rect_width，垂直與三角形對齊（可延伸至整體高度減 padding）
    rect_top = pad
    rect_bottom = H - pad
    rect = [(pad, rect_top), (pad + rect_width, rect_bottom)]

    # 建立透明底圖
    img = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(img)

    # 畫長方形（填滿灰色）
    draw.rectangle(rect, fill=triangle_color)

    # 畫等邊三角形（填滿灰色）
    triangle = [top_vertex, tip_vertex, bottom_vertex]
    # PIL 需要整數座標
    triangle_int = [(int(round(x)), int(round(y))) for (x,y) in triangle]
    draw.polygon(triangle_int, fill=triangle_color)

    # 儲存
    img.save(out_file)
    print(f"Saved: {os.path.abspath(out_file)} (size={W}x{H})")
    return img

if __name__ == "__main__":
    # 範例：512x256 圖片，左側長方形寬 80 px
    create_rect_plus_right_equilateral_triangle(
        size=(600, 512),
        padding=24,
        rect_width=80,
        triangle_color=(200,200,150,255),
        out_file="rect_triangle_R.png"
    )
