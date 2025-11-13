#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import os

def find_font(preferred_sizes, text, prefer_bold=True):
    """嘗試載入常見系統字型，回傳 ImageFont 或 None。"""
    candidates = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
        "FreeSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for size in preferred_sizes:
        for c in candidates:
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return None

def create_triangle_with_exclamation(size=512,
                                     corner_radius=24,
                                     triangle_color=(128,128,128,255),
                                     out_file="triangle_rgba.png"):
    # 1) 建透明底圖
    img = Image.new("RGBA", (size, size), (0,0,0,0))

    # 2) 建三角形蒙版 (L)，先畫乾淨的尖角三角形
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    triangle = [(size//2, int(size*0.08)), (int(size*0.08), int(size*0.9)), (int(size*0.92), int(size*0.9))]
    md.polygon(triangle, fill=255)

    # 3) 以 GaussianBlur 模糊蒙版再閾值化 => 圓滑的圓角效果
    if corner_radius > 0:
        blurred = mask.filter(ImageFilter.GaussianBlur(radius=corner_radius))
        # 閾值化，保持主要形狀但角被圓滑
        mask = blurred.point(lambda p: 255 if p > 128 else 0)

    # 4) 在另個圖層上填滿灰色，然後用 mask 合成到透明底圖
    color_layer = Image.new("RGBA", (size, size), triangle_color)
    img = Image.composite(color_layer, img, mask)

    # 5) 建驚嘆號遮罩 (白=255 其餘=0)，以便從 alpha 減去（挖空）
    ex_mask = Image.new("L", (size, size), 0)
    ed = ImageDraw.Draw(ex_mask)

    # 選字型（嘗試多個大小與字體）
    # 期望驚嘆號約佔圖形的大面積
    preferred_sizes = [int(size*0.65), int(size*0.6), int(size*0.5), int(size*0.4), int(size*0.3)]
    font = find_font(preferred_sizes, "!")
    if font is None:
        # fallback to default (會比較小)
        font = ImageFont.load_default()

    # 計算文字大小並置中
    text = "!"
    bbox = ed.textbbox((0,0), text, font=font)  # (x0,y0,x1,y1)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pos = ((size - tw) // 2 - bbox[0], (size - th) // 2 - bbox[1] + 30)
    ed.text(pos, text, fill=255, font=font)

    # optional: 調整驚嘆號粗細（若使用矢量字型可透過描邊實作，但這裡用簡單法：膨脹一次模糊+閾值）
    ex_mask = ex_mask.filter(ImageFilter.GaussianBlur(radius=max(1, size//120)))
    ex_mask = ex_mask.point(lambda p: 255 if p > 128 else 0)
    
    # 6) 把驚嘆號挖空：取出當前 alpha，然後用 ImageChops.subtract(alpha, ex_mask)
    alpha = img.getchannel("A")
    new_alpha = ImageChops.subtract(alpha, ex_mask)
    img.putalpha(new_alpha)
    
    # 7) 儲存
    img.save(out_file)
    print(f"Saved: {os.path.abspath(out_file)}")

if __name__ == "__main__":
    # 範例：512x512，圓角 24 px
    create_triangle_with_exclamation(size=512, corner_radius=24, out_file="triangle_rgba.png")
