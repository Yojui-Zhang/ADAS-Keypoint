#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

def find_font(preferred_sizes):
    """嘗試載入系統常見字型（回傳 ImageFont），無法找到則回傳預設字型"""
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
    return ImageFont.load_default()

def create_triangle_with_white_exclamation(size=512,
                                           corner_radius=24,
                                           triangle_color=(255,200,0,255),
                                           out_file="triangle_white_exclaim.png"):
    # 1) 透明底
    img = Image.new("RGBA", (size, size), (0,0,0,0))

    # 2) 建三角形蒙版 (L)
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    triangle = [(size//2, int(size*0.08)),
                (int(size*0.08), int(size*0.9)),
                (int(size*0.92), int(size*0.9))]
    md.polygon(triangle, fill=255)

    # 3) 用 GaussianBlur + 閾值化 做圓角（越大 corner_radius 越圓）
    if corner_radius > 0:
        blurred = mask.filter(ImageFilter.GaussianBlur(radius=corner_radius))
        mask = blurred.point(lambda p: 255 if p > 128 else 0)

    # 4) 填灰色三角形
    color_layer = Image.new("RGBA", (size, size), triangle_color)
    img = Image.composite(color_layer, img, mask)

    # 5) 建白色驚嘆號圖層（RGBA），再以三角形 mask 裁切（確保白色只出現在三角形內）
    text_layer = Image.new("RGBA", (size, size), (0,0,0,0))
    td = ImageDraw.Draw(text_layer)

    # 選字型大小（嘗試幾個大小）
    preferred_sizes = [int(size*0.65), int(size*0.6), int(size*0.55), int(size*0.5)]
    font = find_font(preferred_sizes)

    text = "!"
    bbox = td.textbbox((0,0), text, font=font)   # (x0,y0,x1,y1)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pos = ((size - tw) // 2 - bbox[0], (size - th) // 2 - bbox[1] + 30)

    # 實心白色驚嘆號
    td.text(pos, text, fill=(0,0,0,255), font=font)

    # 將白色文字限制在三角形區域內
    masked_text = Image.composite(text_layer, Image.new("RGBA", (size,size), (0,0,0,0)), mask)

    # 6) 合成到基底
    out = Image.alpha_composite(img, masked_text)

    # 儲存
    out.save(out_file)
    print(f"Saved: {os.path.abspath(out_file)}")

if __name__ == "__main__":
    create_triangle_with_white_exclamation()
