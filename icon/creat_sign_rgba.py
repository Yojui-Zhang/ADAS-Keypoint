from PIL import Image, ImageDraw, ImageFont

def make_ring_number_png(
    text="110",
    out_path="110km.png",
    img_size=(512, 512),
    ring_color_red=(255, 0, 0, 255),   # 紅色（含透明度）
    ring_color_white=(255, 255, 255, 255),   # 紅色（含透明度）
    ring_margin=24,                # 圓圈離邊界的內縮距離
    ring_width=60,                 # 圓圈線寬
    font_path=None,                # 指定字型路徑（建議 .ttf）
    font_width = 10,
    font_size=180,   		    # 字體大小
    text_color=(0, 0, 0, 255)      # 黑色（含透明度）
):
    # 1) 透明背景 RGBA
    w, h = img_size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 2) 畫中空紅色圓圈
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 2 - ring_margin
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.ellipse(bbox, fill=ring_color_white)
    draw.ellipse(bbox, outline=ring_color_red, width=ring_width)

    # 3) 準備字型（若未指定，嘗試常見字型，最後退回預設字型）
    if font_path is None:
        tried = False
        for fp in ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf"]:
            try:
                # 先用臨時大小載入，稍後會依圈內可用空間調整
                font = ImageFont.truetype(fp, size=font_size)
                font_path = fp
                tried = True
                break
            except Exception:
                continue
        if not tried:
            # 找不到可調整大小的 TTF，就用內建字型（尺寸不可調）
            font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, size=font_size)

    # 5) 置中文字
    tb = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
    text_x = cx - text_w / 2
    text_y = cy - text_h / 2 - 40
    draw.text((text_x, text_y), text, font=font, fill=text_color, stroke_width=font_width, stroke_fill=text_color)

    # 6) 輸出 PNG
    img.save(out_path, format="PNG")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    make_ring_number_png()

