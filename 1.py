import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_chinese_font(size):
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, size)
            return font
        except:
            continue
    
    try:
        return ImageFont.load_default()
    except:
        return None


def add_watermark(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        print(f"无法读取图片: {input_path}")
        return None, None
    
    h, w = img.shape[:2]
    text = "AI生成"
    
    font_size = int(min(w, h) / 25)
    font = get_chinese_font(font_size)
    
    if font is None:
        print("无法加载中文字体，将使用OpenCV默认字体（可能无法显示中文）")
        return add_watermark_opencv(img, output_path)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img, 'RGBA')
    
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w = right - left
    text_h = bottom - top
    
    margin_x = int(w * 0.02)
    margin_y = int(h * 0.02)
    padding = int(font_size / 5)
    
    x = w - text_w - margin_x - padding
    y = h - text_h - margin_y - padding
    
    rect_x1 = x - padding
    rect_y1 = y - padding
    rect_x2 = x + text_w + padding
    rect_y2 = y + text_h + padding
    
    draw.rectangle(
        [(rect_x1, rect_y1), (rect_x2, rect_y2)],
        fill=(0, 0, 0, 128)
    )
    
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
    
    img_with_watermark = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, img_with_watermark)
    print(f"水印已添加并保存到: {output_path}")
    print(f"水印文字: {text}")
    print(f"水印位置: 左上角({rect_x1}, {rect_y1}), 右下角({rect_x2}, {rect_y2})")
    
    template_w = rect_x2 - rect_x1
    template_h = rect_y2 - rect_y1
    template_pil = Image.new('RGB', (template_w, template_h), (0, 0, 0))
    template_draw = ImageDraw.Draw(template_pil)
    template_draw.rectangle(
        [(0, 0), (template_w, template_h)],
        fill=(0, 0, 0)
    )
    template_draw.text((padding, padding), text, font=font, fill=(255, 255, 255))
    template = cv2.cvtColor(np.array(template_pil), cv2.COLOR_RGB2BGR)
    
    return img_with_watermark, template


def add_watermark_opencv(img, output_path):
    h, w = img.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "AI生成"
    
    font_scale = min(w, h) / 500
    thickness = int(font_scale * 2)
    
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    baseline += thickness
    
    margin_x = int(w * 0.02)
    margin_y = int(h * 0.02)
    
    x = w - text_w - margin_x
    y = h - margin_y
    
    overlay = img.copy()
    
    padding = int(font_scale * 5)
    cv2.rectangle(overlay, 
                  (x - padding, y - text_h - padding), 
                  (x + text_w + padding, y + padding),
                  (0, 0, 0), -1)
    
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    cv2.putText(img, text, (x, y), font, font_scale, 
                (255, 255, 255), thickness, cv2.LINE_AA)
    
    cv2.imwrite(output_path, img)
    print(f"水印已添加并保存到: {output_path}")
    
    watermark_template = np.zeros((text_h + 2 * padding, text_w + 2 * padding, 3), dtype=np.uint8)
    cv2.putText(watermark_template, text, (padding, text_h + padding), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    
    return img, watermark_template


def search_watermark(img, template):
    if img is None or template is None:
        print("图片或模板为空，无法搜索")
        return None
    
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    threshold = 0.5
    
    if max_val < threshold:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val < threshold:
            print(f"模板匹配未找到足够匹配的水印，最大匹配值: {max_val}")
            return search_watermark_by_color(img)
    
    h, w = template.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    print(f"找到水印位置: 左上角({top_left[0]}, {top_left[1]}), 右下角({bottom_right[0]}, {bottom_right[1]})")
    print(f"匹配度: {max_val:.2f}")
    
    return (top_left, bottom_right)


def search_watermark_by_color(img):
    print("尝试通过颜色特征搜索水印...")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    h, w = img.shape[:2]
    search_region = int(h * 0.5)
    mask[:h-search_region, :] = 0
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("未找到白色文字区域")
        return None
    
    all_contours = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        all_contours.append((x, y, cw, ch, area))
    
    if not all_contours:
        print("未找到符合条件的水印区域")
        return None
    
    all_contours.sort(key=lambda c: c[0])
    
    min_x = min(c[0] for c in all_contours)
    min_y = min(c[1] for c in all_contours)
    max_x = max(c[0] + c[2] for c in all_contours)
    max_y = max(c[1] + c[3] for c in all_contours)
    
    padding = 15
    top_left = (max(0, min_x - padding), max(0, min_y - padding))
    bottom_right = (min(w, max_x + padding), min(h, max_y + padding))
    
    print(f"通过颜色找到水印位置: 左上角({top_left[0]}, {top_left[1]}), 右下角({bottom_right[0]}, {bottom_right[1]})")
    
    return (top_left, bottom_right)


def crop_watermark(img, location, output_path):
    if img is None or location is None:
        print("图片或位置为空，无法裁剪")
        return None
    
    top_left, bottom_right = location
    
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    h, w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    watermark_crop = img[y1:y2, x1:x2]
    
    cv2.imwrite(output_path, watermark_crop)
    print(f"水印已裁剪并保存到: {output_path}")
    
    return watermark_crop


def main():
    input_image = "1.jpg"
    watermarked_image = "2.jpg"
    cropped_watermark = "3.jpg"
    
    print("=" * 50)
    print("步骤1: 添加水印到图片右下角")
    print("=" * 50)
    img_with_watermark, watermark_template = add_watermark(input_image, watermarked_image)
    
    if img_with_watermark is None:
        return
    
    print("\n" + "=" * 50)
    print("步骤2: 搜索水印位置（不使用已知位置信息）")
    print("=" * 50)
    location = search_watermark(img_with_watermark, watermark_template)
    
    if location is None:
        print("无法找到水印位置")
        return
    
    print("\n" + "=" * 50)
    print("步骤3: 裁剪并保存水印")
    print("=" * 50)
    crop_watermark(img_with_watermark, location, cropped_watermark)
    
    print("\n" + "=" * 50)
    print("所有操作完成!")
    print(f"1. 原图: {input_image}")
    print(f"2. 加水印后: {watermarked_image}")
    print(f"3. 裁剪的水印: {cropped_watermark}")
    print("=" * 50)


if __name__ == "__main__":
    main()
