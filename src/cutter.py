import cv2
import numpy as np
from paddleocr import PaddleOCR
import os

# Инициализация OCR
ocr_ext = PaddleOCR(use_angle_cls=True, lang='ru', show_log=False, det_db_score_mode='fast', det_db_box_thresh=0.4)


def fix_leaked_letters(text):
    """Исправляет буквы, которые OCR ошибочно подставил вместо цифр."""
    replacements = {
        'O': '0', 'o': '0', 'О': '0', 'о': '0',
        'I': '1', 'i': '1', 'l': '1', 'L': '1',
        'B': '8', 'В': '8', 'S': '5', 's': '5',
        'G': '6', 'T': '7', 'Z': '2', 'z': '2'
    }
    has_digits = any(char.isdigit() for char in text)
    is_short = len(text) <= 3

    if has_digits or is_short:
        res = list(text)
        for i, char in enumerate(res):
            if char in replacements:
                res[i] = replacements[char]
        return "".join(res)
    return text


def merge_labels(labels, x_threshold=50, y_threshold=35):
    """Объединяет блоки текста и возвращает их в формате cnt/wh."""
    if not labels: return []
    # Сначала фильтруем вложенные (логика остается той же, работаем с временным bbox)
    labels.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))

    merged_raw = []
    while labels:
        curr = labels.pop(0)
        i = 0
        while i < len(labels):
            other = labels[i]
            c_x1, c_y1, c_x2, c_y2 = curr['bbox']
            o_x1, o_y1, o_x2, o_y2 = other['bbox']

            is_same_line = abs(c_y1 - o_y1) < 12
            horizontal_join = is_same_line and -10 < (o_x1 - c_x2) < x_threshold

            y_dist = o_y1 - c_y2
            overlap_width = min(c_x2, o_x2) - max(c_x1, o_x1)
            vertical_join = (overlap_width > min(c_x2 - c_x1, o_x2 - o_x1) * 0.5) and 0 <= y_dist < y_threshold

            if horizontal_join or vertical_join:
                curr['text'] = fix_leaked_letters(f"{curr['text']} {other['text']}")
                curr['bbox'] = [min(c_x1, o_x1), min(c_y1, o_y1), max(c_x2, o_x2), max(c_y2, o_y2)]
                labels.pop(i)
                i = 0
            else:
                i += 1
        merged_raw.append(curr)

    # ПРЕОБРАЗОВАНИЕ В НОВЫЙ ФОРМАТ
    final_compact = []
    for item in merged_raw:
        x1, y1, x2, y2 = item['bbox']
        final_compact.append({
            "txt": item['text'].strip(),
            "cnt": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            "wh": [int(x2 - x1), int(y2 - y1)]
        })
    return final_compact


def clean_diagram_v3(img_input, output_dir=None):
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    else:
        img = img_input.copy()
    if img is None: return [], None
    h, w = img.shape[:2]

    scale_factor = 2
    upscaled = cv2.resize(img, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LANCZOS4)
    result = ocr_ext.ocr(upscaled, cls=True)

    mask = np.zeros((h, w), dtype=np.uint8)
    raw_labels = []

    if result and result[0]:
        for line in result[0]:
            text_content = fix_leaked_letters(line[1][0])
            score = line[1][1]
            if len(text_content) < 1: continue

            poly_points = np.array(line[0], dtype=np.float32)
            poly_orig = (poly_points / scale_factor).astype(np.int32)
            x1, y1 = np.min(poly_orig, axis=0)
            x2, y2 = np.max(poly_orig, axis=0)

            raw_labels.append({
                "text": text_content,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": score
            })
            cv2.fillPoly(mask, [poly_orig], 255)

    # Объединяем и переводим в формат cnt/wh
    final_labels = merge_labels(raw_labels)

    # Очистка изображения (удаление текста)
    clean_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    clean_img[mask > 0] = (255, 255, 255)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, '2_text_removed.png'), clean_img)

    return final_labels, clean_img