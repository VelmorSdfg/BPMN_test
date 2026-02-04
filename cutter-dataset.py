import cv2
import numpy as np
from paddleocr import PaddleOCR
import os
import glob
import random

# Инициализация PaddleOCR (lang='cyrillic' для твоей версии)
ocr = PaddleOCR(use_angle_cls=True, lang='cyrillic', show_log=False)


def test_clean_single_dataset_item(target_folder):
    # 1. Ищем все изображения в папке bpmn\test
    extensions = ('*.png', '*.jpg', '*.jpeg')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(target_folder, ext)))

    if not files:
        print(f"❌ В папке {target_folder} пусто!")
        return

    # 2. Выбираем один случайный файл для теста
    image_path = random.choice(files)
    print(f"🔄 Тестовая очистка файла: {os.path.basename(image_path)}")

    img = cv2.imread(image_path)
    if img is None: return

    # 3. Увеличение x2 для точности детекции
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    # 4. OCR детекция
    result = ocr.ocr(upscaled, cls=True)

    clean_img = img.copy()

    if result and result[0]:
        for line in result[0]:
            # Возвращаем координаты к оригиналу (деление на 2)
            box = np.array(line[0], dtype=np.float32)
            box_orig = (box / 2).astype(np.int32)

            # Определяем границы рамки
            x_min, y_min = np.min(box_orig, axis=0)
            x_max, y_max = np.max(box_orig, axis=0)

            # Берем пробу цвета фона (чуть левее и выше текста)
            sample_x = max(0, int(x_min) - 2)
            sample_y = max(0, int(y_min) - 2)
            bg_color = [int(c) for c in img[sample_y, sample_x]]

            # --- УЛУЧШЕНИЕ: Закраска с запасом (Padding) ---
            # Добавляем +2 пикселя к каждой стороне, чтобы убрать ореолы букв
            p = 2
            cv2.rectangle(clean_img,
                          (max(0, x_min - p), max(0, y_min - p)),
                          (min(w, x_max + p), min(h, y_max + p)),
                          bg_color, -1)

    # 5. Сохранение результата в ту же папку
    output_path = os.path.join(target_folder, "TEST_RESULT_CLEANED.png")
    cv2.imwrite(output_path, clean_img)

    print(f"✅ Готово! Результат тут: {output_path}")


if __name__ == "__main__":
    # Твой путь к папке теста
    DATASET_PATH = r'C:\Users\VelmorSDFG\PycharmProjects\BPMN\uploads\raw\bpmn\test'
    test_clean_single_dataset_item(DATASET_PATH)