import cv2
import os
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Инициализация OCR
ocr_local = PaddleOCR(use_angle_cls=True, lang='ru', show_log=False)

MODEL_WEIGHTS_PATH = r'best.pt'
CONFIDENCE_THRESHOLD = 0.5
CLEAN_DIR = r'C:\Users\VelmorSDFG\PycharmProjects\BPMN\src\result'

try:
    model = YOLO(MODEL_WEIGHTS_PATH)
    print("Модель YOLO загружена")
except Exception as e:
    print(f"Ошибка загрузки YOLO: {e}")
    exit()


def simple_text_clean(text):
    if not text: return ""
    # Убираем странные одиночные символы, которые часто плодит OCR
    text = " ".join([w for w in text.split() if len(w) > 1 or w.isdigit()])
    return text.strip()


def predict_and_show(image_path):
    if not os.path.exists(image_path):
        return [], None

    img = cv2.imread(image_path)
    if img is None: return [], None

    clean_img = img.copy()
    debug_img = img.copy()
    h, w = img.shape[:2]

    new_h, new_w = int((h + 31) // 32 * 32), int((w + 31) // 32 * 32)
    results = model.predict(source=image_path, conf=CONFIDENCE_THRESHOLD, imgsz=(new_h, new_w), verbose=False)

    nodes_data = []
    p = 4  # Увеличенный отступ для OCR

    if results and results[0].boxes:
        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Расчет нового формата (Центр + Размеры)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            width = x2 - x1
            height = y2 - y1

            # OCR ВНУТРИ УЗЛА
            x1_p, y1_p = max(0, x1 - p), max(0, y1 - p)
            x2_p, y2_p = min(w - 1, x2 + p), min(h - 1, y2 + p)
            node_crop = img[y1_p:y2_p, x1_p:x2_p]

            node_text = ""
            if node_crop.size > 0:
                crop_res = cv2.resize(node_crop, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                ocr_res = ocr_local.ocr(crop_res, cls=True)
                if ocr_res and ocr_res[0]:
                    node_text = " ".join([line[1][0] for line in ocr_res[0]])

            node_text = simple_text_clean(node_text)

            nodes_data.append({
                "id": f"n{i}",
                "type": label,
                "txt": node_text,
                "cnt": [center_x, center_y],
                "wh": [width, height]
            })

            # Отрисовка дебага
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(clean_img, (x1_p, y1_p), (x2_p, y2_p), (255, 255, 255), -1)

    os.makedirs(CLEAN_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(CLEAN_DIR, "0_debug.png"), debug_img)
    cv2.imwrite(os.path.join(CLEAN_DIR, "1_nodes_removed.png"), clean_img)

    return nodes_data, clean_img