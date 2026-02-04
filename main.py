import os
import json
import cv2
import numpy as np
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(PROJECT_ROOT, 'src', 'result')
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

SOURCE_IMAGE_DEFAULT = r'C:\Users\VelmorSDFG\PycharmProjects\BPMN\uploads\34.png'

from src.test_model import predict_and_show
from src.cutter import clean_diagram_v3
from src.slip_arrows import detect_orthogonal_arrows


def run_smart_pipeline(source_image_path):
    os.makedirs(RESULT_DIR, exist_ok=True)
    final_data = {"source_file": os.path.basename(source_image_path), "nodes": [], "labels": [], "arrows": []}

    # ЭТАП 1: детекция узлов
    print("1. YOLO + Внутренний OCR")
    nodes, img_nodes_removed = predict_and_show(source_image_path)

    # ЭТАП 2: внешний текст
    print("2. OCR Внешнего текста")
    external_labels, img_fully_cleaned = clean_diagram_v3(img_nodes_removed, output_dir=RESULT_DIR)

    final_data["nodes"] = nodes
    final_data["labels"] = external_labels

    # ЭТАП 3: поиск стрелок
    print("3. Поиск стрелок")
    cv2.imwrite(os.path.join(RESULT_DIR, "final_cleaned_for_arrows.png"), img_fully_cleaned)

    arrows = detect_orthogonal_arrows(img_fully_cleaned, output_dir=RESULT_DIR)
    final_data["arrows"] = arrows

    # Итоговый Json
    json_path = os.path.join(RESULT_DIR, "analysis_result.json")

    def conv(obj):
        return int(obj) if isinstance(obj, np.integer) else (obj.tolist() if isinstance(obj, np.ndarray) else str(obj))

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4, default=conv)

    print(f"\n Результаты: {RESULT_DIR}")


if __name__ == "__main__":
    run_smart_pipeline(SOURCE_IMAGE_DEFAULT)