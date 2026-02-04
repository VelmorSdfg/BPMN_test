import cv2
import os

# Список классов должен СТРОГО совпадать с твоим основным скриптом
CLASSES = ['Task', 'Gateway', 'StartEvent', 'EndEvent', 'IntermediateEvent']
# Цвета для классов (BGR): Задачи - синий, Шлюзы - зеленый, События - красный/желтый
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 165, 255), (255, 0, 255)]


def draw_yolo_labels(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(folder_path):
        if not file.endswith('.png'):
            continue

        base_name = os.path.splitext(file)[0]
        txt_path = os.path.join(folder_path, base_name + '.txt')
        img_path = os.path.join(folder_path, file)

        if not os.path.exists(txt_path):
            continue

        # Читаем картинку
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5: continue

            cls_id = int(parts[0])
            # YOLO format: cx, cy, nw, nh (normalized)
            cx, cy, nw, nh = map(float, parts[1:])

            # Пересчитываем в пиксели
            x1 = int((cx - nw / 2) * w)
            y1 = int((cy - nh / 2) * h)
            x2 = int((cx + nw / 2) * w)
            y2 = int((cy + nh / 2) * h)

            # Рисуем рамку
            color = COLORS[cls_id] if cls_id < len(COLORS) else (0, 255, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Пишем текст
            label = f"{CLASSES[cls_id]}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Сохраняем результат
        save_path = os.path.join(output_folder, f"check_{file}")
        cv2.imwrite(save_path, img)
        print(f"📸 Проверка готова: {save_path}")


if __name__ == "__main__":
    # Укажи путь к папке, где лежат твои сгенерированные .png и .txt
    input_dir = r'C:\Users\VelmorSDFG\PycharmProjects\BPMN\uploads\raw\bpmn\02-Results'
    output_dir = os.path.join(input_dir, 'debug_view')

    draw_yolo_labels(input_dir, output_dir)