from ultralytics import YOLO
import torch

def train_bpmn_model():
    # 1. Проверка доступности GPU (видеокарты NVIDIA)
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Обучение будет запущено на: {device}")

    # 2. Загружаем предобученную модель (начальные веса)
    model = YOLO('yolov8n.pt')

    # 3. Запуск обучения
    model.train(
        data='data.yaml',
        epochs=150,
        imgsz=1024,  # Оставляем высокое качество
        batch=16,  # На 4060 должно летать
        device=device,
        project='BPMN_Project',
        name='v8_bpmn_v1',
        patience=30,
        optimizer='AdamW',
        augment=True,
        rect=True,  # Оставляем для поддержки разного Aspect Ratio
        multi_scale=False,  # ВЫКЛЮЧАЕМ (это решит проблему ZeroDivisionError)
        workers=0
    )

    print("✅ Обучение завершено!")

if __name__ == "__main__":
    train_bpmn_model()