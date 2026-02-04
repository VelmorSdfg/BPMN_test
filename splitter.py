import os
import random
import shutil


def split_dataset(source_folder, train_size=0.85):
    # Путь к новому датасету
    base_dir = os.path.join(os.path.dirname(source_folder), 'dataset')

    # Создаем структуру папок
    for split in ['train', 'val']:
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(base_dir, split, sub), exist_ok=True)

    # Собираем все базовые имена файлов (без расширений)
    files = [os.path.splitext(f)[0] for f in os.listdir(source_folder) if f.endswith('.png')]
    random.shuffle(files)  # Перемешиваем для честности

    split_idx = int(len(files) * train_size)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    def move_files(file_list, target_split):
        for name in file_list:
            # Копируем картинку
            shutil.copy2(
                os.path.join(source_folder, name + '.png'),
                os.path.join(base_dir, target_split, 'images', name + '.png')
            )
            # Копируем разметку
            shutil.copy2(
                os.path.join(source_folder, name + '.txt'),
                os.path.join(base_dir, target_split, 'labels', name + '.txt')
            )

    move_files(train_files, 'train')
    move_files(val_files, 'val')

    print(f"✅ Сплит завершен!")
    print(f"📈 Train: {len(train_files)} пар")
    print(f"📉 Val: {len(val_files)} пар")
    print(f"📂 Путь: {base_dir}")


# Твоя папка с результатами
src = r'C:\Users\VelmorSDFG\PycharmProjects\BPMN\uploads\raw\bpmn\02-Results'
split_dataset(src)