import os
import pandas as pd
from tqdm import tqdm


folders = [
    '001_transcriptions_v3',
    '002_transcriptions_v3',
    '003_transcriptions_v3',
    '004_transcriptions_v3',
    '005_transcriptions_v3'
]

data = []

all_txt_files = []
for folder in folders:
    folder_path = folder
    if not os.path.isdir(folder_path):
        print(f"Предупреждение: папка не найдена — {folder_path}")
        continue

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            all_txt_files.append(file_path)

print(f"Найдено {len(all_txt_files)} .txt файлов. Начинаем обработку...\n")

for file_path in tqdm(all_txt_files, desc="Обработка файлов", unit="файл"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        data.append({'filename': file_path, 'content': content})
    except Exception as e:
        print(f"\nОшибка при чтении {file_path}: {e}")

if data:
    df = pd.DataFrame(data)
    output_csv = 'all_transcriptions.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n\nГотово! Создан файл: {output_csv}")
    print(f"Всего успешно обработано файлов: {len(df)}")
else:
    print("\nНе удалось прочитать ни одного файла.")
