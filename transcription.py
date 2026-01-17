import os
from tqdm import tqdm
import gigaam


folders = [
    '001_resample',
    '002_resample',
    '003_resample',
    '004_resample',
    '005_resample'
]

os.environ["HF_TOKEN"] = "токен"

print("Загружаем модель GigaAM")
model = gigaam.load_model("v3_e2e_rnnt")

total_files = 0
processed = 0

for folder in folders:
    if not os.path.exists(folder):
        print(f"Папка не найдена: {folder} — пропускаем")
        continue

    wav_files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]

    if not wav_files:
        print(f"В папке {folder} нет .wav файлов — пропускаем")
        continue

    total_files += len(wav_files)
    print(f"\nОбрабатываем папку {folder}: {len(wav_files)} файлов")

    txt_dir = folder.replace('_resample', '_transcriptions_v3')
    os.makedirs(txt_dir, exist_ok=True)

    for filename in tqdm(wav_files, desc=f"Транскрипция {folder}"):
        audio_path = os.path.join(folder, filename)
        txt_path = os.path.join(txt_dir, filename.replace('.wav', '.txt'))

        try:
            result = model.transcribe_longform(audio_path)

            with open(txt_path, 'w', encoding='utf-8') as f:
                for utterance in result:
                    start = gigaam.format_time(utterance['boundaries'][0])
                    end = gigaam.format_time(utterance['boundaries'][1])
                    text = utterance['transcription']
                    f.write(f"[{start} - {end}]: {text}\n")

            processed += 1

        except Exception as e:
            print(f"\nОшибка при обработке {audio_path}: {e}")

print(f"\nГотово! Успешно обработано файлов: {processed} из {total_files}")
print("Транскрипции сохранены в папках:")
for folder in folders:
    print(f"  → {folder.replace('_resample', '_transcriptions_v3')}")