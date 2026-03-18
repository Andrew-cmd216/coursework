import pandas as pd
import glob

# Находим все файлы вида *_resample.csv в текущей папке
csv_files = glob.glob("*_resample.csv")

print(f"Найдено файлов: {len(csv_files)}")
print("Список файлов:")
for f in sorted(csv_files):
    print("  ", f)

if not csv_files:
    print("Файлы не найдены.")
    exit()

dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file, low_memory=False)
        dfs.append(df)
    except Exception as e:
        print(f"Ошибка при чтении {file}: {e}")

if not dfs:
    print("Не удалось прочитать ни один файл.")
    exit()

data = pd.concat(dfs, ignore_index=True)
print(f"\nВсего записей (строк): {len(data):,}")

if 'diagnosis' in data.columns:
    data['diagnosis'] = data['diagnosis'].astype(str).str.strip().str.lower()

    is_norm = data['diagnosis'].str.startswith('pn-') | (data['diagnosis'] == 'nan')
    is_bipolar = data['diagnosis'].str.contains('bipolar', na=False)
    is_schizo_spectrum = data['diagnosis'].str.contains(
        'schizo|schizophrenia|schizoaffective|schizotypal', na=False)

    total = len(data)
    norm = is_norm.sum()
    bipolar = is_bipolar.sum()
    schizo = is_schizo_spectrum.sum()
    remaining = total - norm - bipolar - schizo

    print("\nРаспределение:")
    print(f"  Норма → {norm:4d}  ({norm/total:6.1%})")
    print(f"  Биполярное расстройство → {bipolar:4d}  ({bipolar/total:6.1%})")
    print(f"  Шизо-спектр → {schizo:4d}  ({schizo/total:6.1%})")
    print(f"  Остальные диагнозы → {remaining:4d}  ({remaining/total:6.1%})")

    print("\nТоп-10 самых частых меток:")
    print(data['diagnosis'].value_counts().head(10))
else:
    print("Столбец 'diagnosis' не найден.")