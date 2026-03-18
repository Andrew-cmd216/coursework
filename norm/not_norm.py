import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os

# ────────────────────────────────────────────────
# 1. Собираем все файлы 001_resample.csv ... 005_resample.csv
file_list = [f"{i:03d}_resample.csv" for i in range(1, 6)]  # 001 до 005

# Проверка наличия файлов
missing = [f for f in file_list if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(f"Не найдены файлы: {missing}")

dfs = []
for file in file_list:
    df_part = pd.read_csv(file)
    print(f"Загружен {file}: {len(df_part)} строк")
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)
print(f"\nВсего объединено строк: {len(df)}\n")

# ────────────────────────────────────────────────
# 2. Подготовка диагнозов
df['diagnosis'] = df['diagnosis'].astype(str).str.strip().str.lower()
df['diagnosis'] = df['diagnosis'].replace({'nan': 'norm', '': 'norm'})

# Определяем группы
norm = df[df['diagnosis'] == 'norm'].copy()
patients = df[df['diagnosis'] != 'norm'].copy()

# Балансировка: все пациенты + столько же случайных норм
n_patients = len(patients)
norm_sampled = norm.sample(n=n_patients, random_state=42) if len(norm) > n_patients else norm.copy()

print(f"Группы после выравнивания:")
print(f"  Норма (взято случайно)   → {len(norm_sampled)} записей из {len(norm)}")
print(f"  Пациенты (все)           → {len(patients)} записей")
print(f"  Всего в анализе          → {len(norm_sampled) + len(patients)} строк\n")

# ────────────────────────────────────────────────
# 3. Признаки (все столбцы между filename и diagnosis)
features = df.columns[1:-1].tolist()

# ────────────────────────────────────────────────
def compare_norm_vs_all_patients(norm_group, patient_group):
    results = []

    for feat in features:
        n_vals = pd.to_numeric(norm_group[feat], errors='coerce').dropna()
        p_vals = pd.to_numeric(patient_group[feat], errors='coerce').dropna()

        if len(n_vals) < 3 or len(p_vals) < 3:
            continue

        # Тест на нормальность (Shapiro-Wilk)
        shapiro_norm = stats.shapiro(n_vals)[1]
        shapiro_pat  = stats.shapiro(p_vals)[1]
        normal = (shapiro_norm > 0.05) and (shapiro_pat > 0.05)

        if normal:
            stat, p = stats.ttest_ind(n_vals, p_vals, equal_var=False)
            test = "Welch t-test"
        else:
            stat, p = stats.mannwhitneyu(n_vals, p_vals, alternative='two-sided')
            test = "Mann-Whitney U"

        results.append({
            'признак': feat,
            'тест': test,
            'p_сырое': p,
            'ср_норма': n_vals.mean(),
            'ср_пациенты': p_vals.mean(),
            'std_норма': n_vals.std(),
            'std_пациенты': p_vals.std(),
            'n_норма': len(n_vals),
            'n_пациенты': len(p_vals)
        })

    if not results:
        print("Нет признаков с достаточным количеством данных")
        return

    res = pd.DataFrame(results)
    res['p_сырое'] = res['p_сырое'].round(6)

    # --- Поправка на множественные сравнения (BY) ---
    res['p_корр_BY'] = multipletests(res['p_сырое'], alpha=0.05, method='fdr_by')[1]
    res['значимо_BY'] = res['p_корр_BY'] < 0.05

    # --- Дополнительная строгость: разница средних > 0.2 ---
    res['разница_средних'] = abs(res['ср_норма'] - res['ср_пациенты'])
    res['строго_значимо'] = (res['значимо_BY']) & (res['разница_средних'] > 0.2)

    res = res.sort_values('p_корр_BY')

    # Сохраняем все статистически значимые (BY)

    # Сохраняем строго значимые (BY + разница > 0.2)
    strict = res[res['строго_значимо']]
    if not strict.empty:
        strict.to_csv("норма_vs_пациенты_BY_СТРОГО_ЗНАЧИМЫЕ.csv", index=False, encoding='utf-8-sig')
        print(f"Сохранено строго значимых (BY + разница > 0.2): норма_vs_пациенты_BY_СТРОГО_ЗНАЧИМЫЕ.csv ({len(strict)} шт.)")
        print("\nТОП-15 строго значимых признаков (по возрастанию p_corr_BY):")
        print(strict.head(15)[['признак', 'тест', 'p_сырое', 'p_корр_BY', 'разница_средних', 'ср_норма', 'ср_пациенты']].to_string(index=False))
    else:
        print("\nСтрого значимых признаков (BY + разница > 0.2) не найдено")

    return res

# ────────────────────────────────────────────────
# Запуск анализа
print("═" * 80)
compare_norm_vs_all_patients(norm_sampled, patients)
print("═" * 80)
