import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os

file_list = [
    '001_resample.csv',
    '002_resample.csv',
    '003_resample.csv',
    '004_resample.csv',
    '005_resample.csv'
]

dfs = []
for fname in file_list:
    if os.path.exists(fname):
        try:
            df_temp = pd.read_csv(fname)
            dfs.append(df_temp)
            print(f"Прочитан: {fname} → {len(df_temp)} строк")
        except Exception as e:
            print(f"Ошибка чтения {fname}: {e}")
    else:
        print(f"Файл не найден, пропускаем: {fname}")

if not dfs:
    print("Ни один файл не найден. Завершение.")
    # exit()  # или raise Exception — по желанию
else:
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nОбъединено строк: {len(df)} из {len(dfs)} файлов\n")

df['diagnosis'] = df['diagnosis'].fillna('norm').astype(str).str.strip().str.lower()

bipolar = df[df['diagnosis'].str.contains('bipolar', na=False)]

schizo_mask = (
    df['diagnosis'].str.contains('schizo', na=False) |
    df['diagnosis'].str.contains('schizoaffective', na=False) |
    df['diagnosis'].str.contains('schizotypal', na=False)
)
schizo = df[schizo_mask]

norm = df[df['diagnosis'] == 'norm']

print(f"Группы после объединения:")
print(f"  Биполярное       → {len(bipolar):4d} записей")
print(f"  Шизофрения-спектр→ {len(schizo):4d} записей")
print(f"  Норма            → {len(norm):4d} записей\n")

features = df.columns[1:-1].tolist()

def compare_and_save(group_a, group_b, name_a, name_b):
    if len(group_a) < 3 or len(group_b) < 3:
        print(f"Недостаточно данных в одной из групп: {name_a} ({len(group_a)}) vs {name_b} ({len(group_b)})")
        return None

    results = []
    for feat in features:
        a = group_a[feat].dropna().astype(float)
        b = group_b[feat].dropna().astype(float)
        
        if len(a) < 3 or len(b) < 3:
            continue
        
        _, p_a = stats.shapiro(a)
        _, p_b = stats.shapiro(b)
        normal = (p_a > 0.05) and (p_b > 0.05)
        
        if normal:
            _, p = stats.ttest_ind(a, b, equal_var=False)
            test_name = "Welch t-test"
        else:
            _, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            test_name = "Mann-Whitney U"
        
        results.append({
            'признак': feat,
            'тест': test_name,
            'p_сырое': round(p, 6),
            f'ср_{name_a}': round(a.mean(), 3),
            f'ср_{name_b}': round(b.mean(), 3),
            f'n_{name_a}': len(a),
            f'n_{name_b}': len(b)
        })
    
    if not results:
        print(f"Нет валидных признаков для сравнения {name_a} vs {name_b}")
        return None
    
    res = pd.DataFrame(results)
        
    res['p_корр_BY'] = multipletests(res['p_сырое'], alpha=0.05, method='fdr_by')[1]
    res['значимо_BY'] = res['p_корр_BY'] < 0.05

    res['разница_средних'] = abs(res[f'ср_{name_a}'] - res[f'ср_{name_b}'])
    strict = res[(res['значимо_BY']) & (res['разница_средних'] > 0.2)]

    if not strict.empty:
        fname_strict = f"статистика_{name_a}_vs_{name_b}_строго.csv"
        strict.to_csv(fname_strict, index=False, encoding='utf-8-sig')
        print(f"Сохранено СТРОГИХ: {fname_strict} ({len(strict)} признаков)")
    


    sig = res[res['значимо_BY']]
    if not sig.empty:
        fname_sig = f"статистика_{name_a}_vs_{name_b}_значимые.csv"
        sig.to_csv(fname_sig, index=False, encoding='utf-8-sig')
        print(f"Сохранено ЗНАЧИМЫХ: {fname_sig} ({len(sig)} признаков)")
        print("\nЗначимые признаки после BY (p_corr < 0.05):")
        print(sig[['признак', 'тест', 'p_сырое', 'p_корр_BY', f'ср_{name_a}', f'ср_{name_b}']].to_string(index=False))
    else:
        print(f"После BY значимых различий между {name_a} и {name_b} НЕ найдено")
    
    return res



compare_and_save(bipolar, norm, "биполярное", "норма_BY")
compare_and_save(schizo, norm, "шизофрения_спектр", "норма_BY")

print("\nГотово.")