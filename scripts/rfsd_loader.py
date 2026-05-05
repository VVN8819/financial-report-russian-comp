import os
import time
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from huggingface_hub import HfFileSystem, hf_hub_download

# Настройки проекта
base_dir = Path(__file__).parent.resolve()
data_dir = base_dir / "data"
data_dir.mkdir(exist_ok=True)  # Создаём папку, если нет

output_csv = data_dir / "rfsd_sample.csv"
year = 2022
num_rows = 4000
random_state = 42

# Колонки (24 шт)
columns = [
    'inn', 'year', 'region', 'okved',
    'line_2110', 'line_2400', 'line_1600', 'line_1300',
    'line_1500', 'line_4100', 'age', 'eligible',
    'filed', 'outlier', 'line_1100', 'line_1200', 'line_1250', 'line_1410',
    'line_2120', 'line_2200', 'line_2300', 'line_4200', 'line_4300', 'line_4400'
]

def load_rfsd():
    start = time.time()

    # Поиск файлов за нужный год
    fs = HfFileSystem()
    files = fs.glob(f"datasets/irlspbru/RFSD/RFSD/year={year}/*.parquet")
    if not files:
        raise ValueError(f" Не найдено файлов за {year} год.")
    print(f" Найдено файлов: {len(files)}")

    # Загрузка и чтение
    tables = []
    for file in files:
        local_path = hf_hub_download(
            repo_id="irlspbru/RFSD",
            filename=file.replace("datasets/irlspbru/RFSD/", ""),
            repo_type="dataset"
        )
        table = pq.read_table(local_path, columns=columns)
        tables.append(table)
        if sum(t.num_rows for t in tables) >= num_rows * 2:
            break

    # Объединение
    combined = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    df = combined.to_pandas().sample(n=num_rows, random_state=random_state).reset_index(drop=True)
    print(f"Данные загружены за {time.time() - start:.2f} сек | Строк: {len(df)}")

    # Переименование колонок
    try:
        renaming_df = pd.read_csv(
            'https://raw.githubusercontent.com/irlcode/RFSD/main/aux_data/descriptive_names_dict.csv'
        )
        rename_dict = dict(zip(renaming_df['original'], renaming_df['descriptive']))
        df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
        print("Колонки переименованы")
    except Exception as e:
        print(f"Словарь не загружен: {e}.")

    # Сохранение
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f" Сохранено в: {output_csv}")
    return df

if __name__ == "__main__":
    df = load_rfsd()
    print("\nПервые 3 строки:")
    print(df.head(3))

# ======================== Исследовательский анализ EDA ============================
df.info(memory_usage='deep')

# выявим пропущенные значения с помощью .isnull() и посчитаем их количество через mean()
print(df.isnull().mean())

# ======================== Очистка df ==============================================
# переменные B_longterm_debt, B_noncurrent_assets, CF_balance_operating, CF_balance_invest,
# CF_balance_fin, CF_balance , скорее всего, не являются самыми важными
cols_to_drop = [
    'CF_balance_operating',
    'CF_balance_invest',
    'CF_balance_fin',
    'CF_balance',
    'B_longterm_debt',
    'B_noncurrent_assets'
]

df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Оставляем только компании с важными колонками для анализа
imp_cols = [
    'PL_revenue', 'B_assets', 'B_total_equity', 'PL_net_profit',
    'PL_cost_of_sales', 'PL_profit_from_sales', 'PL_before_tax'
]

# Фильтруем строки, где есть хотя бы 3 из 7 ключевых показателей
df_imp = df[df[imp_cols].notna().sum(axis=1) >= 3].copy()

print(f'Осталось {len(df_imp)} из {len(df)} компаний ({len(df_imp)/len(df)*100:.1f}%)')
print(f'Пропуски после фильтрации:\n{df_imp.isnull().mean()}')

# проверим на дубли
print(f'Найдено дублей: {df_imp.duplicated(subset=['inn', 'year']).sum()} шт')

# ====================== Заполним медианой остатки =====================
# Медиана для пустых значений
empt_cols = [
    'PL_revenue', 'PL_net_profit', 'B_total_equity', 'B_shortterm_liab', 'B_current_assets',
    'PL_cost_of_sales', 'PL_profit_from_sales', 'B_cash_equivalents', 'PL_before_tax'
]
for col in empt_cols:
    if col in df_imp.columns and df_imp[col].isnull().any():
        median_val = df_imp[col].median()
        df_imp[col] = df_imp[col].fillna(median_val)
        print(f'{col}: заполнена медианой={median_val}')

# ====================== Сохранение очищенного csv файла ===================
# проверим пропуски очищенного df
print(df_imp.isnull().mean())

# сохраним полученный df в csv файл
base_dir = Path(__file__).parent.resolve()
data_dir = base_dir / "data"
data_dir.mkdir(exist_ok=True)  # Создаём папку, если нет

output_clean_csv = data_dir / "rfsd_sample_clean.csv"

df_imp.to_csv(output_clean_csv, index=False, encoding='utf-8-sig')
print(f'Сохранено в: {output_clean_csv}')