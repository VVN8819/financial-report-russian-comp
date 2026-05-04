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
year = 2023
num_rows = 2000
random_state = 42

# Колонки (20 шт)
columns = [
    'inn', 'year', 'region', 'okved',
    'line_2110', 'line_2400', 'line_1600', 'line_1300',
    'line_1500', 'line_4100', 'age', 'eligible',
    'filed', 'outlier', 'line_1250', 'line_1410',
    'line_2120', 'line_2200', 'line_2300', 'line_4400'
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

    # Сохранение
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f" Сохранено в: {output_csv}")
    return df

if __name__ == "__main__":
    df = load_rfsd()
    print("\nПервые 3 строки:")
    print(df.head(3))