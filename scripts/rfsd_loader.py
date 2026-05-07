# import os
import time
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from huggingface_hub import HfFileSystem, hf_hub_download

def load_rfsd(
    year: int = 2022,
    num_rows: int = 2000,
    random_state: int = 42,
    data_dir: Path = None
) -> pd.DataFrame:
    
    start = time.time()
    
    # Настройка путей
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    output_csv = data_dir / "rfsd_sample.csv"
    
    # Колонки (24 шт)
    columns = [
        'inn', 'year', 'region', 'okved',
        'line_2110', 'line_2400', 'line_1600', 'line_1300',
        'line_1500', 'line_4100', 'age', 'eligible',
        'filed', 'outlier', 'line_1100', 'line_1200', 'line_1250', 'line_1410',
        'line_2120', 'line_2200', 'line_2300', 'line_4200', 'line_4300', 'line_4400'
    ]
    
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
        
        # проверка
        cols_to_rename = {k: v for k, v in rename_dict.items() if k in df.columns}
        for orig, new in list (cols_to_rename.items())[:5]:
            print(f"   {orig:20} → {new}")
            
        df = df.rename(columns=cols_to_rename)
        print("Колонки переименованы")
    except Exception as e:
        print(f"Словарь не загружен: {e}.")

    # Сохранение
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f" Сохранено в: {output_csv}")
    return df