from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from rfsd_loader import load_rfsd
from preprocess import preprocess_pipeline, clean_df_save
from eda import run_eda

def main():
    # сырые данные
    df_raw = load_rfsd(
        year=2022,
        num_rows=2000,
        random_state=42
    )
    
    # 1-ый запуск eda
    run_eda(df_raw)
    
    # очистка df_raw
    df_clean = preprocess_pipeline(df_raw)
    
    # 2-ой запуск eda
    run_eda(df_clean)
    
    # сохранение df_clean
    output_path = clean_df_save(df_clean, "rfsd_sample_clean.csv")
    
    base_path = Path(__file__).parent
    print(f"Сырые данные: {base_path / 'scripts' / 'data' / 'rfsd_sample.csv'}")
    print(f'Очищенные данные: {output_path}')
    print(f'Итоговый размер: {df_clean.shape[0]} строк × {df_clean.shape[1]} колонок')
    
if __name__ == "__main__":
    main()
    