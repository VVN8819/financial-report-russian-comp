from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from rfsd_loader import load_rfsd
from preprocess import preprocess_pipeline, clean_df_save
from eda import run_eda
from visualization import cols_for_histograms

def main():
    # сырые данные
    df_raw = load_rfsd(
        year=2022,
        num_rows=2000,
        random_state=42
    )
    
    base_path = Path(__file__).parent
    scripts_path = base_path / "scripts"
    reports_path = scripts_path / "reports"  # Папка для графиков
    reports_path.mkdir(exist_ok=True)  # Создаём, если нет
    
    # 1-ый запуск eda
    run_eda(df_raw)
    
    # очистка df_raw
    df_clean = preprocess_pipeline(df_raw)
    
    # 2-ой запуск eda
    run_eda(df_clean)
    
    # сохранение df_clean
    output_path = clean_df_save(df_clean, "rfsd_sample_clean.csv")
    
    # Гистограммы для 3 выбранных колонок
    cols_for_histograms(df_clean, save_dir=reports_path)
    
    print(f"Сырые данные: {base_path / 'scripts' / 'data' / 'rfsd_sample.csv'}")
    print(f'Очищенные данные: {output_path}')
    print(f'Итоговый размер: {df_clean.shape[0]} строк × {df_clean.shape[1]} колонок')
    print(f"Графики сохранены в: {reports_path}")
    
if __name__ == "__main__":
    main()
    