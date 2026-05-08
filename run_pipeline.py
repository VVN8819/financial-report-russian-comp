from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from rfsd_loader import load_rfsd
from preprocess import preprocess_pipeline, clean_df_save
from eda import run_eda
from visualization import cols_for_histograms
from correlations import plot_correlation_matrix, plot_scatter_pairs

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
    
    # Корреляция
    plot_correlation_matrix(
        df_clean, 
        save_path=reports_path / "correlation_matrix.png"
    )
    
    # Диаграммы рассеивания
    pairs_to_plot = [
        ('PL_revenue', 'PL_cost_of_sales', 'Выручка vs Себестоимость'),
        ('PL_before_tax', 'PL_net_profit', 'Прибыль до налогов vs Чистая прибыль'),
        ('B_assets', 'B_shortterm_liab', 'Активы vs Краткосрочные обязательства'),
        ('B_assets', 'B_current_assets', 'Активы vs Оборотные активы'),
        ('B_shortterm_liab', 'B_current_assets', 'Кратк. обязательства vs Оборотные активы'),
        ('PL_profit_from_sales', 'B_total_equity', 'Прибыль от продаж vs Капитал'),
        ('PL_profit_from_sales', 'PL_before_tax', 'Прибыль до налогов vs Прибыль от продаж')
    ]
    
    plot_scatter_pairs(
        df_clean,
        pairs_to_plot,
        save_dir=reports_path
    )
    
    print(f"Сырые данные: {base_path / 'scripts' / 'data' / 'rfsd_sample.csv'}")
    print(f'Очищенные данные: {output_path}')
    print(f'Итоговый размер: {df_clean.shape[0]} строк × {df_clean.shape[1]} колонок')
    print(f"Графики сохранены в: {reports_path}")
    
if __name__ == "__main__":
    main()
    