import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ================== Гистограммы для 3 выбранных колонок ============================
def cols_for_histograms(df: pd.DataFrame) -> None:

    # Выбираем cols: ('имя_колонки', использовать_логарифм)
    cols_to_plot = [
        ('PL_revenue', True), # выручка
        ('PL_net_profit', True), # прибыль
        ('age', False) # возраст
    ]

    for col, use_log in cols_to_plot:
        if col in df.columns:
            plot_histogram(df, col, use_log=use_log)
        else:
            print(f'Колонка {col} не найдена')

def plot_histogram(df: pd.DataFrame, col: str, use_log: bool = True) -> int:

    data = df[col].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Гистограмма
    ax1 = axes[0]
    plot_data = data if not use_log else np.log1p(data[data >= 0])

    if len(plot_data) > 0:
        ax1.hist(plot_data, bins=40, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.axvline(plot_data.median(), color='red', linestyle='--', linewidth=2, label='Медиана')
        ax1.set_title(f'{col} {"(лог-шкала)" if use_log else ""}')
        ax1.set_xlabel('Значение')
        ax1.set_ylabel('Количество компаний')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Анализ: {col}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
