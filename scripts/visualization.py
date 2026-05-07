import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ================== Гистограммы для 3 выбранных колонок ============================
def cols_for_histograms(
    df: pd.DataFrame,
    save_dir: str = None # для сохранения графика
) -> None:

    # Выбираем cols: ('имя_колонки', использовать_логарифм)
    cols_to_plot = [
        ('PL_revenue', True, 'revenue_hist.png'), # выручка
        ('PL_net_profit', True, 'profit_hist.png'), # прибыль
        ('age', False, 'age_hist.png') # возраст
    ]

    for col, use_log, filename in cols_to_plot:
        if col not in df.columns:
            print(f'Колонка {col} не найдена')
            continue
        
        # Путь для сохранения графика
        save_path = None
        if save_dir:
            save_path = Path(save_dir) / filename
            
        plot_histogram(df, col, use_log=use_log, save_path=save_path)

def plot_histogram(
    df: pd.DataFrame, 
    col: str,
    use_log: bool = True,
    save_path: str = None
) -> int:

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
        
    # Boxplot
    ax2 = axes[1]
    box_data = data if not use_log else np.log1p(data[data >= 0])
    if len(box_data) > 0:
        ax2.boxplot(box_data.dropna(), vert=False, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='blue'),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_title(f'Boxplot: {col} {"(лог)" if use_log else ""}')
        ax2.set_xlabel('Значение')
        ax2.grid(axis='x', alpha=0.3)

    plt.suptitle(f'Анализ: {col}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Сохраняем графики
    if save_path:
        # Создаём папку, если нет
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'График сохранён: {save_path}')
    
    # Показываем график
    plt.show()
    plt.close(fig)
    
    # ========================== Ищем выбросы (метод IQR) ========================
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]
    outlier_pct = len(outliers) / len(data) * 100

    print(f'\nВыбросы в {col}:')
    print(f'Диапазон нормальных: [{lower_bound:,.0f}; {upper_bound:,.0f}]')
    print(f'Выбросов: {len(outliers)} шт. ({outlier_pct:.1f}%)')

    if len(outliers) > 0 and len(outliers) <= 5:
        print(f'Значения: {outliers.tolist()}')
    elif len(outliers) > 5:
        print(f'Примеры: {outliers.head(3).tolist()} ...')

    return len(outliers)