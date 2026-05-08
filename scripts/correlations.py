import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ========== Расчёт и визуализация корреляций ==================
# Строим тепловую карту корреляций
def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: str = None,
) -> None:

    # Выбираем только числовые колонки
    num_cols = df.select_dtypes(include=[np.number]).columns

    # Считаем корреляцию Пирсона
    corr_matrix = df[num_cols].corr()

    # ================== Тепловая карта ======================
    plt.figure(figsize=(14, 12))

    sns.heatmap(
        corr_matrix,
        annot=True, # показывать значения
        fmt='.2f', # 2 знака после запятой
        cmap='viridis',
        center=0, # белый цвет для 0
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Корреляция"}
    )

    plt.title('Матрица корреляций', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)

    # Сохранение
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'График сохранён: {save_path}')

    plt.tight_layout()
    plt.show()

    return corr_matrix

# ======================= Диаграммы рассеивания ==============================
# Строит Scatter Plot (диаграмму рассеивания) для списка пар колонок
def plot_scatter_pairs(df: pd.DataFrame, pairs: list, save_dir: str = None) -> None:

    for x_col, y_col, title in pairs: # цыкл для нескольких пар
        # Проверка наличия колонок
        if x_col not in df.columns or y_col not in df.columns:
            print(f'Нет колонок {x_col} или {y_col}')
            continue

        # Фильтруем данные для Лог-шкалы
        mask = (df[x_col] > 0) & (df[y_col] > 0)
        plot_df = df.loc[mask, [x_col, y_col]]

        if len(plot_df) < 10:
            print(f'Мало данных для {x_col} vs {y_col}')
            continue

        # Рисуем график
        plt.figure(figsize=(10, 8))

        # regplot рисует точки + линию линейной регрессии (тренд)
        sns.regplot(
            x=x_col,
            y=y_col,
            data=plot_df,
            scatter_kws={'alpha': 0.6, 'color': 'blue', 's': 40}, # Точки
            line_kws={'color': 'red', 'linewidth': 2} # Линия тренда
        )
        
        # Лог-шкала
        plt.xscale('log')
        plt.yscale('log')

        plt.title(f'{title} (Log)', fontsize=14)
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.3)

        # Сохранение
        filename = f"scatter_log_{x_col}_vs_{y_col}.png"
        if save_dir:
            full_path = Path(save_dir) / filename
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f'Сохранено: {filename}')

        plt.tight_layout()
        plt.show()