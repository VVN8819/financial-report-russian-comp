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