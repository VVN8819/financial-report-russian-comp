import pandas as pd
import numpy as np

# ======================== Исследовательский анализ EDA ============================
def basic_info(df: pd.DataFrame) -> None:
    df.info(memory_usage='deep')
    print(f'\nПервые 3 строки: \n{df.head(3)}')

# выявим пропущенные значения с помощью .isnull() и посчитаем их количество через mean()
def missing_values_rep(df: pd.DataFrame) -> None:
    print(df.isnull().mean())

# Описательная статистика
def descrip_stat(df: pd.DataFrame) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns
    print(df[num_cols].describe().T.round(2))

def run_eda(df: pd.DataFrame) -> None:
    basic_info(df)
    missing_values_rep(df)
    descrip_stat(df)
    