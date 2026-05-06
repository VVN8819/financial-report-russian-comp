import pandas as pd
import numpy as np

# ======================== Исследовательский анализ EDA ============================
def basic_info(df: pd.DataFrame) -> None:
    df.info(memory_usage='deep')
    print(f'\nПервые 3 строки: {df.head(3)}')

# выявим пропущенные значения с помощью .isnull() и посчитаем их количество через mean()
def missing_values_rep(df: pd.DataFrame) -> None:
    print(df.isnull().mean())
    
def run_eda(df: pd.DataFrame) -> None:
    basic_info(df)
    missing_values_rep(df)
    