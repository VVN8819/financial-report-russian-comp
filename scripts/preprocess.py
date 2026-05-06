import pandas as pd
from pathlib import Path

# ======================== Очистка df ==============================================
# переменные B_longterm_debt, B_noncurrent_assets, CF_balance_operating, CF_balance_invest,
# CF_balance_fin, CF_balance , скорее всего, не являются самыми важными
def drop_unnecessary_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        'CF_balance_operating',
        'CF_balance_invest',
        'CF_balance_fin',
        'CF_balance',
        'B_longterm_debt',
        'B_noncurrent_assets'
    ]
    
    df_clean = df.copy()
    df_clean.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df_clean

# Оставляем только компании с важными колонками для анализа
def filter_imp_cols(df: pd.DataFrame, min_coverage: float = 3/7) -> pd.DataFrame:
    imp_cols = [
        'PL_revenue', 'B_assets', 'B_total_equity', 'PL_net_profit',
        'PL_cost_of_sales', 'PL_profit_from_sales', 'PL_before_tax'
    ]

    # Фильтруем строки, где есть хотя бы 3 из 7 ключевых показателей
    df_imp_cols = df[df[imp_cols].notna().sum(axis=1) >= 3].copy()

    print(f'Осталось {len(df_imp_cols)} из {len(df)} компаний ({len(df_imp_cols)/len(df)*100:.1f}%)')
    print(f'Пропуски после фильтрации:\n{df_imp_cols[imp_cols].isnull().mean()}')
    
    return df_imp_cols

# проверим на дубли
def duplicates_check(df: pd.DataFrame) -> int:
    num_duplicates = df.duplicated(subset=['inn', 'year']).sum()
    print(f'Найдено дублей: {num_duplicates} шт')
    return num_duplicates

# ====================== Заполним медианой остатки =====================
# Медиана для пустых значений
def impute_median(df: pd.DataFrame, empt_cols: list = None) -> pd.DataFrame:
    if empt_cols is None:
        empt_cols = [
            'PL_revenue', 'PL_net_profit', 'B_total_equity', 'B_shortterm_liab', 'B_current_assets',
            'PL_cost_of_sales', 'PL_profit_from_sales', 'B_cash_equivalents', 'PL_before_tax'
        ]
        
    df_imputed = df.copy()
    
    for col in empt_cols:
        if col in df_imputed.columns and df_imputed[col].isnull().any():
            median_val = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_val)
            print(f'{col}: заполнена медианой={median_val}')
            
    # проверим пропуски очищенного df
    remain_emp_cols = df_imputed[empt_cols].isnull().mean()
    print(f'Пропуски очищенного df: \n{remain_emp_cols}')
    
    return df_imputed

# ====================== Сохранение очищенного csv файла ===================
# сохраним полученный очищенный df в csv файл
def clean_df_save(df: pd.DataFrame, filename: str = "rfsd_sample_clean.csv") -> Path:
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)  # Создаём папку, если нет
    output_clean_csv = data_dir / filename
    
    df.to_csv(output_clean_csv, index=False, encoding='utf-8-sig')
    print(f'Сохранено в: {output_clean_csv}')
    
    return output_clean_csv

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_unnecessary_cols(df)
    df = filter_imp_cols(df)
    duplicates_check(df)
    df = impute_median(df)
    
    return df
    