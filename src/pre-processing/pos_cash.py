import pandas as pd


def one_hot_encode(df):
    original_columns = list(df.columns)
    categoricals = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categoricals, dummy_na=False)
    categoricals = [c for c in df.columns if c not in original_columns]
    return df, categoricals


def aggregate(df, aggregates):
    df = df.groupby('SK_ID_CURR').agg(aggregates)
    df.columns = pd.Index(['{}{}_{}'.format('PCB_', e[0], e[1].upper()) for e in df.columns.tolist()])
    return df.reset_index()


def process_pos_cash():

    df = pd.read_csv('../../data/POS_CASH_balance.csv')
    df, categoricals = one_hot_encode(df)
    categorical_agg = {key: ['mean', 'sum'] for key in categoricals}

    aggregates = {
        'SK_ID_PREV': ['nunique'],
        'MONTHS_BALANCE': ['min', 'max', 'size'],
        'CNT_INSTALMENT': ['min', 'max', 'mean', 'size'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
        'SK_DPD': ['max', 'mean', 'sum', 'var'],
        'SK_DPD_DEF': ['max', 'mean', 'sum'],
        **categorical_agg
    }

    credit_agg = aggregate(df, aggregates)
    credit_agg.to_csv('../../processed data/pos_cash_processed.csv', index=False)

    return credit_agg


if __name__ == "__main__":
    process_pos_cash()
