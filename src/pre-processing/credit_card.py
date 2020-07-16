import pandas as pd


def one_hot_encode(df):

    original_columns = list(df.columns)
    categoricals = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categoricals, dummy_na=False)
    categoricals = [c for c in df.columns if c not in original_columns]
    return df, categoricals


def aggregate(df, aggregates):
    df = df.groupby('SK_ID_CURR').agg(aggregates)
    df.columns = pd.Index(['{}{}_{}'.format('CREDIT_', e[0], e[1].upper()) for e in df.columns.tolist()])
    return df.reset_index()


def process_credit_card():

    df = pd.read_csv('../../data/credit_card_balance.csv')
    df, categorical = one_hot_encode(df)

    # feature engineering
    df['LIMIT_USE'] = df['AMT_BALANCE'] / df['AMT_CREDIT_LIMIT_ACTUAL']
    df['PAYMENT_DIV_MIN'] = df['AMT_PAYMENT_CURRENT'] / df['AMT_INST_MIN_REGULARITY']
    df['LATE_PAYMENT'] = df['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    df['DRAWING_LIMIT_RATIO'] = df['AMT_DRAWINGS_ATM_CURRENT'] / df['AMT_CREDIT_LIMIT_ACTUAL']

    categorical_agg = {key: ['mean'] for key in categorical}
    aggregates = {
        'SK_ID_PREV': ['nunique'],
        'MONTHS_BALANCE': ['min'],
        'AMT_BALANCE': ['max'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max'],
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['max', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['max', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['max', 'mean'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['max', 'mean', 'sum', 'var'],
        'AMT_TOTAL_RECEIVABLE': ['max', 'mean'],
        'CNT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['mean'],
        'SK_DPD': ['mean', 'max', 'sum'],
        'SK_DPD_DEF': ['max', 'sum'],
        'LIMIT_USE': ['max', 'mean'],
        'PAYMENT_DIV_MIN': ['min', 'mean'],
        'LATE_PAYMENT': ['max', 'sum'],
        # categorical
        **categorical_agg
    }

    credit_agg = aggregate(df, aggregates)
    credit_agg.to_csv('../../processed data/credit_card_processed.csv', index=False)

    return credit_agg


if __name__ == "__main__":
    process_credit_card()
