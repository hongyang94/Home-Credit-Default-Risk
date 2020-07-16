import pandas as pd


def one_hot_encode(df):
    original_columns = list(df.columns)
    categoricals = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categoricals, dummy_na=False)
    categoricals = [c for c in df.columns if c not in original_columns]
    return df, categoricals


def group(df, aggregates):
    df = df.groupby('SK_ID_CURR').agg(aggregates)
    df.columns = pd.Index(['{}{}_{}'.format('IP_', e[0], e[1].upper()) for e in df.columns.tolist()])
    return df.reset_index()


def process_installment_payments():

    df = pd.read_csv('../../data/installments_payments.csv')
    df = one_hot_encode(df)

    aggregates = {
        'SK_ID_PREV': ['nunique'],
        'NUM_INSTALMENT_VERSION': ['min', 'max', 'mean'],
        'NUM_INSTALMENT_NUMBER': ['min', 'max', 'mean'],
        'DAYS_INSTALMENT': ['min', 'max', 'mean', 'var'],
        'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean', 'var'],
        'AMT_INSTALMENT': ['min', 'max', 'mean'],
        'AMT_PAYMENT': ['min', 'max', 'mean']
    }

    agg_ip = group(df, aggregates)
    agg_ip.to_csv('../../processed data/installment_processed.csv', index=False)

    return agg_ip


if __name__ == "__main__":
    process_installment_payments()
