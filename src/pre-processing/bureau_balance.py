import pandas as pd


def one_hot_encode(df):
    original_columns = list(df.columns)
    categoricals = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categoricals, dummy_na=False)
    categoricals = [c for c in df.columns if c not in original_columns]
    return df, categoricals


def aggregate_and_merge(df, merge, aggregates):
    df = df.groupby('SK_ID_BUREAU').agg(aggregates)
    df.columns = pd.Index(['{}_{}'.format(e[0], e[1].upper()) for e in df.columns.tolist()])
    df = df.reset_index()
    return merge.merge(df, how='left', on='SK_ID_BUREAU')


def process_bureau_balance():

    # read file
    bureau_balance = pd.read_csv('../../data/bureau_balance.csv')

    # one hot encode
    bureau_balance, categoricals = one_hot_encode(bureau_balance)

    # aggregate
    bb_processed = bureau_balance.groupby('SK_ID_BUREAU')[categoricals].mean().reset_index()
    agg = {'MONTHS_BALANCE': ['min', 'max', 'mean', 'size']}

    bb_processed = aggregate_and_merge(bureau_balance, bb_processed, agg)
    bb_processed.to_csv('../../processed data/bureau_balance_processed.csv', index=False)

    return bb_processed


if __name__ == "__main__":
    process_bureau_balance()
