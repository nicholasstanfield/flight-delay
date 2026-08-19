import pandas as pd 

def concat_data():
    # load in each flight, concat together
    jan = pd.read_csv("data/raw/2025/flight_2025_01.csv")
    feb = pd.read_csv("data/raw/2025/flight_2025_02.csv")
    mar = pd.read_csv("data/raw/2025/flight_2025_03.csv")
    apr = pd.read_csv("data/raw/2025/flight_2025_04.csv")
    may = pd.read_csv("data/raw/2025/flight_2025_05.csv")
    jun = pd.read_csv("data/raw/2025/flight_2025_06.csv")
    jul = pd.read_csv("data/raw/2025/flight_2025_07.csv")
    aug = pd.read_csv("data/raw/2025/flight_2025_08.csv")
    sep = pd.read_csv("data/raw/2025/flight_2025_09.csv")
    oct = pd.read_csv("data/raw/2025/flight_2025_10.csv")
    nov = pd.read_csv("data/raw/2025/flight_2025_11.csv")
    dec = pd.read_csv("data/raw/2025/flight_2025_12.csv")

    df = pd.concat([jan,feb,mar,apr,may,jun,jul,aug,sep,oct,nov,dec], ignore_index=True) #create a new index for the data

    return df 

def drop_duplicates(df):
    num_duplicates = df.duplicated().sum()

    print(num_duplicates, "duplicates dropped")

    df = df.drop_duplicates()

    return df 

def remove_cancelled_flights(df):
    """ Cancelled flights account for all (in the 2025) missing target rows. They are dropped here because they cannot be 
    be used to predict delays """

    is_cancelled = df["CANCELLED"].astype(int) != 0 
    print(f"Total flights: {df.shape[0]}")
    df = df[~is_cancelled]
    df = df.drop("CANCELLED",axis=1)
    print(f"Total flights after removing cancelled flights: {df.shape[0]}")

    return df 

def drop_missing_y_data(df):
    num_missing = df["ARR_DEL15"].isna().sum()

    print(num_missing, "rows with missing delay information removed")

    df = df.dropna(subset=["ARR_DEL15"])

    return df 

def create_airline_column(df):

    carriers = pd.read_csv("data/raw/carrier_lookup_table.csv")
    df = df.merge(carriers, left_on='OP_UNIQUE_CARRIER', right_on='Code', how='left')
    df['AIRLINE'] = df['Description']
    df = df.drop(['OP_UNIQUE_CARRIER','Code','Description'], axis=1)
    print(list(df.columns))
    return df

def drop_redundant_cols(df, redundant_cols: list):
    df = df.drop(redundant_cols, axis=1)
    return df

def make_route(df):
    df["ROUTE"] = df["ORIGIN"] + "-" + df["DEST"]

    print(f"There are {df['ROUTE'].nunique()} routes in the data")

    return df 

def convert_delay_to_int(df):

    df["DELAY"] = df["ARR_DEL15"].astype(int)
    df = df.drop("ARR_DEL15", axis=1)

    return df

#TODO CONVERT CRS TIME INTO HOURS 

#TODO MERGE COORDS 

#TODO MERGE WEATHER DATA 

def sample_data(df, n_per_month):
    df = (
        df.groupby("MONTH", group_keys=False)
          .sample(n=n_per_month, random_state=42)
    )

    print(f"Sampled {n_per_month:,} flights per month")
    print(f"Final sample size: {len(df):,}")

    return df

def output_data(df):
    df.to_csv("data/processed/flight_data_2025.csv", index=False)



df = concat_data()
df = drop_duplicates(df)
df = remove_cancelled_flights(df)
df = drop_missing_y_data(df)
df = create_airline_column(df)
df = drop_redundant_cols(df, ["ORIGIN_CITY_NAME","DEST_CITY_NAME","ARR_DELAY_NEW"])
df = make_route(df)
df = convert_delay_to_int(df)

df = sample_data(df, 30_000)
output_data(df)

