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

def drop_last_day(df):
     # merging weather data later introduces a small amount of missing weather data at the end of the year so safest to drop last day of data
    last_day_mask = (
        (df["MONTH"] == 12) &
        (df["DAY_OF_MONTH"] == 31)
    )

    df = df[~last_day_mask]

    return df

#TODO MERGE COORDS 
def add_airport_coords(df):

    airports = pd.read_csv("data/raw/airport_coordinates_20260819.csv") 
    print(f"Before merge: {df.shape}")
    print(f"Number of missing: {df.isna().sum().sum()}")
    
    df = df.merge(airports, left_on="ORIGIN", right_on="iata_code", how="inner")
    df = df.rename(columns={"latitude_deg":"ORIGIN_LATITUDE","longitude_deg":"ORIGIN_LONGITUDE","timezone":"ORIGIN_TIMEZONE"})
    df = df.drop('iata_code', axis=1)
    
    df = df.merge(airports, left_on="DEST", right_on="iata_code", how="inner")
    df = df.rename(columns={"latitude_deg":"DEST_LATITUDE","longitude_deg":"DEST_LONGITUDE","timezone":"DEST_TIMEZONE"})
    df = df.drop('iata_code', axis=1)
    
    print(f"After merge: {df.shape}")
    print(f"Number of missing: {df.isna().sum().sum()}")

    return df

#TODO CONVERT CRS TIME INTO HOURS 
def convert_crs_to_hours(df):
    """ 

    The primary purpose of this function is to get each row to have a timestamp of departure and arrival in the format 
    yyyy-mm-dd hh-mm-ss+00:00 so that the weather data can be merged into the data. To do this a timestamp column for arrival
    and departure is created, localized to be in UTC and floored at the hour level. 


    """
    df["SCHEDULED_DEP_DATETIME"] = pd.to_datetime({
        "year": 2025,
        "month": df["MONTH"],
        "day": df["DAY_OF_MONTH"],
        "hour": df["CRS_DEP_TIME"] // 100,
        "minute": df["CRS_DEP_TIME"] % 100,
    })

    df["SCHEDULED_DEP_UTC"] = [
        dt.tz_localize(
            tz,
            ambiguous=False, #need to avoid errors when the clocks go back/forward
            nonexistent="shift_forward"
        ).tz_convert("UTC")
        for dt, tz in zip(
            df["SCHEDULED_DEP_DATETIME"],
            df["ORIGIN_TIMEZONE"]
        )
    ]

    df["SCHEDULED_ARR_UTC"] = (
        df["SCHEDULED_DEP_UTC"]
        + pd.to_timedelta(df["CRS_ELAPSED_TIME"], unit="m")
    )

    df["DEP_WEATHER_HOUR"] = (
        df["SCHEDULED_DEP_UTC"].dt.floor("h")
    )

    df["ARR_WEATHER_HOUR"] = (
        df["SCHEDULED_ARR_UTC"].dt.floor("h")
    )

    print(f"Number of missing after data changes: {df.isna().sum().sum()}")

    return df
    
#TODO MERGE WEATHER DATA 
def add_weather_data(df):
    weather = pd.read_parquet("data/processed/weather_2025.parquet")

    dep_weather = weather.rename(columns={"iata_code": "ORIGIN",
                                          "WEATHER_HOUR": "DEP_WEATHER_HOUR",
                                          "temperature_2m": "DEP_TEMPERATURE",
                                          "precipitation": "DEP_PRECIPITATION",
                                          "snowfall": "DEP_SNOWFALL",
                                          "snow_depth": "DEP_SNOW_DEPTH",
                                          "visibility": "DEP_VISIBILITY",
                                          "surface_pressure": "DEP_SURFACE_PRESSURE",
                                          "wind_speed_10m": "DEP_WIND_SPEED",
                                          "wind_gusts_10m": "DEP_WIND_GUSTS",}) 
    

    print(f"Dataframe shape premerge: {df.shape}")

    df = df.merge(dep_weather,on=["ORIGIN", "DEP_WEATHER_HOUR"],how="left",validate="many_to_one")

    arr_weather = weather.rename(columns={"iata_code": "DEST",
                                          "WEATHER_HOUR": "ARR_WEATHER_HOUR",
                                          "temperature_2m": "ARR_TEMPERATURE",
                                          "precipitation": "ARR_PRECIPITATION",
                                          "snowfall": "ARR_SNOWFALL",
                                          "snow_depth": "ARR_SNOW_DEPTH",
                                          "visibility": "ARR_VISIBILITY",
                                          "surface_pressure": "ARR_SURFACE_PRESSURE",
                                          "wind_speed_10m": "ARR_WIND_SPEED",
                                          "wind_gusts_10m": "ARR_WIND_GUSTS",})
    
    df = df.merge(arr_weather,on=["DEST", "ARR_WEATHER_HOUR"],how="left",validate="many_to_one")

    print(f"Dataframe shape postmerge: {df.shape}")


    
    return df 


def create_hour_cols(df):
    df["DEP_HOUR"] = df["CRS_DEP_TIME"] // 100
    df["ARR_HOUR"] = df["CRS_ARR_TIME"] // 100

    return df 

def drop_unneeded_columns(df):
    unneeded_cols = ['ORIGIN_TIMEZONE', 
                     'DEST_TIMEZONE',
                     'SCHEDULED_DEP_DATETIME', 
                     'SCHEDULED_DEP_UTC', 
                     'SCHEDULED_ARR_UTC',
                     'DEP_WEATHER_HOUR', 
                     'ARR_WEATHER_HOUR',
                     'QUARTER',
                     'CRS_DEP_TIME',
                     'CRS_ARR_TIME']
    
    df = df.drop(columns=unneeded_cols)

    return df 

def sample_data(df, n_per_month):
    df = (
        df.groupby("MONTH", group_keys=False)
          .sample(n=n_per_month, random_state=42)
    )

    print(f"Sampled {n_per_month:,} flights per month")
    print(f"Final sample size: {len(df):,}")

    return df

def output_data(df):
    df.to_csv("data/processed/flight_weather_data_2025.csv", index=False)



df = concat_data()
df = drop_duplicates(df)
df = remove_cancelled_flights(df)
df = drop_missing_y_data(df)
df = create_airline_column(df)
df = drop_redundant_cols(df, ["ORIGIN_CITY_NAME","DEST_CITY_NAME","ARR_DELAY_NEW"])
df = make_route(df)
df = convert_delay_to_int(df)
df = drop_last_day(df)

df = sample_data(df, 30_000)

df = add_airport_coords(df)
df = convert_crs_to_hours(df)
df = add_weather_data(df)
df = create_hour_cols(df)
df = drop_unneeded_columns(df)


print(df.shape)
print(df.isna().sum().sort_values(ascending=False).head(20))
print(df["DELAY"].value_counts(normalize=True))
print(df.groupby("MONTH").size())
print(df.dtypes)

output_data(df)