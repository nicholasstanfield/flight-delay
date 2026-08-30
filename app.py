import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import date, timedelta, time


model = joblib.load("models/xgboost_v1.joblib")

st.title("Flight Delay Predictor")

st.write(
    "Enter your flight details to estimate the probability "
    "that your flight will arrive late"
)

data = pd.read_csv("data/processed/routes.csv")
airline_information = pd.read_csv("data/raw/airport_coordinates_20260819.csv")

airlines = ['American Airlines Inc.',
 'United Air Lines Inc.',
 'Delta Air Lines Inc.',
 'Alaska Airlines Inc.',
 'Republic Airline',
 'SkyWest Airlines Inc.',
 'Southwest Airlines Co.',
 'JetBlue Airways',
 'PSA Airlines Inc.',
 'Spirit Air Lines',
 'Allegiant Air',
 'Envoy Air',
 'Frontier Airlines Inc.',
 'Hawaiian Airlines Inc.']

airline = st.selectbox(
    "Airline",
    airlines
)

origins = list(data['ORIGIN'].unique())

origin = st.selectbox(
    "Origin Airport",
    origins
)

destinations = list(data.loc[data["ORIGIN"] == origin, "DEST"].unique())


destination = st.selectbox(
    "Destination Airport",
    destinations
)

route = origin + "-" + destination

route_information = data[
    (data["DEST"] == destination) &
    (data["ORIGIN"] == origin)
].iloc[0]

crs_elapsed_time = int(route_information["CRS_ELAPSED_TIME"])
distance = int(route_information["DISTANCE"])

today = date.today()

departure_date = st.date_input("Date of Departure", today, min_value=today, max_value=today + timedelta(days=7))

month = departure_date.month
day_of_month = departure_date.day
day_of_week = departure_date.isoweekday()

origin_information = airline_information[airline_information['iata_code'] == origin].iloc[0]
origin_latitude = float(origin_information['latitude_deg'])
origin_longitude = float(origin_information["longitude_deg"])

dest_information = airline_information[airline_information['iata_code'] == destination].iloc[0]
dest_latitude = float(dest_information['latitude_deg'])
dest_longitude = float(dest_information["longitude_deg"])

dest_timezone = dest_information['timezone']
origin_timezone = origin_information['timezone']


dep_hour = st.number_input(
    "Scheduled Departure Hour",0,23
)

departure_time = time(hour=dep_hour)

departure_local = pd.Timestamp.combine(
    departure_date,
    departure_time
).tz_localize(origin_timezone)

arrival_local = (
    departure_local
    + pd.to_timedelta(crs_elapsed_time, unit="m")
).tz_convert(dest_timezone)

arr_hour = arrival_local.hour

departure_utc = departure_local.tz_convert("UTC")
arrival_utc = arrival_local.tz_convert("UTC")

dep_weather_hour = departure_utc.floor("h")
arr_weather_hour = arrival_utc.floor("h")

weather_variables = [
    "temperature_2m",
    "precipitation",
    "snowfall",
    "snow_depth",
    "visibility",
    "surface_pressure",
    "wind_speed_10m",
    "wind_gusts_10m",
]

url = "https://api.open-meteo.com/v1/forecast"


def get_weather_data(latitude, longitude, weather_hour):

    weather_date = weather_hour.date().isoformat()
    hour = weather_hour.hour

    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(weather_variables),
        "start_date": weather_date,
        "end_date": weather_date,
        "timezone": "GMT"
    }

    response = requests.get(url, params=payload)
    response.raise_for_status()

    weather_data = response.json()

    weather_df = pd.DataFrame(weather_data["hourly"])

    weather_df["time"] = pd.to_datetime(weather_df["time"])
    weather_df["hour"] = weather_df["time"].dt.hour

    weather_df = weather_df[
        weather_df["hour"] == hour
    ]

    return weather_df



if st.button("Predict Delay"):

    dep_weather = get_weather_data(origin_latitude, origin_longitude, dep_weather_hour)

    dep_temperature = dep_weather["temperature_2m"].iloc[0]
    dep_precipitation = dep_weather["precipitation"].iloc[0]
    dep_snowfall = dep_weather["snowfall"].iloc[0]
    dep_snow_depth = dep_weather["snow_depth"].iloc[0]
    dep_visibility = dep_weather["visibility"].iloc[0]
    dep_surface_pressure = dep_weather["surface_pressure"].iloc[0]
    dep_wind_speed = dep_weather["wind_speed_10m"].iloc[0]
    dep_wind_gusts = dep_weather["wind_gusts_10m"].iloc[0]

    arr_weather = get_weather_data(dest_latitude, dest_longitude, arr_weather_hour)

    arr_temperature = arr_weather["temperature_2m"].iloc[0]
    arr_precipitation = arr_weather["precipitation"].iloc[0]
    arr_snowfall = arr_weather["snowfall"].iloc[0]
    arr_snow_depth = arr_weather["snow_depth"].iloc[0]
    arr_visibility = arr_weather["visibility"].iloc[0]
    arr_surface_pressure = arr_weather["surface_pressure"].iloc[0]
    arr_wind_speed = arr_weather["wind_speed_10m"].iloc[0]
    arr_wind_gusts = arr_weather["wind_gusts_10m"].iloc[0]



    # One row with the same columns the model expects
    flight = pd.DataFrame({
        "AIRLINE": [airline],
        "ORIGIN": [origin],
        "DEST": [destination],
        "MONTH": [month],
        "DAY_OF_WEEK": [day_of_week],
        "ARR_SNOWFALL": [arr_snowfall],
        "CRS_ELAPSED_TIME": [crs_elapsed_time],
        "ARR_WIND_GUSTS": [arr_wind_gusts],
        "DISTANCE": [distance],
        "ARR_VISIBILITY": [arr_visibility],
        "ORIGIN_LONGITUDE": [origin_longitude],
        "DEP_VISIBILITY": [dep_visibility],
        "ARR_PRECIPITATION": [arr_precipitation],
        "DEP_WIND_GUSTS": [dep_wind_gusts],
        "DEP_PRECIPITATION": [dep_precipitation],
        "ARR_SURFACE_PRESSURE": [arr_surface_pressure],
        "DEP_WIND_SPEED": [dep_wind_speed],
        "DEP_HOUR": [dep_hour],
        "DEP_SNOWFALL": [dep_snowfall],
        "DEP_SNOW_DEPTH": [dep_snow_depth],
        "ARR_HOUR": [arr_hour],
        "DEP_TEMPERATURE": [dep_temperature],
        "ARR_WIND_SPEED": [arr_wind_speed],
        "DAY_OF_MONTH": [day_of_month],
        "DEST_LONGITUDE": [dest_longitude],
        "ORIGIN_LATITUDE": [origin_latitude],
        "ARR_TEMPERATURE": [arr_temperature],
        "DEST_LATITUDE": [dest_latitude],
        "DEP_SURFACE_PRESSURE": [dep_surface_pressure],
        "ARR_SNOW_DEPTH": [arr_snow_depth],
        "ROUTE": [route]


    })

    probability = model.predict_proba(flight)[0, 1]

    prediction = model.predict(flight)[0]

    st.metric(
        "Probability of 15+ Minute Delay",
        f"{probability:.1%}"
    )

    if prediction == 1:
        st.warning("This flight is predicted to be delayed.")
    else:
        st.success("This flight is predicted to arrive on time.")

with st.expander("Prediction Calculation"):
    st.info("This prediction is based on historical weather and US flight data, this estimate could be inaccurate, particularly for less popular routes")