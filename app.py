import streamlit as st
import pydeck as pdk
import pandas as pd
import joblib
import requests
from datetime import date, timedelta, time

# load in model created by model_training.py
model = joblib.load("models/xgboost_v1.joblib")

st.title("Flight Delay Predictor")

st.write(
    "Enter your flight details to estimate the probability "
    "that your flight will arrive late"
)

data = pd.read_csv("data/processed/routes.csv") # contains flight time and distance for each route

# contains lat, lon, timezone and full name for each airport
airline_information = pd.read_csv("data/processed/airport_information20260901.csv") 

airlines = ['American Airlines Inc.',
 'United Air Lines Inc.',
 'Delta Air Lines Inc.',
 'Alaska Airlines Inc.',
 'Republic Airline',
 'SkyWest Airlines Inc.',
 'Southwest Airlines Co.',
 'JetBlue Airways',
 'PSA Airlines Inc.',
 'Allegiant Air',
 'Envoy Air',
 'Frontier Airlines Inc.',
 'Hawaiian Airlines Inc.'] #full list of airlines from data, removed non-existent airlines such as Spirit

left_col, right_col = st.columns([1, 1.5], gap="large")

with left_col:

    airline = st.selectbox(
        "Airline",
        airlines
    )

    airport_name_lookup = (
    airline_information
    .set_index("iata_code")["municipality"]
    .to_dict())


    origins = list(data['ORIGIN'].unique())

    origin = st.selectbox(
        "Origin Airport",
        origins,
        format_func=lambda code: f"{code} — {airport_name_lookup.get(code, code)}"
    )

    # filter destination by origin to prevent selecting routes not found in the training data
    destinations = list(data.loc[data["ORIGIN"] == origin, "DEST"].unique())


    destination = st.selectbox(
        "Destination Airport",
        destinations,
        format_func=lambda code: f"{code} — {airport_name_lookup.get(code, code)}"
    )

    route = origin + "-" + destination

    route_information = data[
        (data["DEST"] == destination) &
        (data["ORIGIN"] == origin)
    ].iloc[0]

    crs_elapsed_time = int(route_information["CRS_ELAPSED_TIME"])
    distance = int(route_information["DISTANCE"])


    # time and location information
    today = date.today()

    departure_date = st.date_input("Date of Departure", 
                                   today, 
                                   min_value=today, 
                                   max_value=today + timedelta(days=7))

    month = departure_date.month
    day_of_month = departure_date.day
    day_of_week = departure_date.isoweekday()

    dep_time = st.time_input("Scheduled Departure Time")
    dep_hour = dep_time.hour


origin_information = airline_information[airline_information['iata_code'] == origin].iloc[0]
origin_latitude = float(origin_information['latitude_deg'])
origin_longitude = float(origin_information["longitude_deg"])

dest_information = airline_information[airline_information['iata_code'] == destination].iloc[0]
dest_latitude = float(dest_information['latitude_deg'])
dest_longitude = float(dest_information["longitude_deg"])

dest_timezone = dest_information['timezone']
origin_timezone = origin_information['timezone']

departure_time = time(hour=dep_hour)

#add local timezone to departure time
departure_local = pd.Timestamp.combine(
    departure_date,
    departure_time
).tz_localize(origin_timezone)

#create arrival time by adding elapsed time and converting to local time at destination
arrival_local = (
    departure_local
    + pd.to_timedelta(crs_elapsed_time, unit="m")
).tz_convert(dest_timezone)

#extract the hour in the expected format for the model
arr_hour = arrival_local.hour

#convert to standard time zone to match weather api format
departure_utc = departure_local.tz_convert("UTC")
arrival_utc = arrival_local.tz_convert("UTC")

dep_weather_hour = departure_utc.floor("h")
arr_weather_hour = arrival_utc.floor("h")

#route visual
route_map_data = pd.DataFrame({"origin":[[origin_longitude, origin_latitude]], 
                               "destination":[[dest_longitude, dest_latitude]]})

arc_layer = pdk.Layer(
    "ArcLayer",
    data=route_map_data,
    get_source_position="origin",
    get_target_position="destination",
    get_source_color=[70, 160, 255],
    get_target_color=[70, 160, 255],
    get_width=4,
    get_height=0.5,
)

airport_map_data = pd.DataFrame({"airport":[origin, destination], 
                                 "latitude":[origin_latitude, dest_latitude],
                                 "longitude":[origin_longitude, dest_longitude] })

airport_layer = pdk.Layer(
    "ScatterplotLayer",
    data=airport_map_data,
    get_position="[longitude, latitude]",
    get_radius=3000,
    get_fill_color=[255,255,255],
    pickable=True
)

label_layer = pdk.Layer(
    "TextLayer",
    data=airport_map_data,
    get_position="[longitude, latitude]",
    get_text="airport",
    get_size=16,
    get_color=[255, 255, 255],
    get_alignment_baseline="'bottom'",
    get_pixel_offset=[0, -20],
)

view_state = pdk.ViewState(
    latitude=38,
    longitude=-97,
    zoom=2.3,
    pitch=30,
)

route_map = pdk.Deck(
    map_style="dark",
    initial_view_state=view_state,
    layers=[
        arc_layer,
        airport_layer,
        label_layer
    ],
    tooltip={
        "text": "{airport}"
    },
)

with right_col:
    st.pydeck_chart(route_map, width="stretch", height=400)


weather_placeholder = st.empty()

predict_button = st.button(
    "Predict Delay",
    type="primary",
    use_container_width=True)

# get weather forecast for destination and origin at arrival and departure times 
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


#only run the api call on button press to reduce api calls 
if predict_button:

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

    with weather_placeholder.container():

        st.subheader("Weather Forecast")

        dep_col, arr_col = st.columns(2)

        with dep_col:
            st.markdown(f"### Departure")

            st.caption(
                departure_local.strftime("%b %d at %H:%M")
            )

            weather_col1, weather_col2 = st.columns(2)

            weather_col1.metric(
                "Temperature",
                f"{dep_temperature:.1f} °C"
            )

            weather_col2.metric(
                "Precipitation",
                f"{dep_precipitation:.1f} mm"
            )



        with arr_col:
            st.markdown(f"### Arrival")

            st.caption(
                arrival_local.strftime("%b %d at %H:%M")
            )

            weather_col1, weather_col2 = st.columns(2)

            weather_col1.metric(
                "Temperature",
                f"{arr_temperature:.1f} °C"
            )

            weather_col2.metric(
                "Precipitation",
                f"{arr_precipitation:.1f} mm"
            )


    # send all prediction information in the exact same format as the training data
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
        st.warning(
            "This flight is predicted to be delayed"
        )
    else:
        st.success(
            "This flight is predicted to arrive on time"
        )

with st.expander("Prediction Calculation"):

    st.info("""
    This prediction is based on the weather forecast and historical US flight data. The estimate could be inaccurate, particularly for less popular routes.
    If you would like to learn more, please check out the github repo.""")

