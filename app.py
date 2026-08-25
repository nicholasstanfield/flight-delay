import streamlit as st
import pandas as pd
import joblib



model = joblib.load("models/xgboost_v1.joblib")

st.title("Flight Delay Predictor")

st.write(
    "Enter your flight details to estimate the probability "
    "that your flight will arrive 15+ minutes late."
)



airline = st.selectbox(
    "Airline",
    ["Delta", "United", "American", "Southwest"]
)

origin = st.selectbox(
    "Origin Airport",
    ["JFK", "LAX", "ORD", "ATL"]
)

destination = st.selectbox(
    "Destination Airport",
    ["JFK", "LAX", "ORD", "ATL"]
)

month = st.selectbox(
    "Month",
    range(1, 13)
)

day_of_week = st.selectbox(
    "Day of Week",
    range(1, 8)
)

departure_time = st.time_input(
    "Scheduled Departure Time"
)

arr_snowfall = 0
crs_elapsed_time = 100
arr_wind_gusts = 4
distance = 1000
arr_visibility = 10000
origin_longitude = 48.45
dep_visibility = 100000
arr_precipitation = 0
dep_wind_gusts = 5
dep_precipitation = 10
arr_surface_pressure = 10000
dep_wind_speed = 0
dep_hour = 10
dep_snowfall = 0
dep_snow_depth = 0
arr_hour = 15
dep_temperature = 34
arr_wind_speed = 5
day_of_month = 24
dest_longitude = 45.34
origin_latitude = 67.54
arr_temperature = 20
dest_latitude = 34.0
dep_surface_pressure = 100000
arr_snow_depth = 0
route = origin + "-" + destination




if st.button("Predict Delay"):

    # Convert time to same format used during training
    crs_dep_time = (
        departure_time.hour * 100
        + departure_time.minute
    )

    # One row with the same columns the model expects
    flight = pd.DataFrame({
        "AIRLINE": [airline],
        "ORIGIN": [origin],
        "DEST": [destination],
        "MONTH": [month],
        "DAY_OF_WEEK": [day_of_week],
        "CRS_DEP_TIME": [crs_dep_time],
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