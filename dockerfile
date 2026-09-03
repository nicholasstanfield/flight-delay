FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_NO_DEV=1

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies (streamlit, pandas, pydeck, requests)
RUN uv sync --locked --no-dev --no-install-project

# Copy Streamlit application
COPY app.py ./


# Copy trained model
COPY models/xgboost_v1.joblib models/xgboost_v1.joblib

# copy in routes.csv, valid_flights.csv,  airport_information20260901.csv
COPY data/processed/routes.csv data/processed/routes.csv
COPY data/processed/valid_flights.csv data/processed/valid_flights.csv
COPY data/processed/airport_information20260901.csv data/processed/airport_information20260901.csv

# run streamlit app 
EXPOSE 8501

CMD ["uv", "run", "streamlit", "run","app.py", "--server.address=0.0.0.0", "--server.port=8501"]
