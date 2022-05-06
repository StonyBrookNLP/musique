# Dockerfile for ai2 leaderboard for musique

FROM python:3.7.4-alpine3.9
WORKDIR /app
COPY metrics/ metrics/
COPY evaluate_v0.1.py /app/evaluate_v0.1.py
