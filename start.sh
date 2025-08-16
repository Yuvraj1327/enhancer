#!/bin/bash

# Start the FastAPI application with uvicorn
# Use the PORT environment variable provided by Railway
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}