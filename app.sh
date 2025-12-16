#!/bin/bash
# This runs simplest_agent as an app with a UI

# Install dependencies if needed
# pip install fastapi uvicorn

# Configuration
PORT=8888

# Run the chat application
python chat_app.py --debug --port $PORT
