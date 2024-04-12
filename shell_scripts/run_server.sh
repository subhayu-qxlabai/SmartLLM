#!/bin/bash

PYTHON_VERSION="3.11"

python"$PYTHON_VERSION" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
