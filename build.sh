#!/usr/bin/env bash

# Install Tesseract OCR
apt-get update && apt-get install -y tesseract-ocr

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
