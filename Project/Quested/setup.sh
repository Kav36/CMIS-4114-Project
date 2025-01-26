#!/bin/bash

# Step 1: Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm install

# Step 2: Install Python dependencies
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  echo "requirements.txt not found!"
  exit 1
fi

# Step 3: Check for Tesseract installation (Python package)
echo "Checking for Tesseract installation..."
if ! command -v tesseract &> /dev/null
then
  echo "Tesseract not found. Installing Tesseract..."
  # For Linux (Debian/Ubuntu-based)
  sudo apt-get install tesseract-ocr
fi

echo "Setup complete!"
