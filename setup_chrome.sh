#!/bin/bash
# Setup script for installing Chrome/Chromium on Vast AI

echo "Installing Chromium browser and dependencies..."

# Update package list
apt-get update

# Install Chromium, chromedriver, and ALL required dependencies
# These are needed for both Chromium and chromedriver to work
# Note: Ubuntu 24.04 uses t64 suffix for some packages
apt-get install -y \
    chromium-browser \
    chromium-chromedriver \
    libnss3 \
    libnss3-dev \
    libnspr4 \
    libatk-bridge2.0-0t64 \
    libatk1.0-0t64 \
    libatspi2.0-0t64 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2t64 \
    libxss1 \
    libgtk-3-0t64 \
    libgdk-pixbuf2.0-0 \
    libxshmfence1 \
    libgl1

# Make chromedriver executable if it exists
if [ -f /usr/bin/chromedriver ]; then
    chmod +x /usr/bin/chromedriver
    echo "Made chromedriver executable"
fi

# Verify installation
if command -v chromium-browser &> /dev/null; then
    echo "Chromium installed successfully at: $(which chromium-browser)"
    if command -v chromedriver &> /dev/null; then
        echo "Chromedriver found at: $(which chromedriver)"
    else
        echo "Warning: chromedriver not found in PATH, but may be installed"
    fi
else
    echo "Chromium installation failed. Trying alternative method..."
    
    # Alternative: Install Google Chrome
    wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list
    apt-get update
    apt-get install -y google-chrome-stable
    
    if command -v google-chrome-stable &> /dev/null; then
        echo "Google Chrome installed successfully at: $(which google-chrome-stable)"
    else
        echo "Chrome installation failed. Please install manually."
        exit 1
    fi
fi

echo "Setup complete!"

