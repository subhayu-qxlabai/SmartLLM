#!/bin/bash

# Define variables
PYTHON_DEFAULT_VERSION="3.8"
PYTHON_NEW_VERSION="3.11"
PYTHON_OLD_VERSION=$(python3 --version | cut -d " " -f 2)
DEADSNAKES_REPO="ppa:deadsnakes/ppa"

# Function to update system and add deadsnakes repository
update_system_and_repository() {
    echo "Updating system and adding deadsnakes repository..."
    sudo apt update -y && sudo apt upgrade -y
    sudo add-apt-repository -y $DEADSNAKES_REPO
    sudo apt update -y
}

# Function to check if Python new version is available
check_python_new() {
    echo "Checking if Python $PYTHON_NEW_VERSION is available..."
    if apt list --installed | grep -q python$PYTHON_NEW_VERSION; then
        echo "Python $PYTHON_NEW_VERSION is already installed. Exiting..."
        exit 0
    else
        echo "Python $PYTHON_NEW_VERSION is not installed."
    fi
}

# Function to install Python new version
install_python_new() {
    echo "Installing Python $PYTHON_NEW_VERSION..."
    sudo apt install -y python$PYTHON_NEW_VERSION
}

# Function to run Python new version
run_python_new() {
    echo "Running Python $PYTHON_NEW_VERSION..."
    python$PYTHON_NEW_VERSION --version
}

# Function to create and activate virtual environment
create_and_activate_virtualenv() {
    echo "Creating and activating virtual environment..."
    python$PYTHON_NEW_VERSION -m venv env
    source env/bin/activate
}

# Function to set aliases for Python 3
set_python_aliases() {
    echo "Setting aliases for Python 3..."
    echo "alias py=/usr/bin/python3" >> ~/.bashrc
    echo "alias python=/usr/bin/python3" >> ~/.bashrc
}

# Function to set Python new version as default (can disrupt some OS packages that rely on Python 3.8)
set_default_python_new() {
    echo "Setting Python $PYTHON_NEW_VERSION as default..."
    sudo sed -i "s|#!/usr/bin/python3|#!/usr/bin/python$PYTHON_DEFAULT_VERSION|" /usr/bin/gnome-terminal
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python$PYTHON_DEFAULT_VERSION 1
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python$PYTHON_NEW_VERSION 2
    sudo update-alternatives --config python3
}

# Function to fix pip and disutils errors
fix_pip_and_distutils_errors() {
    echo "Fixing pip and distutils errors..."
    sudo apt remove --purge -y python3-apt
    sudo apt autoclean
    sudo apt install -y curl
    sudo apt install -y python3-apt
    sudo apt install -y python$PYTHON_NEW_VERSION-distutils
    curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    sudo python$PYTHON_NEW_VERSION get-pip.py
    sudo apt install -y python$PYTHON_NEW_VERSION-venv
}

# Main script
check_python_new
update_system_and_repository
install_python_new
run_python_new
# create_and_activate_virtualenv
# set_python_aliases
# set_default_python_new # can disrupt some OS packages that rely on Python 3.8
fix_pip_and_distutils_errors

echo "Python upgrade completed successfully."
