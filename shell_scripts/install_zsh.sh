#!/bin/bash

sudo apt install zsh -y
sudo chsh -s $(which zsh) $USER
CHSH=no;RUNZSH=no sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
bash <(curl --proto '=https' --tlsv1.2 -sSf https://setup.atuin.sh)
