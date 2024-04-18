#!/bin/bash

apt install zsh -y
chsh -s $(which zsh) $USER
export CHSH=no
export RUNZSH=no
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
bash <(curl --proto '=https' --tlsv1.2 -sSf https://setup.atuin.sh)
