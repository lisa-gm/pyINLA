#!/bin/bash

# This script only purpose is to isntall and configure git-lfs on Daint@ALPS
# as it is not avaialble by default in the system...

# 1. Move to your home directory
cd $HOME || { echo "Could not change to home directory"; exit 1; }

# 2. Download git-lfs for ARM64
wget https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-linux-arm64-v3.7.1.tar.gz || { echo "Could not download git-lfs"; exit 1; }

# 3. Extract the downloaded tarball
tar -xvf git-lfs-linux-arm64-v3.7.1.tar.gz || { echo "Could not extract git-lfs tarball"; exit 1; }

# 4. Move in the extracted directory and make the installer executable
cd git-lfs-3.7.1 || { echo "Could not change to git-lfs directory"; exit 1; }
chmod +x install.sh || { echo "Could not make installer executable"; exit 1; }

# 5. Change the installer prefix to your home directory
sed -i 's|^prefix="/usr/local"$|prefix="$HOME/.local"|' install.sh || { echo "Could not modify installer prefix"; exit 1; }

# 6. Make the .local/bin directory if it does not exist
mkdir -p "$HOME/.local/bin" || { echo "Could not create .local/bin directory"; exit 1; }

# 7. Run the installer
./install.sh || { echo "Could not install git-lfs"; exit 1; }

# 8. Add .local/bin to your PATH if not already present
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# 9 Add .local/bin to your PATH in .bashrc for future sessions
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

# 10. Verify the installation
if command -v git-lfs &> /dev/null; then
    echo "git-lfs installed successfully!"
else
    echo "git-lfs installation failed"
    exit 1
fi