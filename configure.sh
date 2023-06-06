echo 'deb [trusted=yes] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list

sudo apt-get update -y

sudo apt-get install -y nvhpc-22-3

sed -i '1 i\export PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/compilers/bin/:$PATH"' ~/.bashrc

source ~/.bashrc