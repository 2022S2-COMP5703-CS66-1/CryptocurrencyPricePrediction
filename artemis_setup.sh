cd ~/CryptocurrencyPricePrediction

mkdir data

rsync -rt /rds/PRJ-Crypto_S22_22/* ~/CryptocurrencyPricePrediction/data

module load python/3.8.2 magma/2.5.3

virtualenv --system-site-packages ~/pytorch

source ~/pytorch/bin/activate

pip install /usr/local/pytorch/cuda10.2/torch-1.10.1-cp38-cp38-linux_x86_64.whl

