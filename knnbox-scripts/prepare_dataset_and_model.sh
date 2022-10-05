# download pre-processed multi-domain de-en dataset from Google driver
# !!!! If your server can't access google driver, you can maually download dataset from the link below and unzip&&copy it to main project directory
# You should have a `data-bin/` folder under main directory after running following scripts 
pip install gdown
gdown https://drive.google.com/file/d/18TXCWzoKuxWKHAaCRgddd6Ub64klrVhV/view?usp=sharing
unzip data-bin.zip -d ../ 
rm -rf ../__MACOSX
rm data-bin.zip


# download wmt19 de-en model from fairseq
# You should have a `pretrain-models/` folder under main directory after running following scripts
wget -c https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.ffn8192.tar.gz
mkdir -p ../pretrain-models/wmt19.de-en
tar -xvf wmt19.de-en.ffn8192.tar.gz -C ../pretrain-models/wmt19.de-en
rm wmt19.de-en.ffn8192.tar.gz
