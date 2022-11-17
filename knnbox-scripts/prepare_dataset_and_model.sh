:<<! 
[script description]: download pre-trained neural models, pre-processed multi-domain dataset 
and create folder to store them.

Hint: If your server's network can't access google driver or dl.fbaipublicfiles.com, you can
step 1. download models and datasets from another machine, link are placed below
step 2. copy them to current folder
step 3. comment the download instructions below and then execute the remaining instructions
!

# download pre-processed multi-domain de-en dataset from Google driver
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/18TXCWzoKuxWKHAaCRgddd6Ub64klrVhV/view?usp=sharing
# download wmt19 de-en model from fairseq
wget -c https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.ffn8192.tar.gz


# create data-bin folder and unzip data-bin.zip to it
unzip data-bin.zip -d ../ 
rm -rf ../__MACOSX
rm data-bin.zip
# create pretrain-models folder and unzip wmt19.de-en.ffn8192.tar.gz to it
mkdir -p ../pretrain-models/wmt19.de-en
tar -xvf wmt19.de-en.ffn8192.tar.gz -C ../pretrain-models/wmt19.de-en
rm wmt19.de-en.ffn8192.tar.gz
