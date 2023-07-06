#!/usr/bin/env bash
# So far RNA has been tested only on linux systems

mkdir -p data/rfam_learn/{raw,test,train,validation}

cd data/
wget https://www.dropbox.com/s/cfhnkzdx4ciy7zf/rfam_learn.tar.gz?dl=1 -O rfam_learn.tar.gz
tar xf rfam_learn.tar.gz
rm -f rfam_learn.tar.gz