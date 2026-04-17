#!/bin/bash
DATA_DIR="../data/SIFT1M"
mkdir -p $DATA_DIR && cd $DATA_DIR

echo "=== Downloading SIFT1M Dataset ==="
if [ ! -f "sift_base.fvecs" ]; then
    wget -c ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
    tar -xzf sift.tar.gz && mv sift/* . && rmdir sift && rm sift.tar.gz
fi
ls -lh *.fvecs *.ivecs 2>/dev/null
