#!/bin/bash
DATA_DIR="../data/MSMARCO"
mkdir -p $DATA_DIR && cd $DATA_DIR

echo "=== Downloading MS-MARCO Passage Ranking Dataset ==="
echo "Source: UWaterloo mirror (official Azure blob no longer public)"

# 1. collectionandqueries.tar.gz 包含 collection.tsv + queries.dev.small.tsv 等
if [ ! -f "collection.tsv" ] || [ ! -f "queries.dev.small.tsv" ]; then
    echo "Downloading collectionandqueries.tar.gz (~1GB)..."
    wget -c https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/data/collectionandqueries.tar.gz
    if [ $? -ne 0 ]; then
        echo "UWaterloo mirror failed, trying Dropbox..."
        wget -c -O collectionandqueries.tar.gz "https://www.dropbox.com/s/9f54jg2f71ray3b/collectionandqueries.tar.gz?dl=1"
    fi
    tar -xzf collectionandqueries.tar.gz && rm collectionandqueries.tar.gz
else
    echo "collection.tsv and queries already exist, skipping."
fi

# 2. Dev qrels (passage-level 评测参考)
if [ ! -f "qrels.dev.small.tsv" ]; then
    echo "Downloading qrels.dev.small.tsv..."
    wget -c https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/data/qrels.dev.small.tsv
    if [ $? -ne 0 ]; then
        echo "qrels.dev.small.tsv may already be included in collectionandqueries.tar.gz"
    fi
else
    echo "qrels.dev.small.tsv already exists, skipping."
fi

echo ""
echo "=== Downloaded files ==="
ls -lh *.tsv 2>/dev/null
