mkdir data

# wiki sentence aligned
curl -o sentence-aligned-v2.tar https://cs.pomona.edu/~dkauchak/simplification/data.v2/sentence-aligned.v2.tar.gz
tar -xvf sentence-aligned-v2.tar -C data
rm sentence-aligned-v2.tar

# wiki doc aligned
curl -o document-aligned-v2.tar https://cs.pomona.edu/~dkauchak/simplification/data.v2/document-aligned.v2.tar.gz
tar -xvf document-aligned-v2.tar -C data
rm document-aligned-v2.tar

# get glove embeddings and unpack
curl -o glove_6B.zip http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip -q glove_6B.zip
rm glove_6B.zip

# move to data directory
mkdir data/glove_6B
mv glove.6B.*.txt data/glove_6B
