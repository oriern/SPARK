# SPARK
Data for the paper "The Power of Summary-Source Alignments" presented at ACL Findings 2024.

We denote this data suite as ``SPARK'', for Summary Proposition Alignment for Reconstructive Knowledgebases.

## Derive SPARK datasets for train and val sets##

1. Download MultiNews train and dev datasets [here](https://github.com/Alex-Fabbri/Multi-News).
2. Parse the data:
```
  python parseMultiNews.py
```
3. Cluster the data and add query:
```
  python add_query.py
```
4.   SuperPAL alignments of MultiNews train and val datasets can be found [here](https://drive.google.com/drive/folders/1JnRrdbENzBLpbae5ZIKmil1fuZhm2toc?usp=sharing).

5. Generate derived datasets out of an alignment file use:
```
  python createSubDatasets.py -alignments_path <ALIGNMENTS_PATH>  -out_dir_path <OUT_DIR_PATH> -doc_path <PARSED_DOCUMENT_DIR_PATH> -summ_path <PARSED_SUMMARY_DIR_PATH>
```
