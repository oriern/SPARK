# SPARK
Data for the paper "The Power of Summary-Source Alignments" presented at ACL Findings 2024.

We denote this data suite as ``SPARK'', for Summary Proposition Alignment for Reconstructive Knowledgebases.

## Data generation ##

SuperPAL alignments of MultiNews train and val datasets can be found [here](https://drive.google.com/drive/folders/1JnRrdbENzBLpbae5ZIKmil1fuZhm2toc?usp=sharing).

To generate derived datasets (salience, clustering and generation) out of an alignment file use:
```
  python createSubDatasets.py -alignments_path <ALIGNMENTS_PATH>  -out_dir_path <OUT_DIR_PATH>
```
