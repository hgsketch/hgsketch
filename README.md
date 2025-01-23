## HGSketch

This repository contains the source code of HGSketch.

## Environment Setup

To set up the environment, follow these steps:

1. Create a conda environment:
   ```bash
   conda create --name hgsketch python=3.9
   conda activate hgsketch

2. Install required dependencies:
   ```bash
   conda install pytorch=1.12.1
   conda install gudhi=3.10.1
   conda install scipy=1.13.1
   conda install scikit-learn=1.6.1
   pip install torch-geometric==2.6.1

## Usage

To run heterogeneous graph classification experiments, use the provided shell scripts. Each script corresponds to a heterogeneous graph dataset.
   ```bash
   sh run_Cuneiform.sh
   sh run_sr_ARE.sh
   sh run_DBLP.sh
   sh run_nr_BIO.sh
