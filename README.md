# BenchIMP: REQUIREMENTS and INSTALLATION instructions

This project contains the code related to the paper entitled "BenchIMP: A Benchmarking Approach for Incident Management Assessment".

## Requirements

The following packages and libraries are required to run the benchmark:
- pandas
- matplotlib
- sklearn
- pebble
- numpy
- scipy
- tabular_augmentation

Opting for the benchmark installation with Docker, it automatically installs these requirements in the container.

## Installation instructions

#### Using Docker container:

1) Build the docker container in the main folder of the project
2) Inside the container, move inside the "BenchIMP" directory
3) Configure the file src/config.py
4) Launch the file main.py

#### Without Docker container:

1) Install the requirements described above
2) Configure the file src/config.py
3) Launch the file main.py inside the BenchIMP directory

## Configuration

In the file BenchIMP/src/config.py the following main parameters can be configured (more parameters are available in the file):

- num_cores (the number of cores to use to run the benchmark, default is 1)
- ratio_step (the step of noising experiments from 1 to 99, default is 20)
- messing_step (the step of noising experiments from 1 to 99, default is 20)
- magnitude_step = (the step of magnitude experiment from 1 to 99, default is 20)
- perform_sampling (boolean to activate sampling, default is True)
- sampling_percentage (the percentage of entries to sample, default is 50, ignored if perform_sampling is False)
- in_memory = (if store file in hdd (False) or store all the data in_memory(True). This latter will make the computation much heavier, use it carefully. Default is False.)
