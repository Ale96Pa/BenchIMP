# BenchIMP: REQUIREMENTS and INSTALLATION instructions

## Abstract
In the current scenario, where cyber-incidents occur daily, an effective Incident Management Process (IMP) and its assessment have assumed paramount significance. While assessment models, which evaluate the risks of incidents, exist to aid security experts during such a process, most of them provide only qualitative evaluations and are typically validated in individual case studies, predominantly utilizing non-public data. This hinders their comparative quantitative analysis, incapacitating the evaluation of new proposed solutions and the applicability of the existing ones due to the lack of baselines. 

To address this challenge, we contribute a benchmarking approach and system, BenchIMP, to support the quantitative evaluation of IMP assessment models based on performance and robustness in the same settings, thus enabling meaningful comparisons. The resulting benchmark is the first one tailored for evaluating process-based security assessment models and we demonstrate its capabilities through two case studies using real IMP data and state-of-the-art assessment models. We publicly release the benchmark to help the cybersecurity community ease quantitative and more accurate evaluations of IMP assessment models.

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

## Cite this work

Alessandro Palma, Nicola Bartoloni, and Marco Angelini. 2024. BenchIMP: A Benchmark for Quantitative Evaluation of the Incident Management Process Assessment. In The 19th International Conference on Availability, Reliability and Security (ARES 2024), July 30--August 02, 2024, Vienna, Austria. ACM, New York, NY, USA 12 Pages. https://doi.org/10.1145/3664476.3664504


```
@inproceedings{10.1145/3664476.3664504,
  author = {Palma, Alessandro and Bartoloni, Nicola and Angelini, Marco},
  title = {BenchIMP: A Benchmark for Quantitative Evaluation of the Incident Management Process Assessment},
  year = {2024},
  isbn = {979-8-4007-1718-5/24/07},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3664476.3664504},
  doi = {10.1145/3664476.3664504},
  booktitle = {Proceedings of the 19th International Conference on Availability, Reliability and Security},
  numpages = {12},
  keywords = {Cybersecurity Benchmark, Incident Management, Cybersecurity Processes, Quantitative Assessment},
  location = {Vienna, Austria},
  series = {ARES '24}
}
```
