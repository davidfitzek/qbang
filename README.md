# Optimizing Variational Quantum Algorithms with qBANG: A Hybrid Approach to Tackle Flat Energy Landscapes

This repo contains the code for the optimizers introduced in the paper "Optimizing Variational Quantum Algorithms with qBANG: A Hybrid Approach to Tackle Flat Energy Landscapes" [[arXiv](UPDATE LINK)].


## Installation and use

setup conda environment 

conda create --name qbang_env python=3.10  
conda activate qbang


The package `qbang` is prepared such that it can be installed in development mode locally (such that any change in the code is instantly reflected). To install `qbang` locally, clone/download the repository and run the following command from the `qbang` folder:

```
pip install -e .
```

Feel free to reach out to me at dpfitzek@gmail.com if you face any issues.

For testing and verification of the software run:

```
pytest src/qbang/tests
```


## Background

![VQA](resources/figures/workflow.png "VQA")



If you find this repo useful for your research, please consider citing our paper:

```bibtex
@article{opt-vqa-with-qbang,
  title={Optimizing Variational Quantum Algorithms with qBANG: A Hybrid Approach to Tackle Flat Energy Landscapes},
  author={Fitzek, David and Jonsson, Robert S. and Dobrautz, Werner and Schäfer, Christian},
  journal={arXiv preprint arXiv:XXXX.XXXX},
  year={2023},
}
```

## References

ADD REFERENCES