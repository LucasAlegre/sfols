# SFOLS

Code for the paper "Optimistic Linear Support and Successor Features as a Basis for Optimal Policy Transfer" at ICML 2022.

Paper: https://arxiv.org/abs/2206.11326

## Install

To install run:
```bash
git clone https://github.com/LucasAlegre/sfols
cd sfols
pip install -e .
```

## Run Experiments

### Deep Sea Treasure
```bash
python experiments/run_dst.py -algo SFOLS
```
usage: run_dst.py [-h] [-algo {SFOLS,WCPI,Random}]

### Four Room
```bash
python experiments/run_fourroom.py -algo SFOLS
```
usage: run_fourroom.py [-h] [-algo {SFOLS,WCPI,SIP,Random}]

### Reacher
```bash
python experiments/run_reacher.py -SFOLS    
```
usage: run_reacher.py [-h] [-algo {SFOLS,WCPI,Random}]

## Citing

```bibtex
@inproceedings{Alegre+2022,
    author = {Lucas N. Alegre and Ana L. C. Bazzan and Bruno C. da Silva},
    title = {Optimistic Linear Support and Successor Features as a Basis for Optimal Policy Transfer},
    booktitle = {Proceedings of the Thirty-ninth International Conference on Machine Learning},
    address = {Baltimore, MD},
    year = {2022}
}
```
