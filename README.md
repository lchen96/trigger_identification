# Trigger Identification

## Description

This repository contains the code and data for the paper ["A Progressive Framework for Role-Aware Rumor Resolution"](https://aclanthology.org/2022.coling-1.242/) .

We extended the PHEME dataset with message-level annotation. The [system demo](http://fudan-disc.com/project/annotation/propagation/demo.html) shows the annotation guidelines and interface. The dataset and the corresponding description can be download [here](http://fudan-disc.com/data/PHEME_trigger.zip).

We provide the fold information (cascade ID) in the `fold.json` file. We also record the checkpoint with best performance in the `save` directory.

## Environment

- Python 3.8
- Nvidia GeForce RTX3090
- CUDA 11.2

## Python Packages

```
emoji==1.7.0
nltk==3.6.5
numpy==1.21.3
pandas==1.3.4
scikit-learn==1.0.1
tokenizers==0.12.1
torch==1.10.1+cu111
torchaudio==0.10.1+rocm4.1
torchmetrics==0.6.0
torchvision==0.11.2+cu111
transformers==4.18.0
wandb==0.12.16
```

## Implementation

Please first download the data contained in `trigger.csv`, and modify the `data_file` in `config.py` file.

Then `cd` into the directory, and run the code for `RANDOM` validation:

```bash
python main.py --val_type RANDOM --gpu 0
```

or run the code for `LOEO` validation:

```bash
python main.py --val_type LOEO --test_event 0 --gpu 0
```

## Citation



```
@inproceedings{chen-etal-2022-progressive,
  title = "A Progressive Framework for Role-Aware Rumor Resolution",
  author = "Chen, Lei  and
   Li, Guanying  and
   Wei, Zhongyu  and
   Yang, Yang  and
   Zhou, Baohua  and
   Zhang, Qi  and
   Huang, Xuanjing",
  booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
  year = "2022",
  publisher = "International Committee on Computational Linguistics",
  url = "https://aclanthology.org/2022.coling-1.242",
  pages = "2748--2758",
}
```
