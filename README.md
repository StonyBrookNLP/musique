# <h2 align="center"> :musical_note: MuSiQue: Multi-hop Questions via Single-hop Question Composition </h2>

This is the repository for our paper "[MuSiQue: Multi-hop Questions via Single-hop Question Composition](https://arxiv.org/pdf/2108.00573.pdf)". We'll release the more data and code here.

## Installation

```
conda create -n musique python=3.8 -y && conda activate musique
```

## Data

MuSiQue is distributed under a [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/). You can run the following script or download it manually from [here](https://drive.google.com/file/d/1QC6PRRnIWJ8Z1NBccO04K8CIfZBIdJ8p/view?usp=sharing).

```
bash download_data.sh
```

The result will be stored in `data/` directory. It contains train and dev sets of `MuSiQue-Ans` and `MuSiQue-Full` and their sample predictions files.


## Evaluation

You can use `evaluate_v0.1.py` to evaluate your predictions against ground-truths. For eg.:

```
python evaluate_v0.1.py data/musique_ans_v0.1_dev_sample_prediction.jsonl data/musique_ans_v0.1_dev.jsonl
```

## Citation

If you use this in your work, please cite use:

```
@article{trivedi2021musique,
  title={MuSiQue: Multi-hop Questions via Single-hop Question Composition},
  author={Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  journal={arXiv preprint arXiv:2108.00573},
  year={2021}
}
```
