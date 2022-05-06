# <h2 align="center"> :musical_note: MuSiQue: Multi-hop Questions via Single-hop Question Composition </h2>

Repository for our TACL 2022 paper "[MuSiQue: Multi-hop Questions via Single-hop Question Composition](https://arxiv.org/pdf/2108.00573.pdf)"

## Installation

```
conda create -n musique python=3.8 -y && conda activate musique
```

## Data

MuSiQue is distributed under a [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

To get it, you can run the following script or download it manually from [here](https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing).

```
bash download_data.sh
```

The result will be stored in `data/` directory. It contains (i) train, dev and test sets of `MuSiQue-Ans` and `MuSiQue-Full`, (ii) single-hop questions and ids from source datasets (squad, natural questions, trex, mlqa, zerore) that are part of dev or test of MuSiQue.


**DISCLAIMER about Train-Test Leakage**

MuSiQue is built by composing seed questions from 5 existing single-hop datasets. Hence, the dev and test multihop questions often contain constituent single-hop questions that are part of the *training* sets of the seed single-hop datasets. In our train-test splits, we've ensured single-hop questions don't overlap between train and dev/test. So if you're just training/finetuning your model on our splits, you don't have to worry about it.

However, if you're doing any kind of additional pretraining of the models on some of our seed single-hop datasets, we ask the users to not explictly use single-hop questions that are in MuSiQue dev or test set. We also ask the users to not develop any method to solve MuSiQue, that directly looks up question-answers pairs from our seed single-hop datasets. To help you follow this guideline, we've released dev+test single-hop questions and their ids to help and locate them in original single-hop datasets, if needed. You can find it in `data/` directory


## Predictions

We're releasing the model predictions (in official format) for 4 models on dev sets of `MuSiQue-Ans` and `MuSiQue-Full`. To get it, you can run the following script or download it manually from [here](https://drive.google.com/file/d/1XZocqLOTAu4y_1EeAj1JM4Xc1JxGJtx6/view?usp=sharing).

```
bash download_predictions.sh
```


## Evaluation

You can use `evaluate_v1.0.py` to evaluate your predictions against ground-truths. For eg.:

```
python evaluate_v1.0.py predictions/musique_ans_v1.0_dev_end2end_model_predictions.jsonl data/musique_ans_v1.0_dev.jsonl
```

## Leaderboard

We'll release the leaderboard shortly.


## Citation

If you use this in your work, please cite use:

```
@article{trivedi2021musique,
  title={{M}u{S}i{Q}ue: Multihop Questions via Single-hop Question Composition},
  author={Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  journal={Transactions of the Association for Computational Linguistics},
  year={2022}
  publisher={MIT Press}
}
```
