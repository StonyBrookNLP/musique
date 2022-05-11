# <h2 align="center"> :musical_note: MuSiQue: Multi-hop Questions via Single-hop Question Composition </h2>

Repository for our TACL 2022 paper "[MuSiQue: Multi-hop Questions via Single-hop Question Composition](https://arxiv.org/pdf/2108.00573.pdf)"

## Installation

```
conda create -n musique python=3.8 -y && conda activate musique
```

## Data

MuSiQue is distributed under a [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

**Usage Caution:** If you're using any of our seed single-hop datasets ([squad](https://arxiv.org/abs/1606.05250), [T-REx](https://hadyelsahar.github.io/t-rex/paper.pdf), [Natural Questions](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1f7b46b5378d757553d3e92ead36bda2e4254244.pdf), [MLQA](https://arxiv.org/pdf/1910.07475.pdf), [zero shot re](https://arxiv.org/pdf/1706.04115.pdf)) in anyway (e.g. pretraining on them), note that MuSiQue was built by composing questions from the seed datasets, and so single-hop questions used in its dev/test sets may occur in training sets of these seed datasets. To help avoid this leakage, we're releasing *single-hop* questions and ids that are used in MuSiQue dev/test sets. Once you download the data below, it'll be in `data/dev_test_singlehop_questions_v1.0.json`. If you use our seed single-hop dataset/s in any way, please make sure to avoid using these single-hop questions.

To download MuSiQue, either run the following script or download it manually from [here](https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing).

```
bash download_data.sh
```

The result will be stored in `data/` directory. It contains (i) train, dev and test sets of `MuSiQue-Ans` and `MuSiQue-Full`, (ii) single-hop questions and ids from source datasets (squad, natural questions, trex, mlqa, zerore) that are part of dev or test of MuSiQue.


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
