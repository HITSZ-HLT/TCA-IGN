# TCA-IGN

> The official implementation for the conference of the ECAI 2023 paper *Do Topic and Causal Consistency Affect Emotion Cognition? A Graph Interactive Network for Conversational Emotion Detection*.

<img src="https://img.shields.io/badge/Venue-ECAI--23-blue" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Issues-Welcome-red">

## Requirements
* Python 3.7.11
* PyTorch 1.8.0
* Transformers 4.1.1
* CUDA 11.1

## Preparation
Download [**datasets**](https://drive.google.com/file/d/1I47mbbHSc2vkNXZs_NjRng-7cglqDdSd/view?usp=drive_link) and save them in ./data.

Download [**topics**](https://drive.google.com/file/d/1hiV9SWM5hh_KgLJSXWjjvmSfAXtbMhOB/view?usp=drive_link) and save them in ./data.

## Training & Evaluation
You can train the models with the following codes:

For IEMOCAP: ```python run.py --dataset IEMOCAP --gnn_layers 4 --lr 0.0005 --batch_size 16 --epochs 30 --dropout 0.2 ```

For MELD: ```python run.py --dataset MELD --lr 0.00001 --batch_size 64 --epochs 70 --dropout 0.1 ```

For EmoryNLP: ```python run.py --dataset EmoryNLP --lr 0.00005 --batch_size 32 --epochs 100 --dropout 0.3 ```

## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@inproceedings{tu2023topic,
  title={Do topic and causal consistency affect emotion cognition? a graph interactive network for conversational emotion detection},
  author={Tu, Geng and Liang, Bin and Lyu, Xiucheng and Gui, Lin and Xu, Ruifeng},
  booktitle={In The 26th European Conference on Artificial Intelligence (ECAI’23)},
  pages={2362--2369},
  year={2023}
}
```

## Credits
The code of this repository partly relies on [DAG-ERC](https://github.com/shenwzh3/DAG-ERC) and I would like to show my sincere gratitude to the authors behind these contributions.

