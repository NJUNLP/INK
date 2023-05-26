## :memo: INK: Injecting kNN Knowledge in Nearest Neighbor Machine Translation
Code for our ACL 2023 paper "INK: Injecting kNN Knowledge in Nearest Neighbor Machine Translation". 
Our code is highly inspired by [Adaptive kNN-MT](https://aclanthology.org/2021.acl-short.47.pdf). More details and guidance can be found in this repository: https://github.com/zhengxxn/adaptive-knn-mt.

### Requirements and Installation
* python >= 3.7
* pytorch >= 1.10.0
* faiss-gpu >= 1.7.3
* sacremoses == 0.0.41
* sacrebleu == 1.5.1
* fastBPE == 0.1.0

You can install this repository by
```shell
git clone git@github.com:OwenNJU/INK.git
cd INK 
pip install --editable ./
```
Note: Installing faiss with pip is not suggested. For stability, we recommand you to install faiss with conda

```shell
CPU version only:
conda install faiss-cpu -c pytorch

GPU version:
conda install faiss-gpu -c pytorch # For CUDA
```

### Base Model and Data
We use the winner model of WMT'19 German-English news translation tasks as the off-the-shelf NMT model in our experiments, which can be downloaded from this [site](https://github.com/facebookresearch/fairseq/blob/main/examples/wmt19/README.md).

We conduct experiments on four benchmark OPUS dataset. We directly use the preprocessed data released by [Zheng et al](https://aclanthology.org/2021.acl-short.47.pdf)., which can be downloaded from this [site](https://drive.google.com/file/d/18TXCWzoKuxWKHAaCRgddd6Ub64klrVhV/view).

### Scripts
Below we provide scripts to run INK system:
```shell
# training 
bash ./run_scripts/train.ink.sh

# inference
bash ./run_scripts/inference.ink.sh
```

### Citation
If you find this repository helpful, feel free to cite our paper:
```bibtex
@inproceedings{zhu2023ink,
    title = "INK: Injecting kNN Knowledge in Nearest Neighbor Machine Translation",
    author = "Zhu, Wenhao  and
      Xu, Jingjing  and
      Huang, Shujian  and
      Kong, Lingpeng  and
      Chen, Jiajun",
    booktitle = "Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)",
    year = "2023",
}
```