## Installation
a. Create a conda virtual environment and activate it.

```shell
conda create -n env python=3.7 -y
conda activate env
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

c. Install other third-party libraries.

- Install [OpenSlide and openslide-python](https://pypi.org/project/openslide-python/).  


d. Run the code

- TCGA:

  ```
  python train_tcga.py --num_classes [1 for C16 and 2 for tcga] --dataset [C16/tcga] --agg no --feats_size [size of pre-computed features (1024)] --model [abmil/transmil] --dropout_rate [Patch Dropout rate] ---num_augments [10/20/30]
   
  ```