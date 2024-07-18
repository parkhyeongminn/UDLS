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

- Camelyon16

  ```
  parser.add_argument('--mDATA0_dir_train0', default='./camelyon/train.pkl', type=str)
  parser.add_argument('--mDATA0_dir_val0', default='./camelyon/val.pkl', type=str)
  parser.add_argument('--mDATA_dir_test0', default='./camelyon/mDATA_test.pkl', type=str)
   
  ```

- TCGA

  ```
  parser.add_argument('--mDATA0_dir_train0', default='./TCGA_LUNG_slide_unit/train', type=str)
  parser.add_argument('--mDATA0_dir_val0', default='./TCGA_LUNG_slide_unit/validation', type=str)
  parser.add_argument('--mDATA_dir_test0', default='./TCGA_LUNG_slide_unit/test', type=str)
   
  ```

