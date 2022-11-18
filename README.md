# Concept-Conditioned Object Detectors
This repository contains the code used to generate the results reported in the paper: [TODO]() \
Some of our code is based on [DynamicHead](https://github.com/microsoft/DynamicHead). Thanks!

<!---
### Citation

```BibTeX
@InProceedings{Dai_2021_CVPR,
    author    = {Dai, Xiyang and Chen, Yinpeng and Xiao, Bin and Chen, Dongdong and Liu, Mengchen and Yuan, Lu and Zhang, Lei},
    title     = {Dynamic Head: Unifying Object Detection Heads With Attentions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {7373-7382}
}
```
-->

## Dependencies
This project uses the `conda` environment.
```
conda create -n ccobj pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.1 torchtext=0.11.0 -c pytorch -c conda-forge
conda activate ccobj
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
python -m pip install -e ./
pip install nltk
pip install numpy
pip install setuptools==59.5.0  # https://stackoverflow.com/questions/70520120/attributeerror-module-setuptools-distutils-has-no-attribute-version
pip install wandb
```

<!---
In the `root` folder you can find the `.yml` file for the configuration of the `conda` environment and also the `.txt` files for the `pip` environment. 
```
conda create -f env.yml
conda activate ccobj
python -m pip install -e DynamicHead
pip install -r env.txt
```
-->

## Structure
The project is structured as follows: 
* `datasets`: contains datasets;
* `concept`: contains the mapping from the object detector categories to the WordNet concepts and the concepts vocabulary;
* `configs`: contains the configuration files to reproduce each experiment presented in the paper;
* `extra`: contains our main code;
* `results`: contains the results;
* `pretrained`: contains pre-trained checkpoints.


## Datasets
For more details on the datasets used and how to download the online hosted versions, refer to this guide: [README](./datasets/README.md)

## Concepts
As stated in the paper, we adopted pre-trained embeddings for each WordNet concepts using th code of this repository: [WordNet_Embeddings](https://github.com/drigoni/WordNet_Embeddings). \
The pre-trained weights must be placed in the `./concepts/` folder. Download them using the following link: [Download](https://drive.google.com/file/d/1e1R17scyKB_5uc6Op_td8oOOUo4fVIAN/view?usp=sharing). \
The final `concept` folder structure should be the following:
```
|-- concept
    |-- coco_to_synset.json
    |-- vg_to_synset.json
    |-- oid_to_synset.json
    |-- vocab.json
    |-- wn30_holE_500_150_0.1_0.2_embeddings.pickle
```

## Usage
### Train
Use the following command for training:
```
python train_net.py --config {TRAINING_CONFIG_FILE} 
```

If necessary, specify the number of machines and GPUs to use:
```
python train_net.py --config {TRAINING_CONFIG_FILE} --num-gpus 4 --num-machines 4 --dist-url tcp://login1234:19206 
```

### Evaluation
Two functions can be used to evaluate models.
The former performs model evaluation without applying the post-processing filtering before evaluation:
```
python train_net.py --config {TEST_CONFIG_FILE} --eval-only EVALUATOR_TYPE 'default'
```
The second performs post-processing filtering of the bounding boxes that the model predicts. 
In practice, the COCO evaluator loads all the bounding boxes predicted by the model, filters them, and then continues with the COCO evaluation.
```
python train_net.py --config {TEST_CONFIG_FILE} --eval-only EVALUATOR_TYPE 'postProcessing'
```
**NOTE**: Configuration files for testing presents a test set for evaluation that is not always correct. For this reason, during evaluation indicate the dataset to adopt:
```
python train_net.py --config {TEST_CONFIG_FILE} --eval-only EVALUATOR_TYPE 'default' DATASETS.TEST '("coco_2017_val_subset",)'
```
All the datasets can be seen in the file: `./extra/datasets.py'`.

# Model Zoo
Unfortunately, there is not enough space available to host all checkpoints.
However, we provide the models with the best performance.

<!---
| Config | Model | Backbone | Dataset | Weight |    
|:------:|:------|:---------|:--------|:------:|                              
|[cfg](TODO)    |RetinaNet              |ResNet-50      |COCO   |[weight](TODO)  |    
|[cfg](TODO)    |RetinaNet              |ResNet-101     |COCO   |[weight](TODO)  |    
|[cfg](TODO)    |DynamicHead            |ResNet-50      |COCO   |[weight](TODO)  |    
|[cfg](TODO)    |DynamicHead            |ResNet-101     |COCO   |[weight](TODO)  |
|[cfg](TODO)    |DynamicHead            |Swin-Tiny      |COCO   |[weight](TODO)  |
|[cfg](TODO)    |Concept RetinaNet      |ResNet-50      |COCO   |[weight](TODO)  |    
|[cfg](TODO)    |Concept RetinaNet      |ResNet-101     |COCO   |[weight](TODO)  |    
|[cfg](TODO)    |Concept DynamicHead    |ResNet-50      |COCO   |[weight](TODO)  |    
|[cfg](TODO)    |Concept DynamicHead    |ResNet-101     |COCO   |[weight](TODO)  |
|[cfg](TODO)    |Concept DynamicHead    |Swin-Tiny      |COCO   |[weight](TODO)  |
|[cfg](TODO)    |RetinaNet              |ResNet-50      |VG     |[weight](TODO)  |    
|[cfg](TODO)    |RetinaNet              |ResNet-101     |VG     |[weight](TODO)  |    
|[cfg](TODO)    |DynamicHead            |ResNet-50      |VG     |[weight](TODO)  |    
|[cfg](TODO)    |DynamicHead            |ResNet-101     |VG     |[weight](TODO)  |
|[cfg](TODO)    |DynamicHead            |Swin-Tiny      |VG     |[weight](TODO)  |
|[cfg](TODO)    |Concept RetinaNet      |ResNet-50      |VG     |[weight](TODO)  |    
|[cfg](TODO)    |Concept RetinaNet      |ResNet-101     |VG     |[weight](TODO)  |    
|[cfg](TODO)    |Concept DynamicHead    |ResNet-50      |VG     |[weight](TODO)  |    
|[cfg](TODO)    |Concept DynamicHead    |ResNet-101     |VG     |[weight](TODO)  |
|[cfg](TODO)    |Concept DynamicHead    |Swin-Tiny      |VG     |[weight](TODO)  |
-->

| Config | Model | Backbone | Dataset | Weight |    
|:------:|:------|:---------|:--------|:------:|  
|[cfg](configs/COCO/dh/swint/dh_swint_fpn_COCO_test.yaml)    |DynamicHead            |Swin-Tiny      |COCO   |[weight](https://drive.google.com/file/d/17wzANOd-pF9gn5rbajvgChf8ctaoCoYf/view?usp=sharing)  |                            
|[cfg](configs/COCO/dh/swint/dh_swint_fpn_COCO_concepts_test_cat.yaml)    |Concept DynamicHead    |Swin-Tiny      |COCO   |[weight](https://drive.google.com/file/d/1IpqZ9YOoQdTkTssaSKmTspdhwkFSvN_Q/view?usp=sharing)  |
|[cfg](configs/VG/dh/swint/dh_swint_fpn_VG_concepts_test_cat.yaml)    |DynamicHead            |Swin-Tiny      |VG     |[weight](https://drive.google.com/file/d/1TLLEU8FjR-u7xNkAd1LNKif_VwD0uOVe/view?usp=sharing)  |
|[cfg](configs/VG/dh/swint/dh_swint_fpn_VG_concepts_test_cat.yaml)    |Concept DynamicHead    |Swin-Tiny      |VG     |[weight](https://drive.google.com/file/d/1Z-cH-rvVbtfi2p5y3Zoffx5BZNoqDFlf/view?usp=sharing)  |


# Information
For any questions and comments, contact [TODO]().

# License
MIT
