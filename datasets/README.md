# Datasets
Download all datasets:
1. [COCO](https://cocodataset.org/#download): download the 2017 version (images and annotations) of the detection dataset.
2. [Visual Genome](https://visualgenome.org/api/v0/api_home.html): download the 2017 version (images and annotations) of the detection dataset.
3. [Open Images](https://storage.googleapis.com/openimages/web/download_v7.html): download the version v4 (images and annotations) of the detection dataset.

Then, structure the `datasets` folder as follow:
```
Concept-Conditioned-Object-Detector
|-- datasets
    |-- coco
        |-- annotations
            |-- instances_train2017.json            
            |-- instances_val2017.json              (our coco testset)
        |-- train2017
        |-- valid2017
        |-- test2017
    |-- visual_genome
        |-- annotations
            |-- visual_genome_train.json
            |-- visual_genome_val.json
            |-- visual_genome_test.json
        |-- images
    |-- OpenImagesDataset                           (optional)
        |-- annotations
            |-- openimages_v4_train_bbox.json
            |-- openimages_v4_val_bbox.json
            |-- openimages_v4_test_bbox.json
        |-- train
        |-- val
        |-- test
```

Subsequently it is necessary:
1. to convert the OpenImages annotations to the COCO json format;
2. to generate the COCO splits used for model tuning.
3. to generate the new datasets that include the concepts.

## OpenImages Format Conversion
The original OpenImages annotations contained in the `OpenImagesDataset` folder, are generate in COCO json format using the following process:
1. Convert the original dataset format to the COCO json format using the following repository: [openimages2coco](https://github.com/drigoni/openimages2coco). The results should be placed in the `OpenImagesDataset->annotations` folder.
2. However, some images are incorrectly sized and need to be corrected using: `python fix_OpenImagedDataset.py`. The results are reported in the folder `OpenImagesDataset->annotations` with the suffix `_fixSize`.
3. Delete all the previous files and remove the `_fixSize` extension from the newly generated ones.


## Generation of the Tuning COCO Dataset
Follow the commands to use to generate the `Tuning COCO` splits:
```
mkdir -p datasets/tuning_coco/annotations/
python make_tuning_dataset.py  --coco_dataset ./datasets/coco/annotations/instances_train2017.json --n_valid_examples 5000
```

After this command, the ``tuning_coco`` folder with its annotations should be generated.


## Generation of the Datasets with Concepts
Follow the commands to use to generate the new datasets with cocnepts:

```
# creating folders
cd datasets/
mkdir -p datasets/tuning_coco/annotations/
mkdir -p datasets/concept_coco/annotations/
mkdir -p datasets/concept_tuning_coco/annotations/
mkdir -p datasets/concept_visual_genome/annotations/
mkdir -p datasets/concept_OpenImagesDataset/annotations/

# COCO with concepts
python make_concept_dataset.py  --coco_dataset ./datasets/coco/annotations/instances_{SPLIT}2017.json \
                                --coco2concepts ./concept/coco_to_synset.json \
                                --dataset_name concept_coco \
                                --unique true --level {DEPTH_VALUE}  --type {TYPE}
python make_concept_dataset.py  --coco_dataset ./datasets/tuning_coco/annotations/tuning_instances_{SPLIT}2017.json \
                                --coco2concepts ./concept/coco_to_synset.json \
                                --dataset_name concept_tuning_coco \
                                --unique true --level {DEPTH_VALUE}  --type {TYPE}
# VISUAL GENOME with concepts
python make_concept_dataset.py  --coco_dataset ./datasets/visual_genome/annotations/visual_genome_{SPLIT}.json \
                                --coco2concepts ./concept/vg_to_synset.json \
                                --dataset_name concept_visual_genome \
                                --unique true --level {DEPTH_VALUE}  --type {TYPE}

# OPENIMAGES with concepts
python make_concept_dataset.py  --coco_dataset ./datasets/OpenImagesDataset/annotations/openimages_v4_{SPLIT}_bbox.json \
                                --coco2concepts ./concept/oid_to_synset.json \
                                --dataset_name concept_OpenImagesDataset \
                                --unique true --level {DEPTH_VALUE}  --type {TYPE}
```
Where:
* `SPLITS` refers to one of the following choices `[train, val, test]`. COCO does not have the `test` set.
* `TYPE` refers to one of the following choices `[all, subset]` which refers to the case in qhich all the concepts are considered or just the focused dataset is conddiered.
* `DEPTH_VALUE` refers to the depth value to consider in sampling WordNet descendants.





## Final Structure
The final `datasets` structure should be the following:
```
Concept-Conditioned-Object-Detector
|-- datasets
    |-- coco
        |-- annotations
            |-- instances_train2017.json
            |-- instances_val2017.json              (our coco testset)
        |-- train2017
        |-- valid2017
        |-- test2017
    |-- tuning_coco
        |-- annotations
            |-- tuning_instances_train2017.json     (our coco training set)
            |-- tuning_instances_val2017.json       (our coco validation set)
    |-- visual_genome
        |-- annotations
            |-- visual_genome_train.json
            |-- visual_genome_val.json
            |-- visual_genome_test.json
        |-- images
    |-- OpenImagesDataset
        |-- annotations
            |-- openimages_v4_train_bbox.json
            |-- openimages_v4_val_bbox.json
            |-- openimages_v4_test_bbox.json
        |-- train
        |-- val
        |-- test
        
    |-- concept_coco
        |-- annotations
            |-- instances_train2017_all.json
            |-- instances_train2017_subset.json
            |-- instances_val2017_all.json
            |-- instances_val2017_subset.json
            |-- instances_val2017_all_depth0.json
            |-- instances_val2017_subset_depth0.json
            |-- instances_val2017_all_depth2.json
            |-- instances_val2017_subset_depth2.json
            |-- instances_val2017_all_depth3.json
            |-- instances_val2017_subset_depth3.json
            |-- instances_val2017_all_depth4.json
            |-- instances_val2017_subset_depth4.json
    |-- concept_tuning_coco
        |-- annotations
            |-- tuning_instances_train2017_all.json
            |-- tuning_instances_train2017_subset.json
            |-- tuning_instances_val2017_all.json
            |-- tuning_instances_val2017_subset.json
    |-- concept_visual_genome
        |-- annotations
            |-- visual_genome_train_all.json
            |-- visual_genome_train_subset.json
            |-- visual_genome_val_all.json
            |-- visual_genome_val_subset.json
            |-- visual_genome_test_all.json
            |-- visual_genome_test_subset.json
    |-- concept_OpenImagesDataset
        |-- annotations
            |-- openimages_v4_train_bbox_all.json
            |-- openimages_v4_val_bbox_all.json
            |-- openimages_v4_test_bbox_all.json
            |-- openimages_v4_train_bbox_subset.json
            |-- openimages_v4_val_bbox_subset.json
            |-- openimages_v4_test_bbox_subset.json
```


# Datasets
Follow the links to download all the annotations used to validate and test our models with concepts.
| Dataset | Link |  
|:--------|:----:|  
|tuning_coco                |[download](https://drive.google.com/file/d/1FJ6GSIUYTlF8JsM5dyNb5uTXaGa_a3nz/view?usp=sharing)  |                            
|concept_coco               |[download](https://drive.google.com/file/d/1WaHXQMzFEDJwYAlP6GV2oGts0t1qDX7N/view?usp=sharing)  |                            
|concept_tuning_coco        |[download](https://drive.google.com/file/d/11vKID9OjngJw2XQkdXEqkeGPIqed1jXY/view?usp=sharing)  |                            
|concept_visual_genome      |[download](https://drive.google.com/file/d/1F7CjByYCB5epdMgJe-Qi6ad-YGIdKmNG/view?usp=sharing)  |                            
|concept_OpenImagesDataset  |[download](https://drive.google.com/file/d/1hi6DRs-ldNRWoxaLDRsY88-Td3t41XdW/view?usp=sharing)  |                            




