# Artifact-based Domain Generalization of Skin Lesion Models

This is the official repository of the paper "Artifact-based Domain Generalization of Skin Lesion Models", accepted at the ISIC Workshop @ ECCV 2022.

## Reproducing our results:

### Data

The training/validation/test data is passed through two specific parameters:

`{train|val|test}_csv` : is a csv containing the list of samples on the set. On folder `trap_sets`, we include all the csvs used in the work, which are based on ISIC 2019.

`root_dir`: is the directory where samples can be found. Alternatively, it is possible to include the full path on the csvs mentioned above.


The confounder annotation is at the [file](isic_inferred_wocarcinoma.csv), which is referenced in the code at https://github.com/alceubissoto/artifact-generalization-skin/blob/ce89fef63733f6251db75f04a01f55d3770d5c0e/data/skin_dataset.py#L27

For running the out-of-distribution evaluation, include images on the folder `datasets`. They are loaded at https://github.com/alceubissoto/artifact-generalization-skin/blob/ce89fef63733f6251db75f04a01f55d3770d5c0e/train.py#L277-L280

### General

- The code is fully prepared to use wandb, but it is disabled by default. 
- We make use of the sacred library, allowing the organization of the results by folder, according to the name passed in the parameter `exp_name`.
- We make available the [script](run.sh) to run all trainings and evaluations.

