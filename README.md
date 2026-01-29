# EHR-RAG: Retrieval-Augmented Prototype-Guided Foundation Model for Electronic Health Records

[[TOC]]

# Background

Electronic Health Records (EHR) contain rich and diverse longitudinal patient information that is commonly used in the development of predictive machine learning models. However, effectively leveraging historical patient data remains challenging due to long trajectory lengths, event heterogeneity, temporal irregularity, and varying relevance of past visits. In this repo, we introduce EHR-RAG, a new retrieval-augmented foundation model that dynamically integrates the most relevant patient history across diverse types of clinical events. 

![EHR-RAG](assets/main-figure.png)


# ENvironment Setup
To run this repo, you must install and run the libraries in the YAML file below.

```
conda env create -f environment.yml
conda activate ehr-rag
```

# Dataset
We conduct all experiments using [**MIMIC-IV**](https://physionet.org/content/mimiciv/3.1/) V3.1, a publicly available, de-identified critical care electronic health record (EHR) dataset.  
Access to the dataset requires credentialed approval from the data provider.  
Raw data are not included in this repository and needs to be downloaded.


# MEDS Format
We convert the raw EHR records into a standardized **event-oriented format** using publicly available tools based on the Medical [Event Data Standard (MEDS)](https://github.com/Medical-Event-Data-Standard).  
This conversion transforms the original tables into a consistent timeline representation, which serves as input for model training and evaluation.
We leverage the official MEDS export tool available for MIMIC-IV V3.1, [MIMIC_IV_MEDS](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS/tree/main/src/MIMIC_IV_MEDS)

# Experiments
We provide the code for submission purposes, Exact training and evaluation configuration files will be uploaded upon acceptance 
## Pretraining

To run pretraining experiments
```
python pretrain.py
```
## Baselines training (hyperparametrs tuning)

To run hyperparametres experiments
```
python wo_hparams_opt.py
```

## EHR-RAG training (hyperparametrs tuning)

To run hyperparametres tuning experiments
```
python w_hparams_opt.py
```

## Downstream eavaluation

To run hyperparametres experiments
```
python eval.py
```

# Citation

If you use our code, kindly cite our paper:
```
#TODO
```







