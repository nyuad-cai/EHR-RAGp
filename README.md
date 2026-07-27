# EHR-RAGp: Retrieval-Augmented Prototype-Guided Foundation Model for Electronic Health Records


# Background

Electronic Health Records (EHR) contain rich longitudinal patient information and are widely used in predictive modeling applications. However, effectively leveraging historical data remains challenging due to long trajectories, heterogeneous events, temporal irregularity, and the varying relevance of past clinical context. Existing approaches often rely on fixed windows or uniform aggregation, which can obscure clinically important signals. In this work, we introduce \texttt{EHR-RAGp}, a retrieval-augmented foundation model that dynamically integrates the most relevant patient history across diverse clinical event types. We propose a prototype-guided retrieval module that acts as an alignment mechanism and estimates the relevance of retrieved historical chunks with respect to a given prediction task, guiding the model towards the most informative context. Across multiple clinical prediction tasks, \texttt{EHR-RAGp} consistently outperforms state-of-the-art EHR foundation model and transformer-based baselines. Furthermore, integrating \texttt{EHR-RAGp} with existing clinical foundation models yields substantial performance gains. Overall, \texttt{EHR-RAGp} provides a scalable and efficient framework for leveraging long-range clinical context to improve downstream performance.

![EHR-RAG](assets/main-figure.png)


# Environment Setup
To run this repo, you must install and run the libraries in the YAML file below.

```
conda env create -f environment.yml
conda activate ehr-rag
```

# Dataset
We conduct all experiments using [**MIMIC-IV**](https://physionet.org/content/mimiciv/3.1/) V3.1, a publicly available, de-identified critical care electronic health record (EHR) dataset.  Access to the dataset requires approval from the data provider after completing the required training and signing the usage agreement.  Hence, Raw data are not included in this repository and need to be downloaded by the user.


# MEDS Format
We convert the raw EHR records into a standardized **event-oriented format** using publicly available tools based on the  [MedicalEvent Data Standard (MEDS)](https://github.com/Medical-Event-Data-Standard).  This conversion transforms the original tables into a consistent timeline representation, which serves as input for model training and evaluation. We leverage the official MEDS export tool available for MIMIC-IV V3.1, [MIMIC_IV_MEDS](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS/tree/main/src/MIMIC_IV_MEDS)

Official MEDS repository:

https://github.com/Medical-Event-Data-Standard

Original MIMIC-IV MEDS extraction pipeline:

https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS

Customized EHR-RAGp MEDS extraction pipeline:

https://anonymous.4open.science/r/MIMIC_IV_MEDS_EHRRAGP-ECBC

Follow the instructions present in the customized repository


# Data Preprocessing

After MEDS conversion, run the end-to-end preprocessing pipeline:

```bash
python preprocess.py
```

The script performs the full post-MEDS preprocessing workflow, including:

- copying the raw MEDS cohort into the local preprocessing workspace
- cleaning raw MEDS shards
- ICD-9 to ICD-10 diagnosis and procedure mapping
- medication name normalization
- microbiology event cleaning
- laboratory event filtering, metadata enrichment, and value normalization
- ICU event cleaning for chart, procedure, infusion, and fluid-output events
- removal of patients without hospital admissions
- removal of empty admissions
- outpatient and emergency boundary construction
- time-gap token insertion
- MEDS-Transforms outlier occlusion
- MEDS-Transforms numeric normalization
- conversion of normalized parquet shards into a HuggingFace Arrow dataset

# Vector Index Setup

We use [Facebook AI Similarity Search (FAISS)](https://faiss.ai/index.html) to build vector indices. 

Run 
```bash
python create_vdb_idx.py
```
**Note:**

The file is prepared to create vector indices for all chunking strategies. As we create single index per patienet stay (1 stay = 1 index.faiss file), each training example will have its own index. We recommedn using singularity overlay if indices creation is being performed on HPC to avoid exceeding file quota limits.


# Experiments

## Pretraining

To run pretraining experiments
```
python pretrain.py
```
## Baselines training (hyperparameters tuning)

To run hyperparameter experiments
```
python wo_hparams_opt.py
```

## EHR-RAGp training (hyperparameters tuning)

To run hyperparameter tuning experiments
```
python w_hparams_opt.py
```

## Downstream evaluation

To run hyperparameter experiments
```
python final_eval.py
```

# Citation

If you use our code, kindly cite our paper:
```
#TODO
```







