# Substitute Models (substitute/)

These models are trained to approximate the black-box behavior of the victim classifier.

Training process:
- Extract sentence embeddings using `all-MiniLM-L6-v2`
- Generate pseudo-labels using KMeans clustering
- Train several small Chinese RoBERTa models, each with different sampling and initialization
- Store all models to build an ensemble
  
Each sub_x directory contains all necessary files for one substitute model, and a total of 8 such models are included in the ensemble.

---

# Victim Model (victim/)

This directory stores the final classifier being attacked.

In this reproduction:
- Chinese Longformer 110M (fine-tuned) is used as the victim model.
`04_select_and_evaluate.py` loads this model to compute:
- Clean accuracy  
- Adversarial accuracy  
- Attack Success Rate (ASR)

---

# Note on Missing Checkpoint Files

The files model.safetensors and optimizer.pt inside victim/checkpoint2694/ directory are not included in this repository because their sizes exceed GitHub’s upload limits.

Similarly, the model.safetensors file in three victim directories is also too large to be uploaded.

However, the Chinese Longformer 110M (fine-tuned) victim model can be fully retrained and reproduced using the script: cema/04_select_evaluate.py and the BERT-base(hfl/chinese-bert-wwm-ext) and DistilBERT(hfl/rbt6) also can be fully retrained and reproduced using the script: cema/train_bert_model.py.

This script will automatically train and generate the required checkpoints, though the training process may take a considerable amount of time depending on your hardware.
