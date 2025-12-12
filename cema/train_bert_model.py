import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_NAME = "hfl/chinese-bert-wwm-ext"
MODEL_NAME = "hfl/rbt6"
MODEL_DIR = "../models/victim_chinese_distillert"

MAX_LEN = 256
BATCH_SIZE = 16
LR = 3e-5

# -------- Label 处理 --------
def norm_label(x):
    x = str(x).lower().strip()
    if x in ["true", "1", "yes", "y"]: return 1
    if x in ["false", "0", "no", "n"]: return 0
    try: return int(x)
    except: return None


# -------- Dataset --------
class FraudDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            texts,
            max_length=MAX_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


# -------- 训练， --------
def train_once():
    print("读取训练数据...")
    df = pd.read_csv("../data/trainResult.csv").fillna("")
    df["label"] = df["is_fraud"].apply(norm_label)
    df = df[df["label"].notna()]

    texts = df["specific_dialogue_content"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    print(f"训练样本数量: {len(texts)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    dataset = FraudDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ------- RAINING LOOP -------
    model.train()
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # ------- 保存模型 -------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print("\n训练完成！已经保存模型到：", MODEL_DIR)


if __name__ == "__main__":
    train_once()
