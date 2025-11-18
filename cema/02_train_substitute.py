# cema/02_train_substitute.py
# ============================================
# CEMA 第二步：训练替代模型（Substitute Models）
# 说明：
#   - 使用第一步构建的 auxiliary pool
#   - 对文本进行 SBERT embedding 并进行聚类生成伪标签
#   - 基于聚类标签训练多个替代模型，用于模拟受害模型的黑盒行为
# ============================================

import os, random, numpy as np, pandas as pd, torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# ---------------- 1. 配置参数 ----------------
AUX_FILE = "../data/auxiliary/aux_texts.csv"   # 辅助样本 CSV 路径
SAVE_DIR = "../models/substitute/"            # 替代模型保存目录
os.makedirs(SAVE_DIR, exist_ok=True)          # 若目录不存在则创建

# 禁用 wandb 和 transformers 的警告
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # SBERT embedding 模型
BACKBONE = "bert-base-uncased"                         # 替代模型 backbone

# 聚类及训练相关参数
CLUSTERS = 4       # 聚类数量
W = 8              # 训练替代模型数量（ensemble）
FRAC = 0.7         # 每次训练随机抽样比例
BATCH = 8          # 训练 batch size
EPOCHS = 5         # 训练 epoch 数
LR = 3e-5          # 学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU/CPU 自动选择

print(" 当前设备：", DEVICE)
print(" Backbone:", BACKBONE)

# ---------------- 2. 聚类辅助文本 ----------------
def cluster_aux_texts(aux_csv, n_clusters=CLUSTERS):
    """
    功能：
        - 读取辅助文本 CSV
        - 使用 SBERT 生成文本向量
        - KMeans 聚类生成伪标签，用于替代模型训练
    """
    df = pd.read_csv(aux_csv)
    texts = df["text"].astype(str).tolist()
    print(f"Embedding {len(texts)} texts with SBERT...")

    # SBERT 文本向量化
    embedder = SentenceTransformer(EMB_MODEL, device=str(DEVICE))
    emb = embedder.encode(texts, show_progress_bar=True)

    print(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    df["cluster"] = kmeans.fit_predict(emb)  # 聚类标签作为伪标签
    print(" 聚类完成！")
    return df

# ---------------- 3. 训练替代模型 ----------------
def train_substitute(texts, labels, out_dir):
    """
    功能：
        - 使用 tokenizer 对文本编码
        - 训练文本分类替代模型（带可控 dropout / 轻噪声）
        - 评估 self-accuracy 并保存模型
    """
    # 3.1 tokenizer / model 初始化
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BACKBONE, num_labels=len(set(labels))
    ).to(DEVICE)

    # 3.2 可控 dropout 增强鲁棒性
    try:
        if hasattr(model.config, "hidden_dropout_prob"):
            model.config.hidden_dropout_prob = max(
                0.2, getattr(model.config, "hidden_dropout_prob", 0.1) + 0.15
            )
        if hasattr(model.config, "attention_probs_dropout_prob"):
            model.config.attention_probs_dropout_prob = max(
                0.2, getattr(model.config, "attention_probs_dropout_prob", 0.1) + 0.15
            )
    except Exception:
        pass

    # 3.3 文本编码
    enc = tokenizer(texts, truncation=True, padding=True, max_length=256)

    # 3.4 自定义 dataset
    class DS(torch.utils.data.Dataset):
        def __init__(self, e, y):
            self.e, self.y = e, y
        def __len__(self):
            return len(self.y)
        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k,v in self.e.items()}
            item["labels"] = torch.tensor(int(self.y[i]))
            return item

    ds = DS(enc, labels)

    # 3.5 训练参数配置
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=50,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),  # 支持半精度训练
        seed=random.randint(0, 10000),
        report_to=[]  # 关闭所有外部日志
    )

    # 3.6 Trainer 训练
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()

    # 3.7 self-accuracy 测试
    model.eval()
    with torch.no_grad():
        enc_eval = tokenizer(
            texts, truncation=True, padding=True, max_length=256, return_tensors="pt"
        ).to(DEVICE)
        logits = model(**enc_eval).logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        acc = accuracy_score(labels, preds)
        print(f" Substitute self-accuracy: {acc:.2%}")

    # 3.8 保存模型与 tokenizer
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f" 模型已保存至 {out_dir}")

# ---------------- 4. 主程序 ----------------
if __name__ == "__main__":
    # 4.1 聚类生成辅助伪标签
    df = cluster_aux_texts(AUX_FILE, CLUSTERS)

    # 4.2 训练 W 个替代模型（ensemble）
    for i in range(W):
        # 随机采样 FRAC 比例数据
        sample = df.sample(frac=FRAC, random_state=42 + i).reset_index(drop=True)
        sub_dir = os.path.join(SAVE_DIR, f"sub_{i}")
        print(f"\n 训练 substitute 模型 {i+1}/{W}，样本数：{len(sample)} ...")
        train_substitute(sample["text"].tolist(), sample["cluster"].tolist(), sub_dir)

    print("\n 全部替代模型训练完成！")
