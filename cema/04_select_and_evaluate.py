# cema/04_select_and_evaluate.py
# ============================================
# CEMA 第四步：对抗样本筛选与评估模块
# ============================================

"""
CEMA欺诈对话检测复现 - 第4步：对抗样本筛选与评估模块（中文注释版）

功能：
1. 读取测试集及候选改写数据（来自03步骤的输出 test_candidates.csv）
2. 加载多个替代模型（substitute models）模拟CEMA集成判断
3. 可选加载或训练“受害者模型”(victim model)
4. 对每条测试样本：
   - 计算原文和候选改写在替代模型上的预测差异
   - 按概率差 + 语义相似度评分选择最终对抗样本
   - 检查语义相似度过滤语义偏差大的改写
   - method_tag标注候选来源
5. 使用 victim 模型测试原文与对抗文本准确率（
6. 输出原始准确率 / 对抗准确率 / 攻击成功率(ASR)
7. 保存完整结果到 data/test_with_cema_selected.csv
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments, pipeline
)
import csv
csv.field_size_limit(10**7)

# ==================== 路径配置 ==================== #
DATA_DIR = "../data"
TEST_CSV = os.path.join(DATA_DIR, "testResult.csv")               # 测试集 CSV
TEST_CAND_CSV = os.path.join(DATA_DIR, "test_candidates.csv")     # 候选改写 CSV
OUT_CSV = os.path.join(DATA_DIR, "test_with_cema_selected.csv")   # 输出结果 CSV
SUB_DIR = "../models/substitute"                                   # 替代模型目录
VICTIM_DIR = "../models/victim"                                    # 受害者模型目录

# ==================== 模型与参数 ==================== #
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 用于语义相似度计算
SUB_DEVICE = 0 if torch.cuda.is_available() else -1    # 替代模型设备
VICTIM_DEVICE = 0 if torch.cuda.is_available() else -1 # victim 模型设备

SIM_THRESHOLD = 0.8   # 原文与候选语义相似度阈值
TOPK_CAND = 8          # 候选数量上限
USE_VICTIM = True      # 是否使用 victim 模型进行预测

# ================================================= #
# 一、数据加载函数
# ================================================= #
def load_test_candidates(test_csv=TEST_CAND_CSV, fallback_test_csv=TEST_CSV):
    """
    加载测试数据及候选改写
    """
    if os.path.exists(test_csv):
        try:
            df = pd.read_csv(
                test_csv,
                engine="python",
                sep=None,
                quoting=csv.QUOTE_NONE,
                on_bad_lines="skip"
            ).fillna("")
        except Exception as e:
            print(f"[WARN] 自动推断分隔符失败，回退到逗号分隔读取。错误信息: {e}")
            df = pd.read_csv(
                test_csv,
                sep=",",
                engine="python",
                quoting=csv.QUOTE_NONE,
                on_bad_lines="skip"
            ).fillna("")

        # ---- 解析候选 JSON 列 ----
        if "candidates_list" in df.columns:
            def parse_list(x):
                try:
                    j = json.loads(x)
                    if isinstance(j, list):
                        return [c["text"] if isinstance(c, dict) and "text" in c else c for c in j]
                    return [x]
                except:
                    return [x] if x else []
            df["candidates"] = df["candidates_list"].apply(parse_list)
        else:
            df["candidates"] = [[] for _ in range(len(df))]

        return df

    else:
        # 回退到原始测试集
        df = pd.read_csv(fallback_test_csv, sep=None, engine="python").fillna("")
        df["candidates"] = df["specific_dialogue_content"].apply(lambda x: [x])
        return df


# ================================================= #
# 二、模型加载函数
# ================================================= #
def load_substitute_pipelines(sub_dir=SUB_DIR, device=SUB_DEVICE):
    """
    加载替代模型 pipeline 列表
    """
    subs = []
    if not os.path.exists(sub_dir):
        raise FileNotFoundError(f"未找到替代模型目录: {sub_dir}")
    for name in sorted(os.listdir(sub_dir)):
        path = os.path.join(sub_dir, name)
        if os.path.isdir(path):
            try:
                tok = AutoTokenizer.from_pretrained(path, use_fast=True)
                model = AutoModelForSequenceClassification.from_pretrained(path)
                pipe = pipeline("text-classification", model=model, tokenizer=tok, device=device)
                subs.append(pipe)
                print(f" 已加载替代模型: {path}")
            except Exception as e:
                print(f" 加载失败 {path}: {e}")
    if len(subs) == 0:
        raise RuntimeError("未加载到任何替代模型，请检查路径。")
    return subs

def load_or_train_victim(victim_dir=VICTIM_DIR, train_csv="../data/trainResult.csv", device=VICTIM_DEVICE):
    """
    加载或训练 victim 模型
    """
    if os.path.exists(victim_dir) and len(os.listdir(victim_dir)) > 0:
        # 已存在模型，直接加载
        tok = AutoTokenizer.from_pretrained(victim_dir)
        model = AutoModelForSequenceClassification.from_pretrained(victim_dir)
        pipe = pipeline("text-classification", model=model, tokenizer=tok, device=device)
        print(f" 已加载受害者模型: {victim_dir}")
        return pipe

    # 未检测到模型，自动训练
    print(" 未检测到 victim 模型，将自动训练模型...")
    df = pd.read_csv(train_csv, sep=None, engine='python').fillna("")
    labeled = df[df['is_fraud'].notna()]
    if len(labeled) < 10:
        raise RuntimeError("带标签数据过少，无法训练 victim 模型。")

    # 标签归一化
    def norm_label(x):
        x = str(x).lower().strip()
        if x in ['true','1','t','yes','y']: return 1
        if x in ['false','0','f','no','n']: return 0
        try: return int(x)
        except: return 0

    labeled.columns = labeled.columns.str.strip().str.replace('\ufeff', '', regex=True)
    labeled['label_bin'] = labeled['is_fraud'].apply(norm_label)

    texts = labeled['specific_dialogue_content'].astype(str).tolist()
    labels = labeled['label_bin'].tolist()

    model_name = "IDEA-CCNL/Erlangshen-Longformer-110M"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    enc = tok(texts, truncation=True, padding=True, max_length=1024)

    # 构造 Dataset
    class DS(torch.utils.data.Dataset):
        def __init__(self, e, y): self.e, self.y = e, y
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k,v in self.e.items()}
            item["labels"] = torch.tensor(self.y[i])
            return item
    ds = DS(enc, labels)

    # 训练参数
    args = TrainingArguments(
        output_dir=victim_dir, num_train_epochs=3,
        per_device_train_batch_size=16, learning_rate=3e-5, save_total_limit=1
    )
    Trainer(model=model, args=args, train_dataset=ds).train()
    model.save_pretrained(victim_dir)
    tok.save_pretrained(victim_dir)
    print(f" Victim 模型训练完成并保存至: {victim_dir}")
    return pipeline("text-classification", model=model, tokenizer=tok, device=device)

# ================================================= #
# 三、辅助函数
# ================================================= #
def predict_with_pipeline_list(pipelines, texts):
    """
    使用替代模型列表预测文本标签，返回 numpy array
    """
    preds = []
    for p in pipelines:
        out = p(texts, truncation=True, padding=True)
        labels = []
        for o in out:
            lab = o.get('label', '')
            if isinstance(lab, str) and lab.startswith("LABEL_"):
                num = int(lab.split("_")[-1])
            else:
                num = 1 if "1" in lab or "true" in lab.lower() else 0
            labels.append(num)
        preds.append(labels)
    return np.array(preds)

def cosine_similarity(embedder, text_a, texts_b):
    """
    计算文本 a 与文本列表 b 的余弦相似度
    """
    a_emb = embedder.encode([text_a])[0]
    b_embs = embedder.encode(texts_b)
    sims = (b_embs @ a_emb) / (np.linalg.norm(b_embs, axis=1) * np.linalg.norm(a_emb) + 1e-9)
    return sims

# ================================================= #
# 四、主流程
# ================================================= #
def main():
    # ---------------- 数据加载 ----------------
    df = load_test_candidates()
    df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=True)
    print(f"共载入测试样本 {len(df)} 条")

    # ---------------- 模型加载 ----------------
    subs = load_substitute_pipelines()      # 替代模型
    if USE_VICTIM:
        victim = load_or_train_victim()     # victim 模型
    embedder = SentenceTransformer(EMB_MODEL)  # 语义嵌入模型

    results = []
    cache_preds = {}  # 预测缓存，避免重复计算

    # ---------------- 对抗样本筛选 ----------------
    print(" 正在执行多候选对抗筛选与评估（增强版）...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        orig = str(row['specific_dialogue_content']).strip()
        candidates = row['candidates'][:TOPK_CAND] if isinstance(row['candidates'], list) else [orig]

        if not candidates:
            results.append((orig, orig, 0, 0, 0, "none"))
            continue

        # 计算语义相似度
        sims = cosine_similarity(embedder, orig, candidates)
        high_sim = [(c,s) for c,s in zip(candidates,sims) if s >= SIM_THRESHOLD]
        low_sim  = [(c,s) for c,s in zip(candidates,sims) if 0.6 <= s < SIM_THRESHOLD]
        valid = high_sim + low_sim[:TOPK_CAND//2]

        # 原文在替代模型上的预测
        if orig not in cache_preds:
            cache_preds[orig] = predict_with_pipeline_list(subs, [orig])[:, 0]
        orig_preds = cache_preds[orig]

        # 候选最优选择
        best_cand, best_score, best_sim, best_p_c, best_method = orig, -1, 0, 0, "none"
        for cand, sim in valid:
            if cand not in cache_preds:
                cache_preds[cand] = predict_with_pipeline_list(subs, [cand])[:, 0]
            cand_preds = cache_preds[cand]

            diff = np.sum(cand_preds != orig_preds)
            p_c = diff / len(subs)             # 转移概率
            score = 0.6 * p_c + 0.4 * sim      # 综合评分

            method_tag = "03_generated" if cand in (row.get('candidates', []) or []) else "other"

            if score > best_score:
                best_cand, best_score, best_sim, best_p_c, best_method = cand, score, sim, p_c, method_tag

        results.append((orig, best_cand, best_sim, best_p_c, best_score, best_method))

    # ---------------- 保存筛选结果 ----------------
    df['cema_selected_adv'] = [r[1] for r in results]
    df['cema_selected_sim'] = [r[2] for r in results]
    df['num_changed_models'] = [r[3] for r in results]
    df['transfer_score'] = [r[4] for r in results]
    df['method_tag'] = [r[5] for r in results]

    # 文本清理
    def clean_text(t):
        t = str(t).replace("\n"," ").replace("\r"," ").strip()
        t = t.replace("音频内容：","").replace("left:","").replace("right:","")
        return t
    df["specific_dialogue_content"] = df["specific_dialogue_content"].apply(clean_text)
    df["cema_selected_adv"] = df["cema_selected_adv"].apply(clean_text)

    # ---------------- victim 模型预测 ----------------
    if USE_VICTIM:
        print(" Victim 模型预测原文与对抗文本...")
        orig_preds_raw = victim(df["specific_dialogue_content"].tolist())
        adv_preds_raw  = victim(df["cema_selected_adv"].tolist())

        def extract_label(pred):
            if isinstance(pred,list) and len(pred)>0: pred=pred[0]
            label = str(pred.get("label","")).lower()
            return 1 if ("1" in label or "fraud" in label or "positive" in label or "true" in label) else 0

        df["orig_pred"] = [extract_label(p) for p in orig_preds_raw]
        df["adv_pred"]  = [extract_label(p) for p in adv_preds_raw]

        if 'is_fraud' in df.columns:
            def norm_label(x):
                x = str(x).lower().strip()
                if x in ['true','1','t','yes','y']: return 1
                if x in ['false','0','f','no','n']: return 0
                return None
            df['true_label'] = df['is_fraud'].apply(norm_label)
            valid = df[df['true_label'].notna()]
            orig_acc = np.mean(valid['orig_pred']==valid['true_label'])
            adv_acc  = np.mean(valid['adv_pred']==valid['true_label'])
            correct_mask = valid['orig_pred']==valid['true_label']
            asr = np.mean(valid.loc[correct_mask,'adv_pred'] != valid.loc[correct_mask,'true_label'])

            print(f"\n=== 评估结果 ===")
            print(f"原始准确率: {orig_acc*100:.2f}%")
            print(f"对抗后准确率: {adv_acc*100:.2f}%")
            print(f"攻击成功率(ASR): {asr*100:.2f}%")

    # ---------------- 保存 CSV ----------------
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f" 已保存结果文件: {OUT_CSV}")

# ================================================= #
if __name__ == "__main__":
    main()
