# cema/03_generate_candidates.py
# ============================================
# CEMA 第三步：生成多类型对抗候选句
# —— 攻击方式包含：同义词替换（MLM）+ 句子改写（T5）+ 逻辑插入（规则）
# ============================================

import os
import torch
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import jieba
import nltk

# 关闭 wandb 等非必要日志
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# NLTK 词典
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ---------------- 参数配置----------------
TEST_PATH = "../data/testResult.csv"
OUT_PATH = "../data/test_candidates.csv"

SIM_THRESHOLD = 0.8        # 相似度过滤阈值
NUM_CANDS = 60             # 每条样本最大候选数量
MAX_CHANGES = 12           # MaskedLM 最大替换 token 数
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 用于相似度筛选的嵌入模型

# T5 句子改写模型
PARA_MODEL = "uer/t5-base-chinese-cluecorpussmall"
MASKED_MODEL = "bert-base-chinese"  # 中文 MLM 模型

USE_BACKTRANSLATION = False  # 回译攻击开关

ROUNDS = 3                    # 每条文本重复攻击轮数（提高攻击多样性）
PARA_MAX_NEW_TOKENS = 128    # T5 最长生成长度

# T5 的采样策略（增强攻击性）
PARA_TOP_K = 250
PARA_TOP_P = 0.92
PARA_TEMP = 1.5

# MLM 替换策略
MLM_TOP_K = 40
SEED = 42

# 固定随机种子（保证可复现）
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------- 设备检测 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" 当前设备：{device}")

# ---------------- 加载模型 ----------------
print(" 加载模型中（句向量、T5、MLM）...")

# Sentence-BERT，用于计算语义相似度
embedder = SentenceTransformer(EMB_MODEL, device=str(device))

# T5 改写生成器
paraphraser = pipeline(
    "text2text-generation",
    model=PARA_MODEL,
    device=0 if torch.cuda.is_available() else -1
)

# MaskedLM（词级替换）模型
mlm_tokenizer = AutoTokenizer.from_pretrained(MASKED_MODEL)
mlm_model = AutoModelForMaskedLM.from_pretrained(MASKED_MODEL).to(device)
fill_mask = pipeline("fill-mask", model=mlm_model, tokenizer=mlm_tokenizer,
                     device=0 if torch.cuda.is_available() else -1)

print(" 模型加载完成。")

# ---------------- 工具函数 ----------------
def cosine_sim(a, b):
    """计算余弦相似度"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def filter_by_similarity(orig, cands, th=SIM_THRESHOLD, top_k=NUM_CANDS):
    """
    语义过滤函数：
    - 使用 Sentence-BERT 计算与原文相似度
    - 过滤掉相似度过低的对抗样本
    - 最终选出 top_k 个最相似的候选句
    """
    if not cands:
        return []

    texts = [orig] + cands
    emb = embedder.encode(texts, convert_to_numpy=True)

    orig_emb = emb[0]
    cand_embs = emb[1:]
    sims = [cosine_sim(orig_emb, c) for c in cand_embs]

    # 先按相似度排序
    paired = sorted(zip(cands, sims), key=lambda x: x[1], reverse=True)
    kept = [c for c, s in paired if s >= th]

    # 如果过滤后为空，则至少保留前几个避免攻击失败
    if not kept:
        kept = [c for c, s in paired[:min(3, len(paired))]]

    return kept[:top_k]


# ============================================================
# 攻击方法 1： MaskedLM 替换
# —— 词级扰动 + 噪声注入（口语副词、断句符号）
# ============================================================
NOISE_WORDS = ["其实", "大概", "似乎", "就像", "反正", "说实话", "据说", "有人认为"]
NOISE_SYMBOLS = ["…", "～", "——", "。", "！"]

def maskedlm_replacement_candidates(text, num_cands=12, max_changes=MAX_CHANGES, top_k=MLM_TOP_K):
    """
    词级替换攻击：
    - 随机选择若干字符位置
    - 使用 MaskedLM 预测替换词
    - 同时随机加入口语化噪声以提高扰动
    """
    tokens = list(jieba.cut(text))
    L = len(tokens)
    if L == 0:
        return []

    cands = set()
    indices = [i for i, w in enumerate(tokens) if len(w.strip()) > 0]

    attempts = num_cands * 4
    for _ in range(attempts):

        # 随机替换 K 个 token
        k = random.randint(1, min(max_changes, max(1, L)))
        pos_list = random.sample(indices, min(len(indices), k))

        new_tokens = tokens.copy()

        # 执行 MLM 替换
        for p in pos_list:
            masked = "".join(tokens[:p]) + mlm_tokenizer.mask_token + "".join(tokens[p+1:])
            try:
                preds = fill_mask(masked, top_k=top_k)
            except:
                continue

            cand_list = []
            for pred in preds:
                word = pred["token_str"].strip()
                if not word or word == tokens[p]:
                    continue
                cand_list.append(word)

            rep = random.choice(cand_list) if cand_list else tokens[p]
            new_tokens[p] = rep

        # 噪声增强：口语副词
        if random.random() < 0.35:
            insert_pos = random.randint(0, len(new_tokens))
            new_tokens.insert(insert_pos, random.choice(NOISE_WORDS))

        # 噪声增强：断句符
        if random.random() < 0.25:
            insert_pos = random.randint(0, len(new_tokens))
            new_tokens.insert(insert_pos, random.choice(NOISE_SYMBOLS))

        candidate = "".join(new_tokens).strip()
        if candidate and candidate != text:
            cands.add(candidate)

        if len(cands) >= num_cands:
            break

    return list(cands)


# ============================================================
# 攻击方法 2：模糊化 T5 改写
# —— 表达口语化、模糊化、绕，弱化原结构
# ============================================================
def paraphrase_candidates(text, n=6):
    """
    基于 T5 的句子改写：
    - 生成含“不确定性 / 推测性 / 模糊性”的语句
    - 通过高采样温度 + top-k/p 增强攻击多样性
    """
    prompt = (
        f"请将下面的句子重新表达成语义接近但更加模糊、带推测性、"
        f"表达方式有变化、不完全等价、更加口语化或带有不确定性的版本：{text}"
    )

    try:
        results = paraphraser(
            [prompt],
            num_beams=max(8, n),
            num_return_sequences=n,
            do_sample=True,            # 允许采样（增加随机性）
            top_k=PARA_TOP_K,
            top_p=PARA_TOP_P,
            temperature=PARA_TEMP,
            max_new_tokens=PARA_MAX_NEW_TOKENS
        )
    except:
        return []

    outs = []
    for r in results:
        if isinstance(r, dict):
            outs.append(r.get("generated_text", "").strip())
        elif isinstance(r, str):
            outs.append(r.strip())

    outs = [o for o in outs if o and o != text]
    return list(dict.fromkeys(outs))[:n]


# ============================================================
# 攻击方法 3：逻辑插入攻击
# —— 添加模糊推理、虚假前提、不确定性表达
# ============================================================
LOGIC_PATTERNS = [
    "据我观察，",
    "至少表面上看，",
    "大概来说，",
    "好像，",
    "似乎，",
    "也说不准，",
    "可能只是巧合，但，",
    "如果传闻是真的，",
    "假如情况有变，或许",
    "有人认为，",
    "从某种意义来说，",
    "理论上讲，",
    "一般来说，",
]

def logic_edit_candidates(text):
    """
    基于规则的逻辑改写：
    - 插入模糊逻辑前缀
    - 添加推测性后缀
    - 否定词替换为弱化表达
    """
    edits = []

    # 否定弱化（提升攻击性）
    if "不是" in text:
        edits.append(text.replace("不是", "不见得是"))
        edits.append(text.replace("不是", "未必算"))
    if "是" in text and "不是" not in text:
        edits.append(text.replace("是", "好像是"))

    # 前缀逻辑扰动
    for prefix in LOGIC_PATTERNS:
        edits.append(prefix + text)

    # 后缀模糊推测
    edits.append(text + "，但这一点似乎还需要再确定")
    edits.append(text + "，不过这也可能只是表面现象")

    return list(dict.fromkeys([e for e in edits if e != text]))


# ---------------- 主流程 ----------------
if __name__ == "__main__":
    print(f" 读取测试集：{TEST_PATH}")
    df = pd.read_csv(TEST_PATH)
    texts = df["specific_dialogue_content"].astype(str).tolist()

    candidates_all = []
    stats = {"before": [], "after": []}

    print(" 正在生成攻击性候选句...")
    for t in tqdm(texts, desc="Processing"):
        all_cands = set()

        # 多轮攻击叠加（增强多样性）
        for _ in range(ROUNDS):
            # 攻击 1：词级替换（MLM）
            all_cands.update(maskedlm_replacement_candidates(t, num_cands=NUM_CANDS // 2))

            # 攻击 2：句子改写（T5）
            all_cands.update(paraphrase_candidates(t, n=8))

            # 攻击 3：逻辑插入
            all_cands.update(logic_edit_candidates(t))

        all_cands = list(all_cands)
        stats["before"].append(len(all_cands))

        # 语义过滤（保证语义不偏离原句太多）
        filtered = filter_by_similarity(t, all_cands, th=SIM_THRESHOLD, top_k=NUM_CANDS)

        if not filtered:     # 若全部被过滤，则保留原文
            filtered = [t]

        stats["after"].append(len(filtered))
        candidates_all.append(filtered)

    # 保存结果
    df["candidates_list"] = [str(x) for x in candidates_all]
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\n 生成完成！")
    print(f" 筛选前平均候选数：{np.mean(stats['before']):.2f}")
    print(f" 筛选后平均候选数：{np.mean(stats['after']):.2f}")
