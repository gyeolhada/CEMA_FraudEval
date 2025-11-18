# cema/01_build_auxiliary.py
# ============================================
# CEMA 第一步：构建 auxiliary pool（辅助样本池）
# 说明：
#   - 从训练集 CSV 中筛选未标注样本（is_fraud 为空）
#   - 若未标注样本不足，则补充标注样本
#   - 对部分样本进行简单文本增强（随机打乱句子顺序）
# ============================================

import pandas as pd
import random, os, re

# 设置辅助样本池数量
AUX_NUM = 3000
# 训练数据路径
DATA_PATH = "../data/trainResult.csv"
# 输出辅助样本路径
SAVE_PATH = "../data/auxiliary/aux_texts.csv"

# 创建保存路径文件夹（若不存在）
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# 智能读取 CSV 文件，自动判断分隔符
def smart_read_csv(file_path):
    # 读取首行判断分隔符
    with open(file_path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline()
    if '\t' in first_line:
        sep = '\t'
    elif ';' in first_line:
        sep = ';'
    else:
        sep = ','

    # 使用 pandas 读取 CSV 文件
    df = pd.read_csv(file_path, sep=sep, engine="python", encoding="utf-8-sig")
    # 清理列名，去除 BOM 和空格
    df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
    return df

# 随机文本增强：将一句话按标点拆分后随机组合
def random_text_aug(text):
    # 按中文、英文标点拆分
    parts = re.split(r'[，。,\.；;]', text)
    # 去掉空字符串
    parts = [p.strip() for p in parts if len(p.strip()) > 0]
    if len(parts) <= 1:
        return text
    # 随机打乱顺序
    random.shuffle(parts)
    # 随机选取 1~全部 的句子组合为新文本
    out = "，".join(parts[:random.randint(1, len(parts))])
    return out

# 构建辅助样本池函数
def build_auxiliary_pool(train_csv, aux_num=3000):
    # 读取训练集
    df = smart_read_csv(train_csv)

    # 检查是否包含指定列
    if "specific_dialogue_content" not in df.columns:
        raise KeyError(f"未找到列 'specific_dialogue_content'，实际列名为：{df.columns.tolist()}")

    # 筛选未标注样本
    unlabeled_df = df[df['is_fraud'].isna()]
    # 筛选已标注样本
    labeled_df   = df[df['is_fraud'].notna()]

    # 转为文本列表
    unlabeled_texts = unlabeled_df['specific_dialogue_content'].dropna().astype(str).tolist()
    labeled_texts   = labeled_df['specific_dialogue_content'].dropna().astype(str).tolist()

    # 若未标注样本数量足够，则随机采样
    if len(unlabeled_texts) >= aux_num:
        aux_texts = random.sample(unlabeled_texts, aux_num)
    else:
        # 否则使用全部未标注样本，并从标注样本中补充
        aux_texts = unlabeled_texts.copy()
        remaining = aux_num - len(aux_texts)
        if remaining > 0 and len(labeled_texts) > 0:
            aux_texts += random.sample(labeled_texts, min(remaining, len(labeled_texts)))

    # 对部分样本进行随机文本增强
    aux_texts_aug = [random_text_aug(t) if random.random() < 0.5 else t for t in aux_texts]

    print(f" 构建 auxiliary pool 完成：未标注 {len(unlabeled_texts)} 条，总输出 {len(aux_texts_aug)} 条样本。")
    return aux_texts_aug

# 主程序入口
if __name__ == "__main__":
    # 构建辅助样本池
    aux_texts = build_auxiliary_pool(DATA_PATH, AUX_NUM)
    # 保存到 CSV 文件
    pd.DataFrame({"text": aux_texts}).to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f" 已保存到: {SAVE_PATH}")
