from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

# 1. 加载数据集
dataset = load_dataset("llamafactory/OpenR1-Math-94k", split="train")

# 2. 加载 Llama3 tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", trust_remote_code=True)

# 3. 定义函数：将 messages 拼成一个完整对话
def concat_messages(messages):
    """把多轮 message 拼接成单个字符串"""
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

# 4. 统计每条样本的 token 长度
token_lens = []
for sample in tqdm(dataset, desc="Tokenizing"):
    text = concat_messages(sample["messages"])
    n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    token_lens.append(n_tokens)

# 5. 输出平均长度
print(f"平均 token 长度: {np.mean(token_lens):.2f}")
print(f"中位数 token 长度: {np.median(token_lens):.2f}")
print(f"最大 token 长度: {np.max(token_lens)}")
