# 导入所需的库
from transformers import AutoTokenizer          # 用于加载预训练 tokenizer（如 DeepSeek）
import matplotlib.pyplot as plt                 # 用于绘图（例如直方图）
import pandas as pd                             # 用于数据处理和生成 CSV 文件
import json                                     # 用于读取 .jsonl 格式文件中的 JSON 行数据

# 设置中文显示（主要适用于 Windows 系统）
plt.rcParams['font.sans-serif'] = ['SimHei']    # 设置中文字体为 SimHei（黑体），防止中文乱码
plt.rcParams['axes.unicode_minus'] = False      # 正确显示负号（如 -1 而不是乱码）

# ✅ 加载 tokenizer（注意路径可替换为 huggingface 上的模型名）
tokenizer = AutoTokenizer.from_pretrained("./DeepSeek-V3-Base")  # 加载本地 DeepSeek 模型的 tokenizer

# ✅ 如果 tokenizer 没有设置 pad_token（用于补齐），则使用 eos_token（结束符）代替
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

# ✅ 打开并读取训练数据文件（.jsonl 格式，每行一个 JSON 对象）
with open("./fake_news/train_deepseek.jsonl", "r", encoding="utf-8") as f:
    # 逐行读取文件，并解析为 JSON 对象，仅保留包含 'text' 字段的行
    data = [json.loads(line) for line in f if "text" in line]

# ✅ 对每一条样本的 'text' 字段进行分词，统计其 token 数量
token_lengths = [len(tokenizer.tokenize(example["text"])) for example in data]

# ✅ 输出样本的 token 长度统计信息
print(f"样本数: {len(token_lengths)}")                                  # 输出总样本数
print(f"最大 token 数: {max(token_lengths)}")                          # 输出最大 token 数（最长文本）
print(f"最小 token 数: {min(token_lengths)}")                          # 输出最小 token 数（最短文本）
print(f"平均 token 数: {sum(token_lengths)/len(token_lengths):.2f}")   # 输出平均 token 数（保留两位小数）

# ✅ 绘制 token 长度分布直方图
plt.figure(figsize=(10, 5))                                            # 设置图像大小
plt.hist(token_lengths, bins=30, color="skyblue", edgecolor="black")  # 绘制直方图，设置颜色和边框
plt.xlabel("Token 数")                                                # 设置 X 轴标签
plt.ylabel("样本数量")                                                # 设置 Y 轴标签
plt.title("train_deepseek.jsonl 样本的 token 长度分布")                # 设置图标题
plt.grid(True)                                                         # 添加网格线
plt.tight_layout()                                                     # 自动调整布局避免遮挡
plt.show()                                                             # 显示图像

# ✅ 将 token 数量统计结果保存为 CSV 文件
df = pd.DataFrame({
    "样本编号": list(range(1, len(token_lengths)+1)),    # 样本编号从 1 开始
    "Token 数量": token_lengths                         # 对应每条样本的 token 数
})
df.to_csv("token_length_stats.csv", index=False)         # 保存为 CSV 文件，不包含行索引
