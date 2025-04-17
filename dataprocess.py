import pyreadr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
# 设置中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 简体中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 1. 加载数据
result = pyreadr.read_r('./fake_news/fake_news.rda')
df = result['fake_news']

# 2. 提取数值型特征
df['type'] = pd.Categorical(df['type']).codes
exclude_cols = ['title', 'text', 'url', 'authors', 'type']
features = df.drop(columns=[col for col in exclude_cols if col in df.columns])
numeric_features = features.select_dtypes(include=[np.number])

# 3. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_features)

# 4. 动态PCA：保留累计贡献率≥85%的主成分
pca = PCA(n_components=0.85)  # 自动选择组件数
X_pca = pca.fit_transform(X_scaled)
print(f"保留的主成分数量: {pca.n_components_}, 累计方差贡献率: {sum(pca.explained_variance_ratio_):.2%}")

# 5. 可视化
plt.figure(figsize=(15, 5))

# 5.1 方差贡献率图
plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7, label='单个主成分')
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), 'ro-', label='累计贡献率')
plt.axhline(y=0.85, color='k', linestyle='--', label='85% 阈值')
plt.xlabel('主成分序号')
plt.ylabel('方差贡献率')
plt.title('PCA 方差贡献率分析')
plt.legend()

# 5.2 前两个主成分的散点图
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['type'], alpha=0.6, cmap='viridis')
plt.xlabel('第一主成分 (%.2f%%)' % (pca.explained_variance_ratio_[0] * 100))
plt.ylabel('第二主成分 (%.2f%%)' % (pca.explained_variance_ratio_[1] * 100))
plt.title('前两个主成分分布 (颜色=标签)')
plt.colorbar(scatter, label='0=假新闻 | 1=真新闻')
plt.tight_layout()
plt.savefig('pca_visualization.png')  # 保存图像
plt.show()


# 6. 合并特征并保存数据集
df_embed = df[['title', 'text', 'type']].copy()
df_embed['aux_feature'] = [list(x) for x in X_pca]

train_df, test_df = train_test_split(df_embed, test_size=0.2, random_state=42)

def save_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in data.iterrows():
            input_text = f"标题: {row['title']}\nPCA 特征: {', '.join([f'{v:.4f}' for v in row['aux_feature']])}\n新闻内容: {row['text']}"
            json.dump({
                "text": f"请判断以下新闻是否真实（0=假新闻，1=真新闻），请只返回一个数字（0或1），不要多余内容，直接输出:\n{input_text}",
                "label": row['type']  # 假设已经是 0 或 1
            }, f, ensure_ascii=False)
            f.write('\n')

save_to_jsonl(train_df, './fake_news/train_deepseek.jsonl')
save_to_jsonl(test_df, './fake_news/test_deepseek.jsonl')
print("处理完成！结果已保存为 JSONL 文件和可视化图像。")