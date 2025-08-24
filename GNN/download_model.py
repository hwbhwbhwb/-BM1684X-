from sentence_transformers import SentenceTransformer

# 你在代码中使用的模型名称
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
# 你想将模型保存到的本地文件夹名称
SAVE_PATH = './local_embedding_model'

print(f"正在下载模型 '{MODEL_NAME}'...")

# 加载模型（这将从Hugging Face Hub下载）
model = SentenceTransformer(MODEL_NAME)

# 将模型完整地保存到本地路径
model.save(SAVE_PATH)

print(f"模型已成功下载并保存到文件夹: '{SAVE_PATH}'")