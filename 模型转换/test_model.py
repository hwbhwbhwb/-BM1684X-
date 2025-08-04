# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("/deltadisk/guestnju/tpu/Qwen3-Embedding-0.6B")

queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Encode queries with prompt
query_embeddings = model.encode(queries, prompt_name="query", convert_to_tensor=True)
print("Query embedding shape:", query_embeddings.shape)  # ➤ (batch_size, hidden_dim)

# Encode documents
document_embeddings = model.encode(documents, convert_to_tensor=True)
print("Document embedding shape:", document_embeddings.shape)  # ➤ (batch_size, hidden_dim)

# 打印示例输入 tokens 的维度
tokenizer = model.tokenizer
inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
print("Input token ids shape (queries):", inputs['input_ids'].shape)  # ➤ (batch_size, seq_len)

inputs_doc = tokenizer(documents, return_tensors="pt", padding=True, truncation=True)
print("Input token ids shape (documents):", inputs_doc['input_ids'].shape)

# 计算相似度
similarity = model.similarity(query_embeddings, document_embeddings)
print("Similarity matrix:")
print(similarity)
