import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("/deltadisk/guestnju/tpu/Qwen3-Embedding-0.6B-ONNX-lower")

# 加载ONNX模型
onnx_model_path = "/deltadisk/guestnju/tpu/Qwen3-Embedding-0.6B-ONNX-lower/model.onnx"
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# 检查模型输入输出信息
print("Model inputs:")
for inp in session.get_inputs():
    print(f"  - {inp.name}: {inp.shape} ({inp.type})")

print("Model outputs:")
for out in session.get_outputs():
    print(f"  - {out.name}: {out.shape} ({out.type})")

def encode_texts(texts, text_type="document", max_length=512):
    """使用ONNX模型编码文本"""
    embeddings = []
    
    for text in texts:
        # 根据类型添加前缀
        if text_type == "query":
            text = f"query: {text}"
        
        # 分词
        inputs = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='np'
        )
        
        # 生成position_ids
        position_ids = np.arange(inputs['input_ids'].shape[1]).reshape(1, -1).astype(np.int64)
        
        # 准备ONNX模型输入
        onnx_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64),
            'position_ids': position_ids
        }
        # 运行推理
        outputs = session.run(None, onnx_inputs)
        
        # 获取last_hidden_state
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, 1024]
        
        # 进行mean pooling
        attention_mask = inputs['attention_mask']
        mask_expanded = np.expand_dims(attention_mask, axis=-1)
        
        # 使用attention mask进行平均池化
        embedding = np.sum(last_hidden_state * mask_expanded, axis=1) / np.sum(mask_expanded, axis=1)
        
        # L2归一化
        embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
        embeddings.append(embedding[0])  # 去除batch维度
    
    return np.array(embeddings)

def cosine_similarity(a, b):
    """计算余弦相似度"""
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True).T)

# 测试数据
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# 编码查询和文档
print("Encoding queries...")
query_embeddings = encode_texts(queries)
print(f"Query embeddings shape: {query_embeddings.shape}")

print("Encoding documents...")
document_embeddings = encode_texts(documents)
print(f"Document embeddings shape: {document_embeddings.shape}")

# 计算相似度
similarity = np.dot(query_embeddings, document_embeddings.T)
print("\nSimilarity matrix:")
print(similarity)
# 输出也是比较抽象：
# Similarity matrix:
# [[0.88721131 0.557092  ]
#  [0.46133045 0.74679787]]