import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import torch

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

def encode_texts(texts, max_length=512):
    """使用ONNX模型编码文本"""
    embeddings = []
    
    for text in texts:
        # 对于查询，添加查询前缀
        if text in queries:
            text = f"query: {text}"
        
        # 分词
        inputs = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='np'
        )
        
        # 准备ONNX模型输入 - 修改为int64类型
        onnx_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }
        
        # 运行推理
        outputs = session.run(None, onnx_inputs)
        
        # 根据模型输出信息，有两个输出：
        # - token_embeddings: [batch_size, sequence_length, 1024]
        # - sentence_embedding: [batch_size, 1024]
        
        # 直接使用sentence_embedding，这已经是句子级别的嵌入
        if len(outputs) >= 2:
            embedding = outputs[1]  # sentence_embedding
        else:
            # 如果只有一个输出，使用token_embeddings并进行pooling
            token_embeddings = outputs[0]
            attention_mask = inputs['attention_mask']
            mask_expanded = np.expand_dims(attention_mask, axis=-1)
            embedding = np.sum(token_embeddings * mask_expanded, axis=1) / np.sum(mask_expanded, axis=1)
        
        # 归一化
        embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
        embeddings.append(embedding[0])  # 去除batch维度
    
    return np.array(embeddings)

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
# 自己从safetensors转换成onnx模型的输出：
# [[0.7632519  0.32346308]
#  [0.15970892 0.6293224 ]]