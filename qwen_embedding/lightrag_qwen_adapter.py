"""
LightRAG Qwen embedding bmodel 适配器
为 qwen_embedding.py 提供 LightRAG 兼容的接口
保持原始文件不变，仅通过适配器实现集成
"""

import sys
import os
import numpy as np
import torch
from typing import List, Optional
import asyncio
from functools import wraps

try:
    from qwen_embedding import load_model, QwenEmbedding
except ImportError as e:
    print(f"Error importing qwen_embedding: {e}")
    print(f"Please ensure qwen_embedding.py is available in the current directory")
    raise

class LightRAGQwenAdapter:
    """
    LightRAG 兼容的 Qwen embedding bmodel 适配器
    
    这个类将 qwen_embedding.py 中的 QwenEmbedding 包装成 LightRAG 期望的格式
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 batch_size: int = 1,
                 device: str = "tpu"):
        """
        初始化适配器
        
        参数:
        model_path: bmodel 文件路径
        tokenizer_path: 分词器文件路径  
        batch_size: 批处理大小
        device: 设备类型
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.device = device
        self._model: Optional[QwenEmbedding] = None
        self._model_loaded = False
    
    def _ensure_model_loaded(self):
        """延迟加载模型，避免初始化时的性能开销"""
        if not self._model_loaded:
            print("正在加载 Qwen embedding bmodel...")
            self._model = load_model(
                device=self.device,
                batch_size=self.batch_size,
                model_path=self.model_path,
                tokenizer_path=self.tokenizer_path
            )
            self._model_loaded = True
            print("Qwen embedding bmodel 加载完成")
    
    def _get_sentence_embedding(self, text: str, token_embeddings: torch.Tensor) -> np.ndarray:
        """
        从 token embeddings 提取句子级别的嵌入
        
        参数:
        text: 原始文本
        token_embeddings: token 级别的嵌入 [1, 128, 1024]
        
        返回:
        np.ndarray: 句子级别的嵌入 [1024]
        """
        # 获取 token ids 以计算注意力掩码
        tokens = self._model.tokenize(text)  # [1, 128]
        attention_mask = (tokens != 0).float()  # [1, 128]
        
        # 计算加权平均（忽略 padding token）
        valid_embeddings = token_embeddings * attention_mask.unsqueeze(-1)  # [1, 128, 1024]
        
        # 计算平均值
        valid_token_count = attention_mask.sum(dim=1, keepdim=True)  # [1, 1]
        sentence_embedding = valid_embeddings.sum(dim=1) / valid_token_count  # [1, 1024]
        
        return sentence_embedding.squeeze(0).numpy()  # [1024]
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        为文本列表生成嵌入向量（同步版本）
        
        参数:
        texts: 文本列表
        
        返回:
        np.ndarray: 嵌入矩阵，形状为 [batch_size, embedding_dim]
        """
        self._ensure_model_loaded()
        
        embeddings = []
        
        for text in texts:
            # 使用原始模型进行编码
            token_embeddings = self._model.encode_text(text)  # [1, 128, 1024]
            
            # 获取句子级别的嵌入（使用平均池化）
            sentence_embedding = self._get_sentence_embedding(text, token_embeddings)
            embeddings.append(sentence_embedding)
        
        return np.array(embeddings)  # [batch_size, 1024]
    
    async def aembed_texts(self, texts: List[str]) -> np.ndarray:
        """
        异步版本的文本嵌入函数
        
        参数:
        texts: 文本列表
        
        返回:
        np.ndarray: 嵌入矩阵
        """
        # 在事件循环中运行同步函数
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts)


# 全局适配器实例（单例模式）
_global_adapter: Optional[LightRAGQwenAdapter] = None


def get_qwen_adapter(model_path: Optional[str] = None,
                    tokenizer_path: Optional[str] = None,
                    batch_size: int = 1,
                    device: str = "tpu") -> LightRAGQwenAdapter:
    """
    获取全局适配器实例（单例模式）
    
    参数:
    model_path: bmodel 文件路径
    tokenizer_path: 分词器文件路径
    batch_size: 批处理大小
    device: 设备类型
    
    返回:
    LightRAGQwenAdapter: 适配器实例
    """
    global _global_adapter
    
    if _global_adapter is None:
        _global_adapter = LightRAGQwenAdapter(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            batch_size=batch_size,
            device=device
        )
    
    return _global_adapter


# 为 LightRAG 提供的主要接口函数
def get_lightrag_embedding_func(model_path: Optional[str] = None,
                               tokenizer_path: Optional[str] = None,
                               batch_size: int = 1,
                               device: str = "tpu",
                               async_mode: bool = True):  # 默认为异步模式
    """
    获取 LightRAG 兼容的嵌入函数
    
    参数:
    model_path: bmodel 文件路径，如果为 None 则使用默认路径
    tokenizer_path: 分词器文件路径，如果为 None 则使用默认路径
    batch_size: 批处理大小
    device: 设备类型
    async_mode: 是否返回异步版本的函数
    
    返回:
    callable: LightRAG 兼容的嵌入函数
    """
    if async_mode:
        async def async_embed_func(texts: List[str]) -> np.ndarray:
            adapter = get_qwen_adapter(model_path, tokenizer_path, batch_size, device)
            return await adapter.aembed_texts(texts)
        return async_embed_func
    else:
        def sync_embed_func(texts: List[str]) -> np.ndarray:
            adapter = get_qwen_adapter(model_path, tokenizer_path, batch_size, device)
            return adapter.embed_texts(texts)
        return sync_embed_func


def test_adapter():
    """
    测试适配器功能
    """
    print("测试 LightRAG Qwen embedding bmodel 适配器...")
    
    # 测试文本
    test_texts = [
        "这是第一个测试文本，用于验证 Qwen embedding bmodel 的功能。",
        "This is the second test text for embedding verification.",
        "测试中文和英文混合的文本 with mixed content for comprehensive testing."
    ]
    
    try:
        # 获取嵌入函数（同步版本用于测试）
        embed_func = get_lightrag_embedding_func(async_mode=False)
        
        # 测试嵌入
        print(f"输入文本数量: {len(test_texts)}")
        embeddings = embed_func(test_texts)
        
        print(f"嵌入矩阵形状: {embeddings.shape}")
        print(f"嵌入维度: {embeddings.shape[1]}")
        print(f"第一个文本的嵌入前10个值: {embeddings[0][:10]}")
        
        # 验证输出格式
        assert embeddings.shape[0] == len(test_texts), "批次大小不匹配"
        assert embeddings.shape[1] == 1024, "嵌入维度不正确"
        assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64, "数据类型不正确"
        
        print("✅ 适配器测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 适配器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_adapter_async():
    """
    测试适配器的异步功能
    """
    print("测试 LightRAG Qwen embedding bmodel 适配器（异步版本）...")
    
    test_texts = [
        "异步测试文本一",
        "Async test text two",
        "混合异步测试 mixed async test"
    ]
    
    try:
        # 获取异步嵌入函数
        async_embed_func = get_lightrag_embedding_func(async_mode=True)
        
        # 测试异步嵌入
        embeddings = await async_embed_func(test_texts)
        
        print(f"异步嵌入矩阵形状: {embeddings.shape}")
        print(f"异步嵌入维度: {embeddings.shape[1]}")
        
        print("✅ 异步适配器测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 异步适配器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行同步测试
    sync_success = test_adapter()
    
    # 运行异步测试
    async_success = asyncio.run(test_adapter_async())
    
    if sync_success and async_success:
        print("\n🎉 所有测试通过！适配器可以与 LightRAG 集成使用。")
    else:
        print("\n❌ 测试失败，请检查配置和依赖。")