"""
Qwen Embedding 模型推理模块
整合了模型加载、文本处理和推理功能
"""

import os
import time
import torch
import numpy as np
from typing import Union, List
from tokenizers import Tokenizer
from tpu_perf.infer import SGInfer


class EngineOV:
    """NPU推理引擎"""
    
    def __init__(self, model_path="", batch=1, device_id=0):
        # 如果环境变量中设置了device_id，则使用环境变量中的值
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])
            print(f">>>> device_id is in os.environ. and device_id = {device_id}")
        
        # 初始化SGInfer模型，指定模型路径、批处理大小和设备ID
        self.model = SGInfer(model_path, batch=batch, devices=[device_id])
    
    def __call__(self, args):
        # 记录开始时间
        start = time.time()
        
        # 处理输入参数
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
        
        # 将数据放入模型进行推理并获取结果
        task_id = self.model.put(*values)
        task_id, results, valid = self.model.get()
        
        return results


class QwenEmbedding:
    """Qwen Embedding 模型类"""
    
    def __init__(self, 
                 model_path: str = '/data/Qwen3_Embedding_0.6B_my_1684x_128_f32.bmodel',
                 tokenizer_path: str = "/data/Qwen_Embedding_0.6B_config/tokenizer.json",
                 batch_size: int = 1,
                 embed_dim: int = 1024,
                 transformer_width: int = 1024,
                 context_length: int = 128):
        
        self.context_length = context_length
        
        # 加载文本编码器
        self.text_encoder = EngineOV(model_path, batch=batch_size)
        
        # 加载中文分词器
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
    
    def tokenize(self, texts: Union[str, List[str]], truncate: bool = True) -> torch.Tensor:
        """
        中文文本分词
        
        参数:
        texts: 输入文本，可以是字符串或字符串列表
        truncate: 是否截断超长文本
        
        返回:
        torch.Tensor: 分词后的token张量
        """
        # 确保输入是列表格式
        if isinstance(texts, str):
            texts = [texts]
        
        # 使用中文分词器进行批量编码
        tokens_and_encodings = self.tokenizer.encode_batch(
            texts,
            add_special_tokens=True,  # 添加特殊token
            is_pretokenized=False,   # 输入未预分词
        )
        
        # 获取第一个编码的input_ids
        input_ids = tokens_and_encodings[0].ids
        
        # 检查长度是否超过上下文长度
        if len(input_ids) > self.context_length:
            if truncate:
                # 截断到指定长度
                input_ids = input_ids[:self.context_length]
            else:
                # 抛出运行时错误
                raise RuntimeError(f"Input {texts[0]} is too long for context length {self.context_length}")
        else:
            # 用0填充到指定长度
            input_ids += [0] * (self.context_length - len(input_ids))
        
        # 转换为张量并增加一个维度
        return torch.tensor(input_ids).unsqueeze(0)
    
    def encode_text(self, text: Union[str, torch.Tensor]) -> torch.Tensor:
        """
        编码文本为embedding向量
        
        参数:
        text: 输入文本（字符串）或已分词的张量
        
        返回:
        torch.Tensor: token embeddings [1, 128, 1024]
        """
        # 记录开始时间
        st_time = time.time()
        
        # 如果输入是字符串，先进行分词
        if isinstance(text, str):
            text = self.tokenize(text)
        
        # 准备输入数据
        input_ids = text.numpy().astype(np.int32)
        attention_mask = (text != 0).numpy().astype(np.int32)
        
        print(f"输入形状 - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
        
        # 使用文本编码器处理文本，需要提供文本和注意力掩码
        results = self.text_encoder([input_ids, attention_mask])
        
        # 根据bmodel信息，有两个输出：
        # output: token_embeddings_Mul, [1, 128, 1024], float32
        # output: sentence_embedding_Div, [1, 1024], float32
        token_embeddings = torch.from_numpy(results[0])  # [1, 128, 1024]
        sentence_embedding = torch.from_numpy(results[1])  # [1, 1024]
        
        # 打印文本编码时间
        print(f'====================== Text Encoding: {time.time() - st_time:.4f}s')
        print(f'Token embeddings shape: {token_embeddings.shape}')
        print(f'Sentence embedding shape: {sentence_embedding.shape}')
        
        # 返回token embeddings（与原代码保持一致）
        return sentence_embedding


def load_model(device: str = "tpu", 
               batch_size: int = 1,
               model_path: str = None,
               tokenizer_path: str = None) -> QwenEmbedding:
    """
    加载Qwen Embedding模型
    
    参数:
    device: 设备类型
    batch_size: 批处理大小
    model_path: 模型文件路径
    tokenizer_path: 分词器文件路径
    
    返回:
    QwenEmbedding: 模型实例
    """
    # 使用默认路径
    model_path = model_path 
    tokenizer_path = tokenizer_path or "/data/Qwen_Embedding_0.6B_config/tokenizer.json"
    
    # 创建Qwen_Embedding模型实例
    model = QwenEmbedding(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        batch_size=batch_size
    )
    
    print("============== Models are ready. ==============")
    return model


def process_chinese_text(model: QwenEmbedding, text: str) -> torch.Tensor:
    """
    处理中文文本并返回bmodel输出
    
    参数:
    model: Qwen模型实例
    text: 输入文本
    
    返回:
    torch.Tensor: 模型输出结果
    """
    print(f"输入文本: {text}")
    
    # 使用tokenizer处理文本
    input_ids = model.tokenize([text], truncate=True)
    
    print(f"Token IDs shape: {input_ids.shape}")
    
    # 调用模型进行推理
    result = model.encode_text(input_ids)
    
    print(f"bmodel输出形状: {result.shape}")
    print(f"bmodel输出 (前10个值): {result[0][:10]}")
    
    return result


def save_result_to_file(result: torch.Tensor, 
                       input_text: str, 
                       output_file_path: str = "./output.txt"):
    """
    将模型输出结果保存到文件
    
    参数:
    result: 模型输出张量
    input_text: 输入文本
    output_file_path: 输出文件路径
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("Qwen Embedding 模型输出结果\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"输入文本: {input_text}\n\n")
            f.write(f"输出张量形状: {result.shape}\n\n")
            
            # 保存完整的张量数据
            f.write("完整输出数据:\n")
            if len(result.shape) == 3:  # [1, 128, 1024]
                # 保存为二维数组格式，每行是一个token的embedding
                for i, token_embedding in enumerate(result[0]):
                    f.write(f"Token {i}: {token_embedding.tolist()}\n")
            else:
                f.write(f"{result.tolist()}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("数据统计信息:\n")
            f.write(f"最大值: {torch.max(result).item():.6f}\n")
            f.write(f"最小值: {torch.min(result).item():.6f}\n")
            f.write(f"平均值: {torch.mean(result).item():.6f}\n")
            f.write(f"标准差: {torch.std(result).item():.6f}\n")
        
        print(f"结果已保存到: {output_file_path}")
        
    except Exception as e:
        print(f"保存文件时出错: {e}")


def main():
    """主函数"""
    # 设置设备ID
    device = 0
    
    # 加载模型
    model = load_model(device="tpu", batch_size=1)
    
    # 从txt文档中读取文本进行处理
    txt_file_path = "/data/imitation/input.txt"
    
    try:
        if not os.path.exists(txt_file_path):
            print(f"文件不存在: {txt_file_path}")
            # 如果没有输入文件，使用默认测试文本
            test_text = "这是一个测试文本，用于验证Qwen Embedding模型的功能。"
            print(f"使用默认测试文本: {test_text}")
            result = process_chinese_text(model, test_text)
            save_result_to_file(result, test_text)
        else:
            with open(txt_file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                
            if content:
                print(f"从文件读取到文本，长度: {len(content)} 字符")
                result = process_chinese_text(model, content)
                # 保存结果到文件
                save_result_to_file(result, content)
            else:
                print("文件内容为空")
                
    except Exception as e:
        print(f"处理时出错: {e}")


if __name__ == '__main__':
    main()