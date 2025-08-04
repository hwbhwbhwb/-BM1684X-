# 导入操作系统模块，用于文件路径和环境变量操作
import os
# 禁用 tiktoken 缓存目录，避免网络请求

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONUNBUFFERED'] = '1'

import sys
import hashlib
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler

# 添加适配器模块路径
sys.path.append('/data/qwen_embedding')
# 添加 Qwen LLM 适配器路径
sys.path.append('/data/whisper-TPU_py/bmwhisper')

# 导入异步编程模块
import asyncio
# 导入检查模块，用于检查对象类型
import inspect
# 导入日志记录模块
import logging
# 导入日志配置模块
import logging.config
# 导入LightRAG核心组件：主类和查询参数类
from lightrag import LightRAG, QueryParam
# 导入OpenAI兼容的完成函数
from lightrag.llm.openai import openai_complete_if_cache
# 导入工具函数：嵌入函数类、日志器、详细调试设置
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
# 导入共享存储的管道状态初始化函数
from lightrag.kg.shared_storage import initialize_pipeline_status
# 导入PyTorch深度学习框架
import torch
import numpy as np
import time

# 导入我们的 LightRAG Qwen 适配器（用于embedding）
from lightrag_qwen_adapter import get_lightrag_embedding_func

# 导入 Qwen LLM 适配器（用于 LLM，通过 HTTP API）
from qwen_llm_adapter import qwen_llm_model_func, MODEL_CONFIG

# 导入简化的共享管理器用于状态检查
from shared_qwen_manager import simple_qwen

# 修改模型路径
MODEL_CONFIG["llm_model_path"] = "/data/qwen4btune_w4bf16_seq8192_bm1684x_1dev_20250721_195513.bmodel"
MODEL_CONFIG["config_path"] = "/data/LLM-TPU/models/Qwen3/python_demo/config"
MODEL_CONFIG["temperature"] = 0.5

# 定义工作目录路径
WORKING_DIR = "./result/biography_final"
# 在 WORKING_DIR 定义后添加输出目录配置
OUTPUT_DIR = "/data/mermaidRender/dist"

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    print(f"Creating output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR)
else:
    print(f"Output directory already exists: {OUTPUT_DIR},continue...")
    
# 如果工作目录不存在，则创建该目录
if not os.path.exists(WORKING_DIR):
    print(f"Setting directory not exist,creating working directory: {WORKING_DIR}")
    os.mkdir(WORKING_DIR)
else:
    print(f"Working directory already exists: {WORKING_DIR},continue...")

# 文件监控处理器类 - 修改版本
class QuestionFileHandler(FileSystemEventHandler):
    """问题文件变化监控处理器"""
    def __init__(self, callback, loop):
        self.callback = callback
        self.last_content_hash = None
        self.loop = loop  # 保存事件循环引用
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('ragQuestions.txt'):
            # 检查文件内容是否真的改变了
            try:
                with open(event.src_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                content_hash = hashlib.md5(content.encode()).hexdigest()
                
                if content_hash != self.last_content_hash:
                    self.last_content_hash = content_hash
                    # 使用 run_coroutine_threadsafe 安全地在事件循环中调度协程
                    asyncio.run_coroutine_threadsafe(
                        self.callback(event.src_path), 
                        self.loop
                    )
            except Exception as e:
                print(f"读取文件时出错: {e}")

# 定义日志配置函数
def configure_logging():
    """配置应用程序的日志记录"""
    # 重置现有的处理程序以确保干净的配置
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # 从环境变量获取日志目录路径，默认使用当前目录
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    # 构建日志文件的绝对路径
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "lightrag_qwen_bmodel_demo.log")
    )

    print(f"\nLightRAG Qwen embedding bmodel demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # 从环境变量获取日志文件最大大小和备份数量
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # 默认10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # 默认5个备份

    # 配置日志系统
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")

# 定义流式输出打印函数
async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)

# 检查模型状态的函数
def check_model_availability():
    """检查模型是否可用"""
    try:
        status = simple_qwen.get_status()
        print(f"模型状态检查: {status}")
        
        if status["local_model"] or status["external_model"]:
            print("✅ 检测到可用的 LLM 模型")
            return True
        else:
            print("⚠️ 未检测到可用的 LLM 模型")
            return False
    except Exception as e:
        print(f"❌ 模型状态检查失败: {e}")
        return False

# 检查 API 服务器状态
def check_api_server_status():
    """检查API服务器状态"""
    try:
        import requests
        response = requests.get("http://localhost:8899/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ API服务器可用: {status}")
            return True
        else:
            print(f"⚠️ API服务器响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API服务器不可用: {e}")
        return False

async def initialize_rag():
    """使用 Qwen embedding bmodel + HTTP LLM API 初始化 RAG 系统"""
    
    print("正在初始化 LightRAG with Qwen embedding bmodel + HTTP LLM API...")
    
    # 1. 检查 LLM 模型状态
    print("\n=== 检查 LLM 模型状态 ===")
    llm_available = check_model_availability()
    api_available = check_api_server_status()
    
    if not (llm_available or api_available):
        print("⚠️ 警告：LLM 模型和API服务器都不可用")
        print("请确保 sample_audio.py 正在运行")
        print("继续初始化，但LLM功能可能受限...")
    else:
        print("✅ LLM 模型通过 HTTP API 可用")
    
    # 2. 初始化 Embedding 模型（本地加载）
    print("\n=== 初始化 Embedding 模型 ===")
    try:
        qwen_embedding_func = get_lightrag_embedding_func(
            model_path='/data/Qwen3_Embedding_0.6B_my_1684x_128_f16.bmodel',
            tokenizer_path=None,
            batch_size=1,
            device="tpu",
            async_mode=True
        )
        print("✅ Embedding 模型初始化成功（本地bmodel）")
    except Exception as e:
        print(f"❌ Embedding 模型初始化失败: {e}")
        raise
    
    # 3. 创建 LightRAG 实例
    print("\n=== 创建 LightRAG 实例 ===")
    try:
        # 创建自定义 tokenizer 来替代 tiktoken
        class SimpleChineseTokenizer:
            """简单的中文分词器，避免使用 tiktoken"""
            
            def encode(self, text: str) -> list[int]:
                """将文本编码为 token ID 列表"""
                # 字符级编码，适合中文
                return [ord(c) for c in text if ord(c) <= 65535]  # 过滤超出BMP的字符
            
            def decode(self, tokens: list[int]) -> str:
                """将 token ID 列表解码为文本"""
                try:
                    return ''.join([chr(t) for t in tokens if 0 <= t <= 65535])
                except ValueError:
                    return ""
        
        # 创建 tokenizer 实例
        from lightrag.utils import Tokenizer
        custom_tokenizer = Tokenizer(
            model_name="chinese_custom", 
            tokenizer=SimpleChineseTokenizer()
        )
        
        print("✅ 自定义 tokenizer 创建成功（无需 tiktoken）")
        
        # 创建LightRAG实例，使用HTTP API的LLM + 本地embedding + 自定义tokenizer
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=qwen_llm_model_func,  # 使用HTTP API
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=5000,
                func=qwen_embedding_func,  # 使用本地bmodel
            ),
            tokenizer=custom_tokenizer,  # 关键：使用自定义tokenizer
            tiktoken_model_name=None,     # 确保不使用tiktoken
        )
        print("✅ LightRAG 实例创建成功")
        
    except Exception as e:
        print(f"❌ LightRAG 初始化失败: {e}")
        print("尝试使用最小配置重新初始化...")
        
        # 备用方案：更简单的tokenizer
        class BasicTokenizer:
            def encode(self, text: str) -> list[int]:
                # 最简单的字节编码
                return list(text.encode('utf-8'))
            
            def decode(self, tokens: list[int]) -> str:
                try:
                    return bytes(tokens).decode('utf-8', errors='ignore')
                except:
                    return ""
        
        from lightrag.utils import Tokenizer
        basic_tokenizer = Tokenizer(
            model_name="basic", 
            tokenizer=BasicTokenizer()
        )
        
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=qwen_llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=5000,
                func=qwen_embedding_func,
            ),
            tokenizer=basic_tokenizer,    # 使用基础tokenizer
            tiktoken_model_name=None,
        )

    # 4. 初始化存储系统
    print("\n=== 初始化存储系统 ===")
    try:
        await rag.initialize_storages()
        await initialize_pipeline_status()
        print("✅ 存储系统初始化完成")
    except Exception as e:
        print(f"⚠️ 存储系统初始化出现问题: {e}")
        print("继续运行...")

    print("✅ LightRAG 初始化完成（Embedding本地 + LLM远程API）")
    return rag

# 问题文件解析函数
def parse_question_file(file_path: str) -> tuple:
    """解析问题文件，返回 (序号, 问题内容)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if len(lines) >= 2:
            sequence_num = lines[0]
            question = lines[1]
            return sequence_num, question
        elif len(lines) == 1:
            # 只有一行，当作问题处理
            return "1", lines[0]
        else:
            return None, None
            
    except FileNotFoundError:
        print(f"问题文件 {file_path} 不存在")
        return None, None
    except Exception as e:
        print(f"解析问题文件时出错: {e}")
        return None, None

# 文档接口函数
def read_questions_from_file(file_path: str) -> list:
    """从文件中读取问题列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        return questions
    except FileNotFoundError:
        print(f"Warning: {file_path} not found, using default question")
        return ["介绍白血病与白细胞的关系"]  # 默认问题

def save_chunk_ids(chunk_ids: list, file_path: str):
    """将chunk ID保存到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for chunk_id in chunk_ids:
            f.write(f"{chunk_id}\n")
    print(f"Chunk IDs saved to: {file_path}")

# 处理单个问题的函数
async def process_single_question(rag, question: str, sequence_num: str):
    """处理单个问题的完整流程"""
    print(f"\n{'='*60}")
    print(f"处理问题序号 {sequence_num}: {question}")
    print('='*60)
    
    try:
        # 1. 直接调用 chunks_vdb.query 来获取检索结果
        print("\n=====================")
        print("Direct chunk search")
        print("=====================")
        start_time = time.time()
        
        try:
            chunks_result = await rag.chunks_vdb.query(
                query=question,
                top_k=5,
                ids=None
            )
            
            # 提取 chunk IDs
            chunk_ids = [chunk.get('id', 'N/A') for chunk in chunks_result]
            
            # 打印查询结果
            print(f"检索到 {len(chunks_result)} 个相关 chunks:")
            for i, chunk in enumerate(chunks_result):
                print(f"\nChunk {i+1}:")
                print(f"  ID: {chunk.get('id', 'N/A')}")
                print(f"  Score: {chunk.get('distance', 'N/A')}")
                print(f"  Content: {chunk.get('content', '')[:100]}...")
                print(f"  Full Doc ID: {chunk.get('full_doc_id', 'N/A')}")
                print(f"  Chunk Order Index: {chunk.get('chunk_order_index', 'N/A')}")
                print(f"  File Path: {chunk.get('file_path', 'N/A')}")

            # 2. 保存 chunk IDs 到 ragSearch.txt
            chunk_ids_file = os.path.join(OUTPUT_DIR, "ragSearch.txt")
            save_chunk_ids(chunk_ids, chunk_ids_file)
            
        except Exception as e:
            print(f"⚠️ Chunk 检索失败: {e}")
            chunk_ids_file = os.path.join(OUTPUT_DIR, "ragSearch.txt")
            save_chunk_ids(["检索失败"], chunk_ids_file)
        
        # 3. 执行完整的RAG查询
        print(f"\n=====================")
        print("Query mode: naive - Full RAG Search")
        print("=====================")
        start_time = time.time()
        
        try:
            resp = await rag.aquery(
                question,
                param=QueryParam(mode="naive", stream=True),
            )
            end_time = time.time()
            print(f"Search execution time: {end_time - start_time:.2f} seconds")
            
            # 替换原有的 if inspect.isasyncgen(resp): ... else: ... 逻辑
            response_file = os.path.join(OUTPUT_DIR, "RAGResult.txt")

            # 确保文件在使用前是空的
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write("") # 清空文件

            print("Response lightrag_bmodel2 success:")
            # 以追加模式打开文件，在循环中写入
            with open(response_file, 'a', encoding='utf-8') as f:
                if inspect.isasyncgen(resp):
                    # 流式响应
                    async for chunk in resp:
                        if chunk:
                            # 同时输出到终端和文件
                            print(chunk, end="", flush=True)
                            f.write(chunk)
                            f.flush() # 关键：确保立即写入磁盘
                else:
                    # 非流式响应（作为备用逻辑）
                    response_text = str(resp)
                    print(response_text)
                    f.write(response_text)

            # 循环结束后换行，美化终端输出
            print() 

            print(f"\n问题序号 {sequence_num} 处理完成")
            print(f"- Chunk IDs 保存至: {chunk_ids_file}")
            print(f"- 响应流式保存至: {response_file}")

            return True
            
        except Exception as e:
            print(f"⚠️ RAG 查询失败: {e}")
            # 尝试直接使用 LLM
            print("尝试直接使用 LLM 回答...")
            
            return False
        
    except Exception as e:
        print(f"处理问题时出错: {e}")
        import traceback
        traceback.print_exc()

        return False

# 监控和处理问题的主函数 - 修改版本
async def monitor_and_process_questions(rag):
    """监控问题文件并处理"""
    questions_file = os.path.join(OUTPUT_DIR, "ragQuestions.txt")
    processed_sequences = set()  # 记录已处理的序号
    
    async def handle_file_change(file_path):
        """处理文件变化"""
        try:
            # 检查文件是否为空
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                print("问题文件为空，等待用户输入问题...")
                return
            
            sequence_num, question = parse_question_file(file_path)
            
            if sequence_num and question:
                # 检查是否是新问题（基于序号）
                if sequence_num not in processed_sequences:
                    print(f"\n检测到新问题: 序号 {sequence_num}")
                    success = await process_single_question(rag, question, sequence_num)
                    
                    if success:
                        processed_sequences.add(sequence_num)
                        print(f"问题序号 {sequence_num} 处理成功")
                    else:
                        print(f"问题序号 {sequence_num} 处理失败")
                else:
                    print(f"问题序号 {sequence_num} 已处理过，跳过")
            else:
                print("问题文件格式不正确或为空")
        except Exception as e:
            print(f"处理文件变化时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 获取当前事件循环
    loop = asyncio.get_running_loop()
    
    # 创建文件监控处理器，传入事件循环
    event_handler = QuestionFileHandler(callback=handle_file_change, loop=loop)
    
    # 设置文件监控
    observer = Observer()
    observer.schedule(event_handler, OUTPUT_DIR, recursive=False)
    observer.start()
    
    print(f"开始监控问题文件: {questions_file}")
    print("请在 ragQuestions.txt 中写入问题，格式如下:")
    print("第一行: 序号")
    print("第二行: 问题内容")
    print("按 Ctrl+C 退出监控")
    
    # 初始化时读取当前序号，但不处理
    if os.path.exists(questions_file):
        try:
            sequence_num, _ = parse_question_file(questions_file)
            if sequence_num:
                processed_sequences.add(sequence_num)
                print(f"检测到现有问题文件，当前序号: {sequence_num}，等待序号变化...")
        except:
            pass
    
    try:
        # 保持监控运行
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n停止文件监控")
        observer.stop()
    
    observer.join()

def create_example_question_file():
    """创建空的问题文件"""
    questions_file = os.path.join(OUTPUT_DIR, "ragQuestions.txt")
    if not os.path.exists(questions_file):
        # 创建空文件，不写入任何内容
        with open(questions_file, 'w', encoding='utf-8') as f:
            f.write("")
        
        print(f"创建空的问题文件: {questions_file}")
        print("等待用户输入问题...")
    else:
        print(f"问题文件已存在: {questions_file}")
        print("等待问题序号变化...")

# 定义异步主函数
async def main():
    try:
        # 首先检查模型状态
        print("=" * 60)
        print("检查系统状态...")
        print("=" * 60)
        
        llm_available = check_model_availability()
        api_available = check_api_server_status()
        
        if not (llm_available or api_available):
            print("⚠️ 警告：LLM模型可能未准备就绪")
            print("请确保 sample_audio.py 正在运行")
            print("继续初始化，将尝试通过API访问...")
        
        # 初始化RAG实例
        print("\n" + "=" * 60)
        print("初始化 RAG 系统...")
        print("=" * 60)
        
        rag = await initialize_rag()

        # 测试嵌入函数
        test_text = ["This is a test string for embedding with Qwen embedding bmodel."]
        print("\n=======================")
        print("Testing Qwen embedding bmodel embedding function")
        print("========================")
        print(f"Test text: {test_text}")
        
        try:
            embedding = await rag.embedding_func(test_text)
            embedding_dim = embedding.shape[1]
            print(f"✅ Detected embedding dimension: {embedding_dim}")
            print(f"Embedding shape: {embedding.shape}")
            print(f"First 10 values: {embedding[0][:10]}\n")
        except Exception as e:
            print(f"⚠️ Embedding 测试失败: {e}")

        # # 测试LLM函数
        # print("\n=======================")
        # print("Testing Qwen LLM HTTP API function")
        # print("========================")
        # test_prompt = "你好，请简单介绍一下你自己。"
        # print(f"Test prompt: {test_prompt}")
        
        # try:
        #     llm_response = await rag.llm_model_func(
        #         test_prompt,
        #         system_prompt="你是一个有用的助手。"
        #     )
        #     print(f"✅ LLM Response: {llm_response}\n")
        # except Exception as e:
        #     print(f"⚠️ LLM 测试失败: {e}")
        #     print("可能原因：sample_audio.py 未运行或API服务器未启动")
        
        # 创建示例问题文件
        create_example_question_file()
        
        # 开始监控和处理问题
        print("\n" + "=" * 60)
        print("开始问题监控...")
        print("=" * 60)
        
        await monitor_and_process_questions(rag)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'rag' in locals():
            try:
                await rag.finalize_storages()
            except:
                pass

# 程序入口点
if __name__ == "__main__":
    # 在运行主函数前配置日志
    configure_logging()
    
    print("🚀 启动 LightRAG with Qwen bmodel (混合模式)")
    print("=" * 60)
    print("架构说明:")
    print("- LLM: 通过 HTTP API 访问 (qwen_llm_adapter.py)")
    print("- Embedding: 直接加载本地 bmodel")
    print("- 注意：请确保 sample_audio.py 正在运行以提供LLM服务")
    print("=" * 60)
    
    # 运行异步主函数
    asyncio.run(main())
    
    # 打印完成信息
    print("\nDone! Qwen embedding bmodel integration with LightRAG completed successfully.")