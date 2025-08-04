"""
Qwen LLM bmodel 适配器 - 使用HTTP API访问远程模型
为 LightRAG 提供本地 Qwen bmodel 推理接口
"""
import sys
import os
import asyncio
import signal
import requests
import json
import time
from typing import List, Optional, AsyncIterator

# 导入简化的共享管理器用于状态检查
sys.path.append('/data/whisper-TPU_py/bmwhisper')
from shared_qwen_manager import simple_qwen

# ============ 模型路径配置 ============
MODEL_CONFIG = {
    "llm_model_path": "/data/qwen4btune_w4bf16_seq8192_bm1684x_1dev_20250721_195513.bmodel",
    "config_path": "/data/LLM-TPU/models/Qwen3/python_demo/config",
    "device_id": "0",
    "temperature": 0.5,
    "top_p": 1.0,
    "repeat_penalty": 1.0,
    "repeat_last_n": 32,
    "max_new_tokens": 512,
    "generation_mode": "greedy"
}
# =====================================

# API配置
API_BASE_URL = "http://localhost:8899"

class QwenLLMAdapter:
    """Qwen LLM bmodel 适配器类 - 通过HTTP API访问模型"""
    
    def __init__(self, **kwargs):
        self.config = {
            "model_path": kwargs.get("model_path", MODEL_CONFIG["llm_model_path"]),
            "config_path": kwargs.get("config_path", MODEL_CONFIG["config_path"]),
            "device_id": kwargs.get("device_id", MODEL_CONFIG["device_id"]),
            "temperature": kwargs.get("temperature", MODEL_CONFIG["temperature"]),
            "top_p": kwargs.get("top_p", MODEL_CONFIG["top_p"]),
            "repeat_penalty": kwargs.get("repeat_penalty", MODEL_CONFIG["repeat_penalty"]),
            "repeat_last_n": kwargs.get("repeat_last_n", MODEL_CONFIG["repeat_last_n"]),
            "max_new_tokens": kwargs.get("max_new_tokens", MODEL_CONFIG["max_new_tokens"]),
            "generation_mode": kwargs.get("generation_mode", MODEL_CONFIG["generation_mode"]),
            "prompt_mode": "prompted",
            "enable_history": False
        }
        
        self._check_model_availability()
    
    def _check_api_server(self, timeout=10):
        """检查API服务器是否可用"""
        try:
            response = requests.get(f"{API_BASE_URL}/status", timeout=timeout)
            if response.status_code == 200:
                status = response.json()
                return True, status
            else:
                return False, f"HTTP {response.status_code}"
        except requests.exceptions.Timeout:
            return False, "连接超时"
        except requests.exceptions.ConnectionError:
            return False, "连接拒绝"
        except Exception as e:
            return False, str(e)
    
    def _check_model_availability(self):
        """检查模型可用性"""
        # 首先检查简化的共享管理器状态
        status = simple_qwen.get_status()
        print(f"共享管理器状态: {status}")
        
        if status["external_model"]:
            print(f"✅ 检测到外部进程(PID: {status['external_pid']})已加载模型")
            
            # 检查API服务器是否可用
            api_available, api_status = self._check_api_server(timeout=10)
            if api_available:
                print("✅ API服务器可用，可以进行推理")
            else:
                print(f"⚠️ API服务器不可用: {api_status}")
                print("请确保 sample_audio.py 正在运行")
       
        else:
            print("⚠️ 未检测到已加载的模型")
            print("请先运行 sample_audio.py 加载模型")
    
    def _call_api(self, prompt: str, max_new_tokens: int = 512, retries=3) -> str:
        """调用API进行推理，带重试机制"""
        
        for attempt in range(retries):
            try:
                print(f"📡 发送API请求 (尝试 {attempt + 1}/{retries})")
                
                data = {
                    "prompt": prompt,
                    "max_new_tokens": max_new_tokens
                }
                
                # 增加超时时间
                response = requests.post(
                    f"{API_BASE_URL}/generate",
                    json=data,
                    timeout=180  # 3分钟超时
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        print("✅ API请求成功")
                        return result.get("result", "")
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        print(f"❌ API返回错误: {error_msg}")
                        if attempt < retries - 1:
                            print(f"🔄 {5}秒后重试...")
                            time.sleep(5)
                            continue
                        return f"API Error: {error_msg}"
                else:
                    error_msg = f"HTTP {response.status_code}"
                    print(f"❌ HTTP错误: {error_msg}")
                    if attempt < retries - 1:
                        print(f"🔄 {3}秒后重试...")
                        time.sleep(3)
                        continue
                    return f"HTTP Error: {error_msg}"
                    
            except requests.exceptions.ConnectionError:
                error_msg = "无法连接到API服务器"
                print(f"❌ 连接错误: {error_msg}")
                if attempt < retries - 1:
                    print(f"🔄 {3}秒后重试...")
                    time.sleep(3)
                    continue
                return f"Error: {error_msg}。请确保 sample_audio.py 正在运行。"
            except requests.exceptions.Timeout:
                error_msg = "API请求超时"
                print(f"❌ 超时错误: {error_msg}")
                if attempt < retries - 1:
                    print(f"🔄 {5}秒后重试...")
                    time.sleep(5)
                    continue
                return f"Error: {error_msg}"
            except Exception as e:
                error_msg = str(e)
                print(f"❌ 其他错误: {error_msg}")
                if attempt < retries - 1:
                    print(f"🔄 {3}秒后重试...")
                    time.sleep(3)
                    continue
                return f"Error: {error_msg}"
        
        return "Error: 所有重试都失败了"
    
    def generate(self, prompt: str, system_prompt: str = None, 
                history_messages: list = None, stream: bool = False, **kwargs) -> str:
        """生成回复（同步版本）"""
        try:
            # 构建完整的提示
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\nUser: {prompt}"
            
            
            status = simple_qwen.get_status()
            
            # 使用API调用
            print("🔄 使用API推理...")
            return self._call_api(
                full_prompt,
                kwargs.get('max_new_tokens', self.config['max_new_tokens'])
            )
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def agenerate(self, prompt: str, system_prompt: str = None, 
                       history_messages: list = None, stream: bool = False, **kwargs) -> str:
        """异步生成回复"""
        # 在异步环境中调用同步方法
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.generate, 
            prompt, 
            system_prompt, 
            history_messages, 
            stream, 
            **kwargs
        )
    
    async def agenerate_stream(self, prompt: str, system_prompt: str = None, 
                              history_messages: list = None, **kwargs) -> AsyncIterator[str]:
        """异步流式生成（简化版本）"""
        try:
            result = await self.agenerate(prompt, system_prompt, history_messages, **kwargs)
            
            if result.startswith("Error:"):
                yield result
                return
            
            words = result.split()
            for i, word in enumerate(words):
                if i == 0:
                    yield word
                else:
                    yield " " + word
                await asyncio.sleep(0.01)
                
        except Exception as e:
            yield f"Error: {str(e)}"

# 全局适配器实例
_global_llm_adapter: Optional[QwenLLMAdapter] = None

def get_qwen_llm_adapter(model_path: str = None,
                        config_path: str = None,
                        **kwargs) -> QwenLLMAdapter:
    """获取全局 LLM 适配器实例"""
    global _global_llm_adapter
    
    if _global_llm_adapter is None:
        config_kwargs = kwargs.copy()
        if model_path:
            config_kwargs["model_path"] = model_path
        if config_path:
            config_kwargs["config_path"] = config_path
            
        _global_llm_adapter = QwenLLMAdapter(**config_kwargs)
    
    return _global_llm_adapter

async def qwen_llm_model_func(prompt: str,
                             system_prompt: str = None,
                             history_messages: List = None,
                             keyword_extraction: bool = False,
                             stream: bool = False,
                             hashing_kv=None,
                             model_path: str = None,
                             config_path: str = None,
                             **kwargs) -> str:
    """LightRAG 兼容的 LLM 函数"""
    adapter = get_qwen_llm_adapter(
        model_path=model_path,
        config_path=config_path,
        **kwargs
    )
    
    if stream:
        async def stream_generator():
            async for chunk in adapter.agenerate_stream(
                prompt, system_prompt, history_messages, **kwargs
            ):
                yield chunk
        return stream_generator()
    else:
        return await adapter.agenerate(
            prompt, system_prompt, history_messages, stream, **kwargs
        )

def test_shared_manager_status():
    """测试共享管理器状态"""
    print("🔍 检查共享管理器状态...")
    try:
        status = simple_qwen.get_status()
        print(f"管理器状态: {status}")
    
        if status["external_model"]:
            print(f"✅ 外部模型已加载 (PID: {status['external_pid']})")
        else:
            print("⚠️ 未检测到已加载的模型")
            
        return status["model_loaded"]
    except Exception as e:
        print(f"❌ 检查状态失败: {e}")
        return False

def test_qwen_llm_adapter():
    """测试 Qwen LLM 适配器"""
    print("🧪 测试 Qwen LLM bmodel 适配器...")
    
    try:
        if not test_shared_manager_status():
            print("⚠️ 共享管理器状态异常，但继续测试...")
        
        adapter = get_qwen_llm_adapter()
        
        print("📝 测试同步生成...")
        print("提示：这可能需要几分钟时间，请耐心等待...")
        
        response = adapter.generate(
            "简单介绍一下人工智能",
            system_prompt="你是一个有用的助手，请简洁回答。"
        )
        
        print(f"同步生成回复: {response[:200]}...")
        
        if response and not response.startswith("Error:"):
            print("✅ Qwen LLM 适配器同步测试通过！")
            return True
        else:
            print("❌ 同步测试失败或返回错误")
            print(f"完整回复: {response}")
            return False
        
    except Exception as e:
        print(f"❌ Qwen LLM 适配器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        print("🚀 启动 Qwen LLM 适配器测试套件...")
        print("=" * 60)
        
        sync_success = test_qwen_llm_adapter()
        print("-" * 60)
        
        if sync_success:
            print("\n🎉 测试通过！适配器可以正确使用模型。")
        else:
            print("\n⚠️ 测试失败，请检查:")
            print("  1. sample_audio.py 是否正在运行")
            print("  2. 模型是否正忙碌 (等待当前任务完成)")
            print("  3. 网络连接是否正常")
        
    except KeyboardInterrupt:
        print("\n🔄 用户中断，程序退出")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")