import os
import json
import threading
from typing import Optional
from pipelinef import Qwen2

class SimpleQwenManager:
    """简化的 Qwen 模型管理器 - 仅用于本地管理"""
    
    _instance: Optional['SimpleQwenManager'] = None
    _lock = threading.Lock()
    
    # 状态文件
    STATUS_FILE = "/tmp/qwen_model_status.json"
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model: Optional[Qwen2] = None
        self.model_lock = threading.RLock()
        self._initialized = True
    
    def _write_status_file(self, status):
        """写入状态文件"""
        try:
            with open(self.STATUS_FILE, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            print(f"写入状态文件失败: {e}")
    
    def _read_status_file(self):
        """读取状态文件"""
        try:
            if os.path.exists(self.STATUS_FILE):
                with open(self.STATUS_FILE, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {"model_loaded": False, "pid": None, "model_path": None}
    
    def initialize_model(self, args):
        """初始化模型"""
        with self.model_lock:
            if self.model is None:
                print("🔄 初始化 Qwen 模型...")
                self.model = Qwen2(args)
                
                # 更新状态文件
                status = {
                    "model_loaded": True,
                    "pid": os.getpid(),
                    "model_path": args.model_path,
                    "timestamp": __import__('time').time()
                }
                self._write_status_file(status)
                
                print("✅ Qwen 模型初始化完成")
            else:
                print("✅ 模型已经加载")
    
    def get_model(self):
        """获取模型实例"""
        return self.model
    
    def is_loaded(self):
        """检查模型是否已加载"""
        return self.model is not None
    
    def get_status(self):
        """获取状态（主要用于兼容性）"""
        status = self._read_status_file()
        return {
            "model_loaded": self.model is not None,
            "local_model": self.model is not None,
            "external_model": status.get("model_loaded", False) and status.get("pid") != os.getpid(),
            "external_pid": status.get("pid"),
            "current_pid": os.getpid()
        }
    
    def shutdown(self):
        """关闭管理器"""
        if self.model is not None:
            print("🔄 清理模型状态...")
            self._write_status_file({"model_loaded": False, "pid": None, "model_path": None})
        print("✅ 模型管理器已关闭")

# 全局单例
simple_qwen = SimpleQwenManager()

# 简化的任务函数
def generate_mermaid(text, maxWord):
    """思维导图生成"""
    if simple_qwen.model:
        return simple_qwen.model.chat(text, maxWord)
    else:
        raise RuntimeError("模型未加载")

#shared_qwen_manager.py并没有调用这个函数，而是使用了API
def generate_text(prompt, max_new_tokens=512):
    """文本生成"""
    if simple_qwen.model:
        return simple_qwen.model.generate_text(prompt, max_new_tokens)
    else:
        raise RuntimeError("模型未加载")