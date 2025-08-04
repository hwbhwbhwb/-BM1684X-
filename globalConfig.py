import time
import json
# 控制线程运行的标志
running = True
qwen_interval = 30  # Qwen生成时间间隔，秒
# 新增一个标志位，用于RAG任务的优先级控制
rag_task_active = False

CONFIG_PATH = "/data/mermaidRender/dist/config.json"

def load_config():
    """从配置文件读取配置，返回字典"""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        return {}

def config_watcher():
    """配置文件监控线程，实时更新全局配置变量"""
    global running, qwen_interval
    last_config = {}
    print("配置监控线程已启动...")
    while True:
        config = load_config()
        if config != last_config:
            if "running" in config:
                running = bool(config["running"])
                print(f"配置更新: running = {running}")
            if "qwen_interval" in config:
                qwen_interval = float(config["qwen_interval"])
                print(f"配置更新: qwen_interval = {qwen_interval}")
            last_config = config
        time.sleep(1)  # 每秒检查一次
