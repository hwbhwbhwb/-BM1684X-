import pyaudio
import wave
import threading
import queue
import os
import time
import json
from datetime import datetime
import globalConfig
import random
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 导入简化的共享管理器
from shared_qwen_manager import simple_qwen, generate_mermaid, generate_text

# 参数设置
FORMAT = pyaudio.paInt16  # 16位采样
CHANNELS = 1              # 单声道
RATE = 16000              # 采样率（Hz）
CHUNK = 1024              # 每个缓冲区的帧数
RECORD_SECONDS = 30       # 每个文件录音时长
RECORDING_FOLDER = "recordings"  # 录音文件保存文件夹

# 创建保存录音的文件夹
os.makedirs(RECORDING_FOLDER, exist_ok=True)

# 创建一个队列用于存储待处理的音频文件
audio_queue = queue.Queue()

exitFlag = False

class QwenAPIHandler(BaseHTTPRequestHandler):
    """简单的HTTP API处理器，用于跨进程推理"""
    
    def do_POST(self):
        if self.path == '/generate':
            try:
                # 读取请求数据
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                prompt = data.get('prompt', '')
                max_new_tokens = data.get('max_new_tokens', 512)
                
                if not prompt:
                    self.send_error(400, "Missing prompt")
                    return
                
                # 使用简化的任务函数
                result = generate_text(prompt, max_new_tokens)
                
                # 返回结果
                response = {'result': result, 'status': 'success'}
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                error_response = {'error': str(e), 'status': 'error'}
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
    
    def do_GET(self):
        if self.path == '/status':
            # 返回模型状态
            status = simple_qwen.get_status()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(status).encode('utf-8'))
        else:
            self.send_error(404, "Not found")
    
    def log_message(self, format, *args):
        # 禁用默认日志输出
        pass

def start_api_server():
    """启动API服务器"""
    try:
        server = HTTPServer(('localhost', 8899), QwenAPIHandler)
        print("🌐 API服务器启动在 http://localhost:8899")
        print("  - POST /generate : 文本生成")
        print("  - GET /status : 模型状态")
        server.serve_forever()
    except Exception as e:
        print(f"❌ API服务器启动失败: {e}")

def initEnv():
    import os
    # 设置环境变量
    os.environ["LOG_LEVEL"] = "-1"
    os.environ["LD_LIBRARY_PATH"] = "/opt/sophon/libsophon-current/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

    # 启动 pulseaudio（忽略错误）
    os.system("pulseaudio --start 2>/dev/null || true")

def initMermaid(text):
    template = f"""
graph LR
    A[{text}开启全新教学模式] --> B[思维导图]
    A --> C[课本查询]
    A --> D[互动提问]
    A --> E[历史笔记]
"""
    # Ensure the output directory exists
    os.makedirs('/data/mermaidRender/text', exist_ok=True)
    with open('/data/mermaidRender/text/outmermaid.mmd', "w") as f:
        f.write(template)

def record_audio():
    """录音线程：持续录音并每10秒保存一个文件"""
    # 初始化PyAudio
    audio = pyaudio.PyAudio()
    
    # 打开流
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    
    print("录音线程已启动")
    last_state = True
    try:
        while True:
            if exitFlag:
                print("record_audio线程退出")
                break
            if not globalConfig.running:
                time.sleep(1)
                last_state = False
                continue
            # 清空输入缓冲区，避免残留数据影响新录音
            if(not last_state):
                print("录音线程开始，清空输入缓冲区...")
                initMermaid("正在采集中...")
                last_state = True
                stream.stop_stream()
                stream.start_stream()

            frames = []
            filename = os.path.join(
                RECORDING_FOLDER,
                f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{random.randint(1000,9999)}.wav"
            )
            print(f"开始录制文件: {filename}")
            
            # 录制指定秒数的音频
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                if not globalConfig.running:
                    break
                data = stream.read(CHUNK)
                frames.append(data)
            
            # 保存为WAV文件
            if frames:  # 确保有数据再保存
                wf = wave.open(filename, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                # 将文件名添加到处理队列
                audio_queue.put(filename)
                print(f"文件已保存: {filename}")
    
    finally:
        # 停止和关闭流
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("录音线程已结束")

def remove_tail_repeats_auto(s: str) -> str:
    """
    自动判断重复片段长度，从后向前删除连续重复的字符串段，只保留第一次出现。
    支持多种长度的重复递归去重。
    """
    n = len(s)
    changed = True
    while changed:
        changed = False
        for seg_len in range(n // 2, 0, -1):
            # 检查结尾是否有连续重复
            last_seg = s[-seg_len:]
            count = 1
            for i in range(2, n // seg_len + 1):
                start = -i * seg_len
                end = -(i - 1) * seg_len if -(i - 1) * seg_len != 0 else None
                if s[start:end] == last_seg:
                    count += 1
                else:
                    break
            if count > 1:
                cutoff = len(s) - (count - 1) * seg_len
                s = s[:cutoff]
                n = len(s)
                changed = True
                break  # 重新从最大长度开始
    return s

def join_segments_text(segments):
    """
    接收一个包含若干字典（每个字典是一个segment，含'text'字段）的列表，
    返回所有text字段拼接的字符串，用逗号分隔。
    """
    texts = [segment['text'] for segment in segments]
    return '，'.join(texts)  # 使用中文逗号拼接

def clean_mermaid_to_lr(text: str) -> str:
    """
    清理 Mermaid 图内容：
    - 删除 ```mermaid 和 ```
    - 将 graph TD 替换为 graph LR
    """
    lines = text.strip().splitlines()
    cleaned_lines = []
    inside_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped == "```mermaid":
            inside_code_block = True
            continue
        elif stripped == "```":
            inside_code_block = False
            continue
        elif inside_code_block:
            if stripped.startswith("graph TD"):
                cleaned_lines.append(stripped.replace("graph TD", "graph LR", 1))
            else:
                cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

def process_audio(whisperModel):
    """处理线程：处理队列中的音频文件并删除"""
    print("处理线程已启动，等待文件...")
    text = ""
    last_qwen_time = 0
    while True:
        if exitFlag:
            print("process_audio处理线程退出")
            break
        if not globalConfig.running:
            text = ""   
            # 清空所有待处理内容
            while not audio_queue.empty():
                try:
                    filename = audio_queue.get_nowait()
                    audio_queue.task_done()
                    os.remove(filename)
                    print(f"清理: {filename}")
                except queue.Empty:
                    break
            time.sleep(1)
            continue
        try:
            # 从队列中获取文件名
            filename = audio_queue.get()
            
            # 这里添加你的音频处理代码
            print(f"正在处理文件: {filename}")

            resWhisper = whisperModel.transcribe_audio(filename)
            resWhisperText = remove_tail_repeats_auto(join_segments_text(resWhisper["segments"]))
            if(resWhisperText == ''):
                print("内容为空")
                # 标记任务完成并删除文件
                audio_queue.task_done()
                os.remove(filename)
                print(f"文件已删除: {filename}")
                continue
            text += resWhisperText
            #text写入文本
            with open('out.txt', "w") as f:
                f.write(text)
            print(text)
            # Qwen生成间隔控制
            now = time.time()
            if now - last_qwen_time >= globalConfig.qwen_interval:
                print(f"文本长度: {len(text)}")
                
                # 使用简化的任务函数
                try:
                    res = generate_mermaid(text, len(text)*3)
                    last_qwen_time = now
                    
                    if res:
                        # 把res的内容写入文件
                        os.makedirs('/data/mermaidRender/text', exist_ok=True)
                        with open('/data/mermaidRender/dist/text/outmermaid.mmd', "w") as f:
                            f.write(clean_mermaid_to_lr(res))
                        print("思维导图已更新")
                    else:
                        print("思维导图生成失败")
                        
                except Exception as e:
                    print(f"Qwen 处理失败: {e}")
                
            print(f"文件处理完成: {filename}")
            print(f"处理结果: {text}")

            # 标记任务完成
            audio_queue.task_done()
            os.remove(filename)
            print(f"文件已删除: {filename}")
        except queue.Empty:
            # 队列为空，等待globalConfig.running为True再继续
            time.sleep(1)
            continue
        except Exception as e:
            print(f"处理音频时出错: {e}")
            # 确保任务被标记为完成，即使出错
            try:
                audio_queue.task_done()
                if 'filename' in locals() and os.path.exists(filename):
                    os.remove(filename)
            except:
                pass

def initModel():
    """初始化模型"""
    print("=== 初始化模型 ===")
    
    # 首先检查共享管理器状态
    status = simple_qwen.get_status()
    print(f"共享管理器初始状态: {status}")
    
    # 初始化 Qwen 模型（如果尚未加载）
    if not simple_qwen.is_loaded():
        print("🔄 初始化 Qwen 模型...")
        initMermaid("加载Qwen模型...")
        
        class Args:
            model_path = "/data/qwen4btune_w4bf16_seq8192_bm1684x_1dev_20250721_195513.bmodel"
            config_path = "/data/LLM-TPU/models/Qwen3/python_demo/config"
            devid = "0"
            temperature = 0.5
            top_p = 1.0
            repeat_penalty = 1.8
            repeat_last_n = 32
            max_new_tokens = 1024
            generation_mode = "greedy"
            prompt_mode = "prompted"
            enable_history = False

        args = Args()
        
        try:
            simple_qwen.initialize_model(args)
            print("✅ Qwen 模型初始化完成")
        except Exception as e:
            print(f"❌ Qwen 模型初始化失败: {e}")
            raise
    else:
        print("✅ Qwen 模型已经加载")
    
    # 初始化 Whisper 模型
    print("🔄 初始化 Whisper 模型...")
    initMermaid("加载Whisper模型...")
    
    try:
        from transcribef import TranscribeWorker
        worker = TranscribeWorker()
        print("✅ Whisper 模型初始化完成")
    except Exception as e:
        print(f"❌ Whisper 模型初始化失败: {e}")
        raise
    
    initMermaid("")
    
    # 验证最终状态
    final_status = simple_qwen.get_status()
    print(f"模型初始化完成后状态: {final_status}")
    
    return worker

if __name__ == "__main__":
    try:
        print("🚀 启动语音识别思维导图系统...")
        
        # 初始化环境
        initEnv()
        
        # 初始化模型
        worker = initModel()
        
        # 启动API服务器（在单独线程中）
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()
        
        # 创建并启动录音线程
        record_thread = threading.Thread(target=record_audio)
        record_thread.start()
        
        # 创建并启动处理线程 - 注意这里只传入 worker
        process_thread = threading.Thread(target=process_audio, args=(worker,))
        process_thread.start()

        # 创建并启动配置监控线程
        config_thread = threading.Thread(target=globalConfig.config_watcher, daemon=True)
        config_thread.start()
        
        print("系统启动完成！按 Ctrl+C 退出...")
        print("=" * 50)
        
        # 主线程等待用户中断
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n正在停止所有线程...")
        exitFlag = True
        
        # 等待线程结束
        print("等待录音线程结束...")
        record_thread.join(timeout=3)
        
        print("等待处理线程结束...")
        process_thread.join(timeout=3)
        
        # 清理剩余文件
        print("清理剩余文件...")
        while not audio_queue.empty():
            try:
                filename = audio_queue.get_nowait()
                audio_queue.task_done()
                if os.path.exists(filename):
                    os.remove(filename)
                print(f"清理: {filename}")
            except queue.Empty:
                break
            except Exception as e:
                print(f"清理文件时出错: {e}")
        
        # 关闭共享管理器
        simple_qwen.shutdown()
        
        print("所有线程已停止，程序退出")
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 确保清理
        exitFlag = True
        simple_qwen.shutdown()