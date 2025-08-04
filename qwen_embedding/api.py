"""
简化的应用入口，使用合并后的模块
"""
from qwen_embedding import load_model, process_chinese_text, save_result_to_file

if __name__ == '__main__':
    # 加载模型
    model = load_model()
    
    # 处理文本
    text = "你好，这是一个测试。"
    result = process_chinese_text(model, text)
    
    # 保存结果
    save_result_to_file(result, text)