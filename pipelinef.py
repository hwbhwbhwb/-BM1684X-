import argparse
import chat
import time
from transformers import AutoTokenizer
import globalConfig

class Qwen2():

    def __init__(self, args):
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]

        # load tokenizer
        print("Load " + args.config_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.config_path, trust_remote_code=True)

        # warm up
        self.tokenizer.decode([0])

        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = "You are a helpful assistant."
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.EOS = self.tokenizer.eos_token_id
        self.enable_history = args.enable_history

        self.model = chat.Qwen()
        self.init_params(args)
        self.load_model(args.model_path)

    def load_model(self, model_path):
        load_start = time.time()
        self.model.init(self.devices, model_path)
        load_end = time.time()
        print(f"\nLoad Time: {(load_end - load_start):.3f} s")

    def init_params(self, args):
        self.model.temperature = args.temperature
        self.model.top_p = args.top_p
        self.model.repeat_penalty = args.repeat_penalty
        self.model.repeat_last_n = args.repeat_last_n
        self.model.max_new_tokens = args.max_new_tokens
        self.model.generation_mode = args.generation_mode

    def clear(self):
        self.history = [{"role": "system", "content": self.system_prompt}]

    def update_history(self):
        if self.model.token_length >= self.model.SEQLEN:
            print("... (reach the maximal length)", flush=True, end="")
            self.history = [{"role": "system", "content": self.system_prompt}]
        else:
            self.history.append({"role": "assistant", "content": self.answer_cur})

    def encode_tokens(self, content):
        """普通文本对话编码 - 用于 LightRAG"""
        # 为 LightRAG 使用简单的对话模板
        history = [
            {"role": "system", "content": "你是一个有用的AI助手，请简洁明了地回答问题。"},
            {"role": "user", "content": content}
        ]
        
        text = self.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # 明确关闭 think 模式
        )
        
        tokens = self.tokenizer(text).input_ids
        return tokens
    
    def encode_tokens_mermaid(self, content):
        """专门用于思维导图生成的编码方法"""
        template = f'''
<|im_start|>system\n你是一个Mermaid代码生成专家，用户将输入一段录音的语音识别结果，请你详细的总结并绘制一张思维导图，用Mermaid形式，生成符合语法的Mermaid代码。要求：\n1. 仅输出标准Mermaid代码，不包含任何解释或注释；\n2. 节点名称需用中文，但语法关键字（如graph TD、-->）保持英文；\n3. 严格遵循Mermaid语法规范。\n\n注意语法细节：\n- 节点定义格式：节点名[中文标签]，如 A[用户登录]\n- 连接符：使用 -->、---、==>|文字|== 等标准符号\n- 子图需用subgraph包裹并正确闭合\n- 方括号中的文字（即节点标签）内容请用引号包裹\n- 节点名称请使用不会与mermaid语法发生冲突的无意义内容，节点需要显示的内容请放在节点标签（即名称后的中括号）中\n\n*请确保你生成的是Mermaid代码!你只需要生成Mermaid代码，请不要附加其他信息!Mermaid代码请放到代码块中!*\n\n语音识别结果如下：\n<|im_end|>\n<|im_start|>user\n
{content}
\n<|im_end|>\n<|im_start|>system\n现在请你将上述内容详细的总结并绘制一张思维导图，用Mermaid形式，生成符合语法的Mermaid代码。要求：\n1. 仅输出标准Mermaid代码，不包含任何解释或注释；\n2. 节点名称需用中文，但语法关键字（如graph TD、-->）保持英文；\n3. 严格遵循Mermaid语法规范。\n\n注意语法细节：\n- 节点定义格式：节点名[\"中文标签\"]，如 A[\"用户登录\"]\n- 连接符：使用 -->、---、==>|文字|== 等标准符号\n- 子图需用subgraph包裹并正确闭合\n- 方括号中的文字（即节点标签）请用引号包裹\n- 节点名称请使用不会与mermaid语法发生冲突的无意义内容，节点需要显示的内容请放在节点标签（即名称后的中括号）中\n\n*请确保你生成的是Mermaid代码!你只需要生成Mermaid代码，请不要附加其他信息!Mermaid代码请放到代码块中!节点标签中的文字请使用英文引号\"\"包裹!*<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
'''
        tokens = self.tokenizer(template.format(content = content),
                                padding=False,  # "longest",  # "max_length",
                                truncation=True,
                                max_length=5000,
                                return_special_tokens_mask=True).input_ids
        return tokens

    def chat(self, content, maxWord = 100000):
        """专门用于思维导图生成的聊天方法"""
        tokens = self.encode_tokens_mermaid(content)
        
        # check tokens
        if not tokens:
            print("Sorry: your question is empty!!")
            return
        if len(tokens) > self.model.SEQLEN:
            print(
                "The maximum question length should be shorter than {} but we get {} instead."
                .format(self.model.SEQLEN, len(tokens)))
            return
        print("\nAnswer: ", end="")
        return self.stream_answer(tokens, maxWord)
    
    def generate_text(self, content, maxWord = 1024):
        """普通文本生成方法 - 用于 LightRAG"""
        tokens = self.encode_tokens(content)
        
        # check tokens
        if not tokens:
            print("Sorry: your question is empty!!")
            return "Error: Empty input"
        if len(tokens) > self.model.SEQLEN:
            print(
                "The maximum question length should be shorter than {} but we get {} instead."
                .format(self.model.SEQLEN, len(tokens)))
            return f"Error: Input too long ({len(tokens)} tokens > {self.model.SEQLEN})"
        
        # return self.stream_answer_RAG(tokens, maxWord)
        yield from self.stream_answer_RAG(tokens, maxWord)

    def stream_answer(self, tokens , maxWord = 100000):
        """
        Stream the answer for the given tokens.
        """
        interrupted = False
        
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []

        # First token
        first_start = time.time()
        token = self.model.forward_first(tokens)
        first_end = time.time()
        # Following tokens
        full_word_tokens = []
        last_state = True
        last_state_token = 0
        while token != self.EOS and self.model.token_length < self.model.SEQLEN:
             # 新增：在每次循环开始时检查中断标志
            if globalConfig.rag_task_active:
                print("\n[INFO] Mermaid generation interrupted by high-priority RAG task.", flush=True)
                interrupted = True
                break  # 立即退出生成循环

            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in word:
                token = self.model.forward_next()
                tok_num += 1
                continue
            self.answer_token += full_word_tokens
            print(word, flush=True, end="")
            # print(repr(word), flush=True, end="")
            tok_num += 1
            full_word_tokens = []
            token = self.model.forward_next()

            if(last_state and not globalConfig.running):
                last_state_token = tok_num
                last_state = False
            # is_newline = (word == '\n')
            # print(f"\n {is_newline}yes{tok_num} tokens {last_state_token} last_state_tok {last_state}||{globalConfig.running}")
            if((tok_num > maxWord or not globalConfig.running)and "\n" in word):
                print("...find \n (reach the maximal length)", flush=True, end="")#可能存在bug不过考虑到\n一般出现在最后
                break
            if(tok_num -last_state_token > 50 and not globalConfig.running):
                print("... (reach the maximal length break)", flush=True, end="")
                break
            if(tok_num > maxWord+50):
                print("... (reach the maximal length)", flush=True, end="")
                break
            

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration


        if self.enable_history:
            self.answer_cur = self.tokenizer.decode(self.answer_token)
            self.update_history()
        else:
            self.clear()
        
        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")
        result = self.tokenizer.decode(self.answer_token)
        if interrupted:
            print("⚠️ 思维导图生成被RAG中断，返回空结果")
            return ""  # 标记为被中断的内容直接返回空
        else:
            return result

    def stream_answer_RAG(self, tokens, maxWord=100000):
        """
        Stream the answer for the given tokens. (修正版)
        """
        # 初始化部分保持不变
        tok_num = 0
        first_start = time.time()
        token = self.model.forward_first(tokens)
        first_end = time.time()

        # Following tokens
        full_word_tokens = []
        
        while token != self.EOS and self.model.token_length < self.model.SEQLEN:
            # 1. 累加当前的 token
            full_word_tokens.append(token)
            # 2. 尝试解码
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)

            # 3. 检查解码是否完整。如果包含替换字符'�'，说明是部分解码，
            #    需要获取更多token才能组成一个完整的字符。
            if "�" in word:
                # 如果不完整，则获取下一个token，然后继续下一次循环以累加
                token = self.model.forward_next()
                tok_num += 1
                continue

            # 4. 如果代码能执行到这里，说明'word'是一个可读的、完整的片段。
            #    我们将其产出。
            yield word
            
            # 5. 重置 token 缓冲区，为下一个词做准备
            full_word_tokens = []

            # 6. 【关键】为下一次循环获取新的 token
            token = self.model.forward_next()
            tok_num += 1

        # 循环结束后，可以打印一些统计信息，但这对于流式函数不是必需的
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        if next_duration > 0:
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s, TPS: {tps:.3f} token/s")

        # 对于生成器，不需要返回任何值，也不需要处理 history

def main():
    # 固定参数，无需命令行解析
    class Args:
        model_path = "/data/qwen4btune_w4bf16_seq8192_bm1684x_1dev_20250721_195513.bmodel"
        config_path = "config"
        devid = "0"
        temperature = 1.0
        top_p = 1.0
        repeat_penalty = 1.2
        repeat_last_n = 64
        max_new_tokens = 1024
        generation_mode = "greedy"
        prompt_mode = "prompted"
        enable_history = False  # 或 False，根据需要

    args = Args()
    model = Qwen2(args)
    print("--- 开始测试流式输出 ---")
    # 调用现在是生成器的 generate_text
    text_generator = model.generate_text("你好，请介绍一下你自己。")
    for word in text_generator:
        print(word, end="", flush=True)
    print("\n--- 测试结束 ---")


if __name__ == "__main__":
    main()