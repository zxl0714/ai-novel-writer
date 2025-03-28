import os
import sys
from dotenv import load_dotenv
from typing import Dict, List, Optional

# 加载 .env 文件
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "OPENAI").upper() # 默认为 OPENAI

def get_llm_config(timeout: int = 600, temperature: float = 0.7) -> Optional[Dict]:
    """
    获取LLM配置，优先从环境变量读取。
    支持 OpenAI 和 Dashscope (通义千问)。
    """
    config_list: List[Dict] = []

    if LLM_PROVIDER == "OPENAI":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("MODEL_NAME", "gpt-4-turbo")
        if not api_key:
            print("错误：未找到 OPENAI_API_KEY 环境变量。")
            return None
        config_list.append({
            "model": model,
            "api_key": api_key,
            "base_url": base_url, # 如果使用代理或非官方地址，请设置
            "api_type": "openai" # 或 "azure"
        })
    elif LLM_PROVIDER == "DASHSCOPE":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        model = os.getenv("MODEL_NAME", "qwen-max") # 默认使用 qwen-max
        if not api_key:
            print("错误：未找到 DASHSCOPE_API_KEY 环境变量。")
            return None
        # Dashscope 通常通过兼容 OpenAI 的方式或其自己的 SDK 使用
        # 这里我们配置为兼容 OpenAI SDK 的模式
        config_list.append({
            "model": model,
            "api_key": api_key,
            # 对于 Dashscope 兼容模式，通常需要指定 base_url
            "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "api_type": "openai" # 即使是 Dashscope，用 OpenAI 兼容模式时 api_type 仍可以是 openai
        })
    elif LLM_PROVIDER == "SHUZHI":
        api_key = ""
        model = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1") # 默认使用 qwen-max
        config_list.append({
            "model": model,
            "api_key": api_key,
            "base_url": "http://100.67.155.217/sglang2/v1",
            "api_type": "openai" 
        })
    elif LLM_PROVIDER == "DEEPSEEK":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        model = os.getenv("MODEL_NAME", "deepseek-chat")
        if not api_key:
            print("错误：未找到 DEEPSEEK_API_KEY 环境变量。")
            return None
        config_list.append({
            "model": model,
            "api_key": api_key,
            # 对于 Dashscope 兼容模式，通常需要指定 base_url
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "api_type": "openai"
        })
    else:
        print(f"错误：不支持的 LLM_PROVIDER: {LLM_PROVIDER}")
        return None

    if not config_list:
         print("错误：无法构建有效的 LLM 配置列表。")
         return None

    return {
        "timeout": timeout,
        "temperature": temperature,
        "config_list": config_list,
        "cache_seed": None,
    }

# 通用 Agent 配置模板
DEFAULT_AGENT_CONFIG = {
    "human_input_mode": "NEVER", # 全自动运行，无人工输入
    "max_consecutive_auto_reply": 10, # 限制连续自动回复次数
    "code_execution_config": False, # 默认禁用代码执行以保证安全
    # "use_docker": False # 如果需要代码执行，可以配置
}

# 获取并检查 LLM 配置
llm_config = get_llm_config()
if llm_config is None:
    print("LLM 配置加载失败，请检查 .env 文件和环境变量。")
    sys.exit(1) # 退出程序

print(f"使用的 LLM Provider: {LLM_PROVIDER}")
print(f"使用的模型: {llm_config['config_list'][0]['model']}")
