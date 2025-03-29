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

REASONING_LLM_PROVIDER = os.getenv("REASONING_LLM_PROVIDER", LLM_PROVIDER).upper() # 默认使用与上面相同的 Provider

def get_reasoning_llm_config(timeout: int = 1200, temperature: float = 0.5) -> Optional[Dict]:
    """获取推理LLM配置 (可能使用更强但更慢的模型)"""
    config_list: List[Dict] = []
    api_key = None
    base_url = None
    model = None

    # 优先使用推理专用的环境变量
    reasoning_api_key_env = f"{REASONING_LLM_PROVIDER}_API_KEY" # 例如 DASHSCOPE_API_KEY 或 REASONING_OPENAI_API_KEY
    reasoning_base_url_env = f"{REASONING_LLM_PROVIDER}_BASE_URL"
    reasoning_model_env = f"REASONING_MODEL_NAME" # 通用变量名指定模型

    # 尝试读取推理模型的特定环境变量，如果不存在则尝试读取默认模型的环境变量
    api_key = os.getenv(f"REASONING_{reasoning_api_key_env}", os.getenv(reasoning_api_key_env))
    base_url = os.getenv(f"REASONING_{reasoning_base_url_env}", os.getenv(reasoning_base_url_env))
    model = os.getenv(reasoning_model_env) # 必须指定推理模型名称

    if not model: # 如果没有指定推理模型，则无法使用
        print("信息：未在 .env 中指定 REASONING_MODEL_NAME，推理 Agent 将回退使用默认模型。")
        return None
    if not api_key:
        print(f"警告：未找到推理模型 API Key (环境变量: REASONING_{reasoning_api_key_env} 或 {reasoning_api_key_env})，推理 Agent 可能无法工作。")
        # 即使没有 key 也尝试返回配置，让 AutoGen 处理错误
        # return None

    if REASONING_LLM_PROVIDER == "OPENAI":
        if not base_url: base_url = os.getenv("OPENAI_BASE_URL") # 继承默认 URL
        config_list.append({
            "model": model, "api_key": api_key, "base_url": base_url, "api_type": "openai"
        })
    elif REASONING_LLM_PROVIDER == "DASHSCOPE":
        if not base_url: base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1") # 继承默认 URL
        config_list.append({
            "model": model, "api_key": api_key, "base_url": base_url, "api_type": "openai"
        })
    elif REASONING_LLM_PROVIDER == "DEEPSEEK":
        if not base_url: base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com") # 继承默认 URL
        config_list.append({
            "model": model, "api_key": api_key, "base_url": base_url, "api_type": "openai"
        })
    # --- 可以添加其他 Provider 的逻辑 ---
    else:
        print(f"警告：不支持的 REASONING_LLM_PROVIDER: {REASONING_LLM_PROVIDER}")
        return None # 不支持则无法使用

    if not config_list:
         print("错误：无法为推理模型构建有效的配置列表。")
         return None

    print(f"推理模型配置加载: Provider={REASONING_LLM_PROVIDER}, Model={model}")
    return {
        "timeout": timeout, # 推理可能需要更长时间
        "temperature": temperature, # 推理任务通常用较低温度
        "config_list": config_list,
        "cache_seed": None, # 确认禁用缓存
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
reasoning_llm_config = get_reasoning_llm_config() # 获取推理模型配置

# 如果推理配置加载失败，用默认配置替代并给出提示
if reasoning_llm_config is None:
    print("警告：推理 LLM 配置加载失败或未指定，所有 Agent 将使用默认 LLM 配置。")
    reasoning_llm_config_to_use = llm_config
else:
    reasoning_llm_config_to_use = reasoning_llm_config

print(f"默认模型配置: Provider={LLM_PROVIDER}, Model={llm_config['config_list'][0]['model']}")
print(f"推理 Agent 将使用的模型配置: Provider={REASONING_LLM_PROVIDER}, Model={reasoning_llm_config_to_use['config_list'][0]['model']}")
