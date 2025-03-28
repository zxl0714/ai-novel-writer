import os
import re
from typing import Optional, List, Dict, Any
import json

OUTPUT_DIR = "book_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_content(filename: str, content: str):
    """保存内容到文件"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"内容已保存到: {filepath}")
    except IOError as e:
        print(f"错误：无法写入文件 {filepath}: {e}")

def load_content(filename: str) -> Optional[str]:
    """从文件加载内容"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"警告：文件未找到 {filepath}")
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except IOError as e:
        print(f"错误：无法读取文件 {filepath}: {e}")
        return None

def extract_code_block(text: str, language: str = "json") -> Optional[str]:
    """从文本中提取特定语言的代码块"""
    pattern = rf"```{language}\s*([\s\S]+?)\s*```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    # Fallback: 如果没有标记，但看起来像 JSON
    if language == "json" and text.strip().startswith("{") and text.strip().endswith("}"):
        return text.strip()
    if language == "json" and text.strip().startswith("[") and text.strip().endswith("]"):
         return text.strip()
    return None

def parse_json_content(text: Optional[str]) -> Optional[Any]:
    """解析可能是 JSON 格式的文本"""
    if not text:
        return None
    json_text = extract_code_block(text, "json")
    if not json_text:
        # 如果没有代码块标记，尝试直接解析原始文本
        json_text = text
        # 移除可能的解释性前缀
        if "【角色档案】" in json_text:
            json_text = json_text.split("【角色档案】", 1)[-1]

    try:
        # 尝试去掉末尾可能不属于 JSON 的内容
        # 注意：这比较脆弱，最好是让 LLM 输出干净的 JSON
        json_text = json_text.strip()
        # 找到最后一个 '}' 或 ']'
        last_brace = json_text.rfind('}')
        last_bracket = json_text.rfind(']')
        end_index = max(last_brace, last_bracket)
        if end_index != -1:
            json_text = json_text[:end_index + 1]

        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"错误：解析 JSON 内容失败: {e}")
        print(f"原始文本 (前500字符):\n{text[:500]}")
        return None
    except Exception as e:
        print(f"解析内容时发生未知错误: {e}")
        print(f"原始文本 (前500字符):\n{text[:500]}")
        return None


def extract_section(text: str, start_tag: str, end_tag: Optional[str] = None) -> Optional[str]:
    """根据开始和结束标签提取文本区域"""
    start_index = text.find(start_tag)
    if start_index == -1:
        return None
    start_index += len(start_tag)

    if end_tag:
        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            # 如果找不到结束标签，就取到文本末尾
             content = text[start_index:].strip()
        else:
            content = text[start_index:end_index].strip()
    else:
        content = text[start_index:].strip()

    # 去除可能的结束标记自身残留，例如【草稿完】
    if content.endswith("完】"):
         last_bracket_index = content.rfind("【")
         if last_bracket_index != -1:
             content = content[:last_bracket_index].strip()

    return content if content else None