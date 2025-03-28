import autogen
from typing import Optional, List, Dict, Any, Union
from agents import (
    get_memory_keeper_agent, get_writer_agent, get_editor_agent, get_user_proxy_agent
)
from config import llm_config
from utils import save_content, extract_section
import os
import time

MAX_CHAPTER_ATTEMPTS = 2 # 每章最多尝试次数
MAX_WRITER_EDITOR_ROUNDS = 3 # 写手和编辑之间的最大交互轮次

# 定义自定义发言人选择逻辑
def custom_speaker_selection_func(
    last_speaker: autogen.Agent, groupchat: autogen.GroupChat
) -> Union[autogen.Agent, str, None]:
    """
    根据最后发言者和内容决定下一个发言者。
    """
    messages = groupchat.messages
    last_message = messages[-1]

    # 初始情况，User Proxy 发起
    if last_speaker.name == "用户代理":
        # 下一个必须是写手
        return groupchat.agent_by_name("写手")

    # 写手刚完成草稿 (第一次或修改稿)
    elif last_speaker.name == "写手":
        # 下一个必须是编辑来审阅
        return groupchat.agent_by_name("编辑")

    # 编辑刚完成审阅
    elif last_speaker.name == "编辑":
        content = last_message.get("content", "").strip()
        if "【编辑通过】" in content:
            # 编辑通过了，对话结束，可以返回 User Proxy 或 None
            print("编辑已通过，本章结束。")
            return None # 返回 None 结束当前轮次的群聊
        elif "【编辑反馈】" in content:
            # 编辑提出了反馈，需要写手修改
            print("编辑提出反馈，转回写手修改。")
            return groupchat.agent_by_name("写手")
        else:
            # 编辑的回复格式不确定，默认让写手处理或结束？这里先结束避免死循环
            print("警告：编辑回复格式未知，结束本章。")
            return None

    # 其他意外情况，结束对话
    else:
        return None

class BookGenerator:
    def __init__(self, final_outline: List[Dict], character_profiles: str, world_details: str):
        self.final_outline = final_outline
        # 解析角色档案为结构化数据可能更好，但这里简化为文本
        self.character_profiles = character_profiles
        self.world_details = world_details # 加入世界设定以便参考
        self.chapters_memory: List[str] = [] # 存储每章的概要或关键信息
        self.output_dir = "book_output"
        os.makedirs(os.path.join(self.output_dir, "chapters"), exist_ok=True)

        # 初始化核心写作 Agent
        self.memory_keeper = get_memory_keeper_agent()
        self.writer = get_writer_agent()
        self.editor = get_editor_agent()
        self.user_proxy = get_user_proxy_agent()
        # 可以为 Writer_Final 单独创建 agent，或让 Writer 根据编辑反馈再次运行
        # 这里简化，让 Writer 在接收到编辑反馈后再次生成
        # 确保这里使用的名字和函数定义的一致
        assert self.writer.name == "写手"
        assert self.editor.name == "编辑"
        assert self.user_proxy.name == "用户代理"

    def _get_chapter_context(self, chapter_number: int) -> str:
        """准备当前章节需要的上下文信息"""
        context_parts = []
        if self.chapters_memory:
            context_parts.append("【先前章节概要回顾 (由记忆守护者提供)】")
            # 只包含最近 N 章的概要可能更有效，避免上下文过长
            max_prev_chapters = 3
            start_index = max(0, len(self.chapters_memory) - max_prev_chapters)
            for i, summary in enumerate(self.chapters_memory[start_index:], start=start_index + 1):
                context_parts.append(f"第 {i} 章概要:\n{summary}")
            context_parts.append("-" * 20)

        # 可以在这里让 Memory Keeper 动态生成上下文
        # simplified_memory_prompt = f"请基于已完成的 {len(self.chapters_memory)} 章内容，为即将开始的第 {chapter_number} 章提供关键上下文总结和连续性提醒。"
        # ... (call memory keeper) ...

        return "\n".join(context_parts)

    def _generate_chapter(self, chapter_data: Dict) -> Optional[str]:
        """生成单章内容，包含写手和编辑的交互"""
        chapter_number = chapter_data["chapter_number"]
        chapter_title = chapter_data["title"]
        chapter_prompt = chapter_data["prompt"] # 大纲中该章的具体要求

        print(f"\n--- 开始生成 第 {chapter_number} 章: {chapter_title} ---")

        context = self._get_chapter_context(chapter_number)
        full_chapter_task_prompt = f"""现在开始创作 **第 {chapter_number} 章: {chapter_title}**。

【记忆守护者提供的上下文回顾】
{context if context else "这是第一章，没有先前内容。"}

【角色档案参考】(节选或全部)
{self.character_profiles[:1500]}...

【世界设定参考】(节选或全部)
{self.world_details[:1500]}...

【本章大纲要求】
{chapter_prompt}
---
写作流程指示：
1.  **写手 (Writer):** 请根据以上所有信息，创作本章初稿，输出格式为【章节草稿】...【草稿完】。
2.  **编辑 (Editor):** 请审阅【章节草稿】，输出【编辑通过】或【编辑反馈】+【修改建议】。
3.  **写手 (Writer):** 如果收到【编辑反馈】，请根据【修改建议】修改草稿，并再次以【章节草稿】...【草稿完】格式输出最终版本。

请严格按流程进行。写手请注意保持角色一致性并达到目标长度（约1500字）。
"""
        final_chapter_content = None
        # 注意这里的 Agent 列表名称需要和 custom_speaker_selection_func 中使用的名称一致
        agents = [self.user_proxy, self.writer, self.editor]

        group_chat = autogen.GroupChat(
            agents=agents,
            messages=[], # 初始消息将在 initiate_chat 中提供
            max_round=1 + MAX_WRITER_EDITOR_ROUNDS * 2, # 保持轮次限制
            # 使用自定义的发言人选择函数
            speaker_selection_method=custom_speaker_selection_func,
            allow_repeat_speaker=False # 通常不允许连续发言，除非逻辑需要
        )
        manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

        try:
            # 初始消息仍然由 User Proxy 发起
            self.user_proxy.initiate_chat(
                manager,
                message=full_chapter_task_prompt,
                # 注意：initiate_chat 会自动将 message 添加到 history
                # 这里不需要在 GroupChat 的 messages 里预先加入
            )

            # 从聊天历史中提取最终的章节内容
            # 查找最后一个由 Writer 发送的 【章节草稿】
            final_content = None
            for msg in reversed(group_chat.messages):
                 # 检查发送者是否是写手 (根据 agent name)
                 sender_name = msg.get("name")
                 if sender_name == self.writer.name:
                    content = msg.get("content", "")
                    extracted = extract_section(content, "【章节草稿】", "【草稿完】")
                    if extracted:
                        final_content = extracted
                        print(f"第 {chapter_number} 章：找到最终草稿。")
                        break # 找到最新的就停止

            if final_content:
                 final_chapter_content = final_content
                 # TODO: 可以调用 Memory Keeper 来生成本章概要
                 # self.chapters_memory.append(f"第 {chapter_number} 章概要：{final_content[:200]}...") # 简化：使用内容前缀
                 # 这里调用 Memory Keeper 生成更佳
                 memory_summary_prompt = f"请为以下刚完成的第 {chapter_number} 章内容生成一个简洁的摘要（包含关键事件和角色状态变化），用于后续章节的上下文：\n\n{final_content[:1000]}..." # 限制长度
                 try:
                     summary_response = self.memory_keeper.generate_reply(messages=[{"role":"user", "content": memory_summary_prompt}])
                     # 解析 summary_response 来获取概要
                     if isinstance(summary_response, str) and summary_response.strip():
                         # 简单提取，或者让 Memory Keeper 输出带标记的摘要
                         self.chapters_memory.append(summary_response.strip())
                         print(f"第 {chapter_number} 章概要已存入记忆库。")
                     else:
                         print(f"警告：未能从 Memory Keeper 获取第 {chapter_number} 章的有效概要。")
                         self.chapters_memory.append(f"第 {chapter_number} 章概要：{final_content[:200]}...") # Fallback
                 except Exception as mem_e:
                     print(f"调用 Memory Keeper 生成概要时出错：{mem_e}")
                     self.chapters_memory.append(f"第 {chapter_number} 章概要：{final_content[:200]}...") # Fallback
            else:
                 print(f"警告：未能从聊天记录中提取第 {chapter_number} 章的最终内容。")


        except Exception as e:
            print(f"生成第 {chapter_number} 章时发生错误: {e}")
            # 可以加入重试逻辑

        return final_chapter_content


    def generate_book(self):
        """按顺序生成所有章节"""
        print("\n--- 开始生成书籍章节 ---")
        total_chapters = len(self.final_outline)
        for i, chapter_data in enumerate(self.final_outline):
            chapter_number = chapter_data["chapter_number"]
            print(f"\n处理中：第 {chapter_number}/{total_chapters} 章")

            attempt = 0
            generated_content = None
            while attempt < MAX_CHAPTER_ATTEMPTS and generated_content is None:
                attempt += 1
                print(f"第 {chapter_number} 章 - 尝试次数 {attempt}/{MAX_CHAPTER_ATTEMPTS}")
                generated_content = self._generate_chapter(chapter_data)
                if generated_content:
                    # 保存章节内容
                    filename = f"chapters/chapter_{chapter_number:03d}.txt"
                    full_content_to_save = f"第 {chapter_number} 章: {chapter_data['title']}\n\n{generated_content}"
                    save_content(filename, full_content_to_save)
                    print(f"第 {chapter_number} 章生成并保存成功。")
                    break # 成功则跳出重试循环
                else:
                    print(f"第 {chapter_number} 章尝试 {attempt} 失败。")
                    time.sleep(5) # 失败后稍作等待

            if generated_content is None:
                print(f"错误：生成第 {chapter_number} 章失败，已达到最大尝试次数。")
                # 可以选择停止或跳过
                print("跳过此章节，继续下一章...")
                self.chapters_memory.append(f"第 {chapter_number} 章生成失败，内容缺失。") # 记录失败状态

            # 短暂休眠，避免过于频繁地调用 API
            time.sleep(10)

        print("\n--- 书籍章节生成完成 ---")
        # 最后可以合并所有章节到一个文件
        self._combine_chapters()

    def _combine_chapters(self):
         """合并所有章节到一个文件"""
         print("\n--- 合并所有章节 ---")
         full_book_content = []
         chapter_dir = os.path.join(self.output_dir, "chapters")
         try:
             files = sorted([f for f in os.listdir(chapter_dir) if f.startswith("chapter_") and f.endswith(".txt")])
             for filename in files:
                 filepath = os.path.join(chapter_dir, filename)
                 try:
                     with open(filepath, "r", encoding="utf-8") as f:
                         full_book_content.append(f.read())
                     print(f"已加载: {filename}")
                 except IOError as e:
                     print(f"错误：无法读取章节文件 {filepath}: {e}")

             if full_book_content:
                 save_content("full_book_combined.txt", "\n\n---\n\n".join(full_book_content))
                 print("所有章节已合并到 full_book_combined.txt")
             else:
                 print("没有找到可合并的章节文件。")

         except FileNotFoundError:
             print(f"错误：找不到章节目录 {chapter_dir}")
         except Exception as e:
              print(f"合并章节时发生错误: {e}")