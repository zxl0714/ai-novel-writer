# outline_module.py (再次修正 generate_refined_outline 函数)

import autogen
from typing import Optional, List, Dict, Tuple
from agents import (
    get_story_planner_agent, get_world_builder_agent,
    get_character_creator_agent, get_outline_creator_agent,
    get_outline_editor_agent, get_user_proxy_agent
)
from config import llm_config
from utils import save_content, extract_section, OUTPUT_DIR, load_content
import os
import time
import re

MAX_FULL_REFINEMENT_ITERATIONS = 4

# _parse_outline 函数保持不变 (假设上次已包含 import re)
def _parse_outline(outline_text: Optional[str], num_chapters: int) -> Optional[List[Dict]]:
    # ... (之前的 _parse_outline 实现) ...
    if not outline_text:
        print("错误：传入的 outline_text 为空，无法解析。")
        return None

    chapters = []
    # 改进正则表达式以更好地处理换行和可选的空格
    chapter_pattern = re.compile(
        # 匹配行首、可选空格、**、可选空格、第、数字、章、可选空格、冒号(半/全角)、可选空格、标题、可选空格、**、可选空格、行尾
        r"^\s*\*\*\s*第\s*(\d+)\s*章\s*[:：]\s*(.*?)\s*\*\*\s*$"
        r"([\s\S]*?)" # 非贪婪匹配章节内容
        # Lookahead: 直到下一个章节标题(同样允许两种冒号) 或 【大纲完】(行首) 或 文本末尾(\Z)
        r"(?=(?:^\s*\*\*\s*第\s*\d+\s*章\s*[:：])|(?:^\s*【大纲完】\s*$)|\Z)",
        re.MULTILINE
    )

    # 在解析前先移除可能的干扰性结尾标记，如果【大纲完】不在行首
    clean_outline_text = outline_text.strip()
    end_marker = "【大纲完】"
    if clean_outline_text.endswith(end_marker):
         clean_outline_text = clean_outline_text[:-len(end_marker)].strip()

    matches = chapter_pattern.finditer(clean_outline_text)
    processed_indices = set()
    last_match_end = 0

    for match in matches:
        last_match_end = match.end()
        # group(1) 是章节号, group(2) 是标题, group(3) 是内容
        chapter_num_str = match.group(1)
        chapter_title = match.group(2).strip()
        chapter_content = match.group(3).strip()

        try:
            chapter_num = int(chapter_num_str)
        except ValueError:
            print(f"警告：无法解析章节号 '{chapter_num_str}'")
            continue

        if chapter_num in processed_indices:
            # print(f"警告：重复解析到第 {chapter_num} 章，已跳过。") # 减少冗余输出
            continue

        chapter_data = {"chapter_number": chapter_num, "title": chapter_title}

        # 提取各个部分 - 使用更健壮的方式处理多行和边界情况
        def safe_extract(content, start_key, keys_order):
            start_tag = f"{start_key}:"
            start_idx = content.find(start_tag)
            if start_idx == -1:
                # 尝试不带冒号的 key
                start_tag_no_colon = f"{start_key}"
                start_idx = content.find(start_tag_no_colon)
                if start_idx == -1: return None
                start_tag = start_tag_no_colon # 更新 tag
            
            content_start = start_idx + len(start_tag)
            
            # 找到下一个 key 的起始位置作为结束边界
            end_idx = len(content) # 默认为内容末尾
            current_key_index_in_order = keys_order.index(start_key) if start_key in keys_order else -1
            
            if current_key_index_in_order != -1:
                for next_key_index in range(current_key_index_in_order + 1, len(keys_order)):
                    next_key = keys_order[next_key_index]
                    next_key_tag = f"{next_key}:"
                    next_key_tag_no_colon = f"{next_key}"

                    # 优先查找带冒号的下一个 key
                    next_key_idx = content.find(next_key_tag, content_start)
                    if next_key_idx != -1:
                         # 向上查找最近的换行符，确保key在行首附近
                         newline_before = content.rfind('\n', content_start, next_key_idx)
                         if newline_before != -1 and content[newline_before:next_key_idx].strip().startswith(next_key):
                              end_idx = newline_before # 结束于换行符
                              break
                         elif content[next_key_idx:].strip().startswith(next_key_tag): # 确保找到的是 key 开头
                              end_idx = next_key_idx
                              break

                    # 如果没找到带冒号的，尝试不带冒号的
                    next_key_idx = content.find(next_key_tag_no_colon, content_start)
                    if next_key_idx != -1:
                         newline_before = content.rfind('\n', content_start, next_key_idx)
                         if newline_before != -1 and content[newline_before:next_key_idx].strip().startswith(next_key):
                              end_idx = newline_before
                              break
                         elif content[next_key_idx:].strip().startswith(next_key_tag_no_colon):
                               end_idx = next_key_idx
                               break
            
            # 如果是最后一个 key，或者没找到下一个 key，取到末尾
            return content[content_start:end_idx].strip()

        keys_in_order = ["本章目标", "关键事件", "角色发展", "场景设定", "基调", "引出/伏笔"]
        target = safe_extract(chapter_content, "本章目标", keys_in_order)
        events_text = safe_extract(chapter_content, "关键事件", keys_in_order)
        dev = safe_extract(chapter_content, "角色发展", keys_in_order)
        setting = safe_extract(chapter_content, "场景设定", keys_in_order)
        tone = safe_extract(chapter_content, "基调", keys_in_order)
        foreshadow = safe_extract(chapter_content, "引出/伏笔", keys_in_order)


        # 解析关键事件 (假设以 - 或数字. 开头)
        events = []
        if events_text:
            lines = events_text.split('\n')
            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith('-') or re.match(r'^\s*\d+\.', stripped_line):
                    # 去掉前缀
                    event_content = re.sub(r'^[-\s*\d\.]+\s*', '', stripped_line)
                    if event_content:
                         events.append(event_content)

        # 组合 prompt
        prompt_lines = []
        # 保持原始格式，只是确保有内容才添加
        if target: prompt_lines.append(f"- 本章目标: {target}")
        if events:
            formatted_events = "\n".join([f"    - {e}" for e in events])
            prompt_lines.append(f"- 关键事件:\n{formatted_events}")
        # 角色发展可能本身就是多行，直接使用
        if dev: prompt_lines.append(f"- 角色发展:\n{dev}")
        if setting: prompt_lines.append(f"- 场景设定: {setting}")
        if tone: prompt_lines.append(f"- 基调: {tone}")
        if foreshadow: prompt_lines.append(f"- 引出/伏笔: {foreshadow}")

        chapter_data["prompt"] = "\n".join(prompt_lines)

        # 基本验证（可选，减少打印）
        # if len(events) < 2: print(f"警告：第 {chapter_num} 章关键事件少于2个。")
        # if not all([target, events_text, dev, setting, tone]): print(f"警告：第 {chapter_num} 章缺少必要元素。")


        chapters.append(chapter_data)
        processed_indices.add(chapter_num)

    # 检查是否有未匹配的文本（可能格式错误）
    remaining_text = clean_outline_text[last_match_end:].strip()
    # 移除可能的 【大纲完】及其前的空白
    if remaining_text.endswith("【大纲完】"):
         remaining_text = remaining_text[:-len("【大纲完】")].strip()
         
    if remaining_text:
         print(f"警告：解析完成后仍有未匹配的文本，可能部分大纲格式错误或丢失:\n{remaining_text[:200]}...")

    # 检查章节数量
    if len(chapters) != num_chapters:
        print(f"警告：解析出 {len(chapters)} 章，与要求的 {num_chapters} 章不符。")
        chapters.sort(key=lambda x: x["chapter_number"])


    if not chapters:
        print("错误：未能从文本中解析出任何有效的章节信息。")
        return None

    print(f"成功解析出 {len(chapters)} 章。")
    return chapters


def generate_refined_outline(
    initial_prompt: str,
    num_chapters: int,
) -> Tuple[Optional[List[Dict]], Optional[str], Optional[str], Optional[str]]:
    """
    生成并精炼故事大纲、角色、世界和规划。
    包含完整的规划、设定、角色、大纲创建和编辑的迭代循环。
    修正了上一轮输出未传递给下一轮修改的问题。
    返回: (最终大纲结构, 最终角色档案文本, 最终世界设定文本, 最终故事规划文本) 或 (None, None, None, None)
    """
    print("\n--- 开始完整的大纲精炼流程 ---")
    user_proxy = get_user_proxy_agent()
    planner = get_story_planner_agent(num_chapters=num_chapters)
    world_builder = get_world_builder_agent()
    char_creator = get_character_creator_agent()
    outline_creator = get_outline_creator_agent(num_chapters)
    outline_editor = get_outline_editor_agent(num_chapters)

    current_story_plan = None
    current_world_details = None
    current_character_profiles = None
    current_outline_text = None
    last_editor_feedback = "无"

    previous_story_plan = None
    previous_world_details = None
    previous_character_profiles = None
    previous_outline_text = None

    final_approved_outline_structure = None

    for i in range(MAX_FULL_REFINEMENT_ITERATIONS):
        print(f"\n--- 第 {i+1}/{MAX_FULL_REFINEMENT_ITERATIONS} 轮精炼 ---")
        iteration_success_flags = {"planner": False, "builder": False, "char_creator": False, "outline_creator": False, "editor": False}

        # --- 1. 故事规划 ---
        print("步骤 1: 生成/修改故事规划...")
        if i == 0:
             planner_prompt = f"""请根据以下初始想法进行故事规划。
【用户初始想法】
{initial_prompt}
请严格按照你的系统提示格式输出。"""
        else:
             # 使用上一轮的 'current' 作为这一轮的 'previous'
             prompt_previous_plan = previous_story_plan if previous_story_plan else "无"
             planner_prompt = f"""请根据以下信息，修改或完善故事规划。
【用户初始想法】
{initial_prompt}
【上一轮编辑反馈】
{last_editor_feedback}
【你上一轮生成的故事规划】(请在此基础上修改)
{prompt_previous_plan}
请重点解决编辑反馈中与故事规划相关的问题，并输出修改后的**完整**故事规划，严格遵循格式。"""

        try:
            user_proxy.reset()
            user_proxy.initiate_chat(recipient=planner, message=planner_prompt, max_turns=1, clear_history=True)
            plan_msg = user_proxy.last_message(planner)
            if plan_msg and plan_msg["content"]:
                current_story_plan = plan_msg["content"] # 直接用新结果更新 current
                save_content(f"iteration_{i+1}_story_plan.txt", current_story_plan)
                print("故事规划已更新。")
                iteration_success_flags["planner"] = True
            else:
                print("错误：未能从 Story Planner 获取有效回复。")
                # 保持 current_story_plan 不变 (即使用上一轮的值)
        except Exception as e:
            print(f"生成故事规划时出错: {e}")
            # 保持 current_story_plan 不变

        # --- 2. 世界设定 ---
        print("步骤 2: 生成/修改世界设定...")
        if current_story_plan is None: # 检查依赖的上游是否成功
            print("错误：缺少故事规划信息，无法生成世界设定。")
        else:
            if i == 0:
                world_builder_prompt = f"""请根据以下信息构建世界设定：
【用户初始想法】
{initial_prompt}
【本轮最新的故事规划】
{current_story_plan}
请严格按照你的系统提示格式输出。"""
            else:
                prompt_previous_world = previous_world_details if previous_world_details else "无"
                world_builder_prompt = f"""请根据以下信息，修改或完善世界设定。
【用户初始想法】
{initial_prompt}
【本轮最新的故事规划】
{current_story_plan}
【上一轮编辑反馈】
{last_editor_feedback}
【你上一轮生成的世界设定】(请在此基础上修改)
{prompt_previous_world}
请重点解决编辑反馈中与世界设定相关的问题，结合最新的故事规划，输出修改后的**完整**世界设定，严格遵循格式。"""
            try:
                user_proxy.reset()
                user_proxy.initiate_chat(recipient=world_builder, message=world_builder_prompt, max_turns=1, clear_history=True)
                world_msg = user_proxy.last_message(world_builder)
                if world_msg and world_msg["content"]:
                    current_world_details = world_msg["content"]
                    save_content(f"iteration_{i+1}_world_details.txt", current_world_details)
                    print("世界设定已更新。")
                    iteration_success_flags["builder"] = True
                else:
                    print("错误：未能从 World Builder 获取有效回复。")
            except Exception as e:
                 print(f"生成世界设定时出错: {e}")

        # --- 3. 角色创建 ---
        print("步骤 3: 生成/修改角色档案...")
        if current_story_plan is None or current_world_details is None:
            print("错误：缺少规划或设定信息，无法生成角色档案。")
        else:
            if i == 0:
                 char_creator_prompt = f"""请根据以下信息创建角色档案：
【用户初始想法】
{initial_prompt}
【本轮最新的故事规划】
{current_story_plan}
【本轮最新的世界设定】
{current_world_details}
请严格按照你的系统提示格式输出，并确保包含【角色档案】标记。"""
            else:
                 prompt_previous_chars = previous_character_profiles if previous_character_profiles else "无"
                 char_creator_prompt = f"""请根据以下信息，修改或完善角色档案。
【用户初始想法】
{initial_prompt}
【本轮最新的故事规划】
{current_story_plan}
【本轮最新的世界设定】
{current_world_details}
【上一轮编辑反馈】
{last_editor_feedback}
【你上一轮生成的角色档案】(请在此基础上修改)
{prompt_previous_chars}
请重点解决编辑反馈中与角色相关的问题，结合最新的规划和设定，输出修改后的**完整**角色档案，严格遵循格式并包含【角色档案】标记。"""

            try:
                user_proxy.reset()
                user_proxy.initiate_chat(recipient=char_creator, message=char_creator_prompt, max_turns=1, clear_history=True)
                profile_msg = user_proxy.last_message(char_creator)
                if profile_msg and profile_msg["content"]:
                     profiles_text = profile_msg["content"]
                     extracted_profiles = extract_section(profiles_text, "【角色档案】")
                     if extracted_profiles:
                         current_character_profiles = extracted_profiles
                         print("角色档案已更新 (提取)。")
                     else:
                         print("警告：Character Creator 回复中未找到【角色档案】标记，将使用完整回复。")
                         current_character_profiles = profiles_text # 使用完整回复作为档案
                     save_content(f"iteration_{i+1}_character_profiles.txt", current_character_profiles)
                     iteration_success_flags["char_creator"] = True
                else:
                     print("错误：未能从 Character Creator 获取有效回复。")
            except Exception as e:
                print(f"生成角色档案时出错: {e}")

        # --- 4. 大纲创作 ---
        print("步骤 4: 生成/修改章节大纲...")
        if current_story_plan is None or current_world_details is None or current_character_profiles is None:
             print("错误：缺少规划/设定/角色信息，无法生成大纲。")
        else:
            if i == 0:
                outline_creator_prompt = f"""请根据以下所有最新信息，创作一个包含 {num_chapters} 章的详细大纲：
【用户初始想法】
{initial_prompt}
【本轮最新的故事规划】
{current_story_plan}
【本轮最新的世界设定】
{current_world_details}
【本轮最新的角色档案】
{current_character_profiles}
请严格按照你的系统提示中定义的格式输出，包含【章节大纲】标记，并以【大纲完】结束。"""
            else:
                 prompt_previous_outline = previous_outline_text if previous_outline_text else "无"
                 outline_creator_prompt = f"""请根据以下信息，修改或完善章节大纲。
【用户初始想法】
{initial_prompt}
【本轮最新的故事规划】
{current_story_plan}
【本轮最新的世界设定】
{current_world_details}
【本轮最新的角色档案】
{current_character_profiles}
【上一轮编辑反馈】
{last_editor_feedback}
【你上一轮生成的章节大纲】(请在此基础上修改)
{prompt_previous_outline}
请重点解决编辑反馈中与章节细节或大纲结构相关的问题，结合所有最新信息，输出修改后的**完整** {num_chapters} 章大纲，严格遵循格式，并以【大纲完】结束。"""

            try:
                user_proxy.reset()
                user_proxy.initiate_chat(recipient=outline_creator, message=outline_creator_prompt, max_turns=1, clear_history=True)
                outline_msg = user_proxy.last_message(outline_creator)
                if outline_msg and outline_msg["content"]:
                     draft_outline = extract_section(outline_msg["content"], "【章节大纲】", "【大纲完】")
                     if draft_outline:
                         current_outline_text = draft_outline
                         save_content(f"iteration_{i+1}_outline_draft.txt", current_outline_text)
                         print("章节大纲已更新。")
                         iteration_success_flags["outline_creator"] = True
                     else:
                         print("错误：未能从 Outline Creator 回复中提取章节大纲。")
                else:
                     print("错误：未能从 Outline Creator 获取有效回复。")
            except Exception as e:
                print(f"生成章节大纲时出错: {e}")

        # --- **新增：代码层面的完整性预检查** ---
        print("步骤 4.5: 进行大纲草稿完整性预检查...")
        parsed_chapters = _parse_outline(current_outline_text, num_chapters) # 先尝试解析
        is_complete = False
        if parsed_chapters:
            parsed_count = len(parsed_chapters)
            # 检查数量是否足够，并且序号是否大致连续 (允许解析出稍多或稍少几章，但不能差太多)
            # 一个更严格的检查是序号必须完全连续：
            is_sequential = all(parsed_chapters[j]["chapter_number"] == j + 1 for j in range(parsed_count))
            # 考虑到解析可能不完美，先只检查数量是否接近
            if abs(parsed_count - num_chapters) <= 1: # 允许+/-1章的误差？或者严格等于
            # if parsed_count == num_chapters and is_sequential: # 最严格的检查
                 print(f"预检查通过：解析出 {parsed_count} 章，基本符合要求。")
                 is_complete = True
            else:
                 print(f"错误：预检查失败！解析出 {parsed_count} 章 (期望 {num_chapters} 章) 或序号不连续。要求 Outline Creator 重做。")
        else:
            print("错误：预检查失败！未能从大纲草稿中解析出任何章节。要求 Outline Creator 重做。")

        if not is_complete:
             last_editor_feedback = "【系统强制反馈】大纲章节数量或序号严重不符合要求（期望 {} 章，实际解析到 {} 章或无法解析），请务必生成包含从第 1 章到第 {} 章所有内容的完整、连续的大纲，禁止省略！".format(num_chapters, parsed_count if parsed_chapters else 0, num_chapters)
             print("将基于预检查失败的反馈进入下一轮迭代...")
             # 更新 previous 变量以便下一轮参考（即使本轮内容不完整）
             previous_story_plan = current_story_plan
             previous_world_details = current_world_details
             previous_character_profiles = current_character_profiles
             previous_outline_text = current_outline_text # 保存有问题的版本供参考
             time.sleep(5) # 短暂暂停
             continue # 直接跳到下一轮迭代，让 Planner 等 Agent 收到这个强制反馈


        # --- 5. 大纲编辑评审 ---
        print("步骤 5: 进行大纲编辑评审...")
        # 确保所有需要评审的内容都存在
        if not all([current_story_plan, current_world_details, current_character_profiles, current_outline_text]):
             print("错误：缺少评审所需的完整信息，无法进行评审。跳过本轮评审。")
             last_editor_feedback = "错误：缺少评审材料。"
             # 更新 previous 变量，以便下一轮能继续 (如果适用)
             previous_story_plan = current_story_plan
             previous_world_details = current_world_details
             previous_character_profiles = current_character_profiles
             previous_outline_text = current_outline_text
             continue # 跳过本轮编辑
        else:
            editor_input_prompt = f"""请审阅以下完整材料：
【用户初始想法】
{initial_prompt}

【本轮最新故事规划】
{current_story_plan}

【本轮最新世界设定】
{current_world_details}

【本轮最新角色档案】
{current_character_profiles}  # <--- *** 在这里加上缺失的角色档案 ***

【本轮最新大纲草稿】(共 {num_chapters} 章)
{current_outline_text}
【草稿结束】

{"【上一轮编辑反馈】\n" + last_editor_feedback + "\n" if i > 0 and last_editor_feedback != "无" else ""}
{"【上一轮大纲草稿】(供对比参考)\n" + previous_outline_text + "\n" if i > 0 and previous_outline_text else ""}

请根据你的系统提示（包含评审流程和输出格式要求），进行全面评审，并输出 【最终批准】 或 【评审意见】+【修改建议】 或 【驳回重做】。"""
            try:
                user_proxy.reset()
                user_proxy.initiate_chat(recipient=outline_editor, message=editor_input_prompt, max_turns=1, clear_history=True)
                editor_response_msg = user_proxy.last_message(outline_editor)

                if not editor_response_msg or not editor_response_msg["content"]:
                    print(f"错误：未能从 Outline Editor 获取有效回复 (第 {i+1} 轮)。")
                    last_editor_feedback = "错误：编辑无响应。"
                    iteration_success_flags["editor"] = False
                else:
                    editor_response_text = editor_response_msg["content"]
                    save_content(f"iteration_{i+1}_editor_feedback.txt", editor_response_text)
                    print(f"大纲编辑回复 (第 {i+1} 轮):\n{editor_response_text[:500]}...")
                    iteration_success_flags["editor"] = True # 收到回复就算成功

                    if "【最终批准】" in editor_response_text and "优先级-高" not in editor_response_text:
                        print("大纲已获编辑最终批准！流程结束。")
                        parsed_outline = _parse_outline(current_outline_text, num_chapters)
                        if parsed_outline:
                            print("最终大纲解析成功。")
                            final_approved_outline_structure = parsed_outline
                            save_content("final_story_plan.txt", current_story_plan)
                            save_content("final_world_details.txt", current_world_details)
                            save_content("final_character_profiles.txt", current_character_profiles)
                            save_content("final_outline.txt", current_outline_text)
                            return final_approved_outline_structure, current_character_profiles, current_world_details, current_story_plan
                        else:
                             print("错误：最终大纲批准了，但解析失败。请检查 final_outline.txt")
                             return None, current_character_profiles, current_world_details, current_story_plan

                    elif "【驳回重做】" in editor_response_text:
                         print("大纲被编辑驳回重做。将在下一轮尝试。")
                         last_editor_feedback = editor_response_text
                    elif "【评审意见】" in editor_response_text or "【场景规划反馈】" in editor_response_text or "优先级-高" in editor_response_text: # 明确有反馈或高优先级建议就算需要修改
                        print(f"大纲编辑提出了修改建议 (第 {i+1} 轮)，将在下一轮迭代中应用。")
                        last_editor_feedback = editor_response_text
                    else:
                        print(f"警告：大纲编辑的回复格式无法识别 (第 {i+1} 轮)，将视为需要修改。")
                        last_editor_feedback = editor_response_text

            except Exception as e:
                print(f"大纲编辑评审过程中出错 (第 {i+1} 轮): {e}")
                last_editor_feedback = f"错误：评审过程中发生异常: {e}"
                iteration_success_flags["editor"] = False

        # --- 迭代结束前，更新 "previous" 变量 ---
        # 使用本轮最终确认的 current 值来更新下一轮的 previous
        previous_story_plan = current_story_plan
        previous_world_details = current_world_details
        previous_character_profiles = current_character_profiles
        previous_outline_text = current_outline_text

        # 如果本轮任何关键步骤失败，也许可以提前退出或记录
        if not all(iteration_success_flags.values()):
             print(f"警告：第 {i+1} 轮部分步骤未能成功完成。")

        # 每次迭代后稍作暂停
        print("暂停10秒...")
        time.sleep(10)

    # --- 循环结束 ---
    print(f"警告：达到最大迭代次数 ({MAX_FULL_REFINEMENT_ITERATIONS})，大纲仍未获得最终批准。")
    print("将使用最后一轮生成的结果。建议检查最后一次的编辑反馈。")

    # 保存最后一轮的成果
    if current_story_plan: save_content("final_unapproved_story_plan.txt", current_story_plan)
    if current_world_details: save_content("final_unapproved_world_details.txt", current_world_details)
    if current_character_profiles: save_content("final_unapproved_character_profiles.txt", current_character_profiles)
    if current_outline_text: save_content("final_unapproved_outline.txt", current_outline_text)

    # 尝试解析最后一轮的大纲
    parsed_outline = _parse_outline(current_outline_text, num_chapters) if current_outline_text else None
    if parsed_outline:
        print("最后一轮大纲解析成功。")
        return parsed_outline, current_character_profiles, current_world_details, current_story_plan
    else:
        print("警告：最后一轮大纲未能成功生成或解析。")
        return None, current_character_profiles, current_world_details, current_story_plan