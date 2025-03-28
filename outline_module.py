# outline_module.py (需要重构 generate_refined_outline)

import autogen
from typing import Optional, List, Dict, Tuple
from agents import (
    get_story_planner_agent, get_world_builder_agent,
    get_character_creator_agent, get_outline_creator_agent,
    get_outline_editor_agent, get_user_proxy_agent
)
from config import llm_config
from utils import save_content, extract_section, OUTPUT_DIR
import os
import time
import re

MAX_FULL_REFINEMENT_ITERATIONS = 3 # 完整流程的最大迭代次数

def generate_refined_outline(
    initial_prompt: str,
    num_chapters: int,
) -> Tuple[Optional[List[Dict]], Optional[str], Optional[str], Optional[str]]:
    """
    生成并精炼故事大纲、角色、世界和规划。
    此函数现在包含了完整的规划、设定、角色、大纲创建和编辑的迭代循环。
    返回: (最终大纲结构, 最终角色档案文本, 最终世界设定文本, 最终故事规划文本) 或 (None, None, None, None)
    """
    print("\n--- 开始完整的大纲精炼流程 ---")
    user_proxy = get_user_proxy_agent()
    planner = get_story_planner_agent()
    world_builder = get_world_builder_agent()
    char_creator = get_character_creator_agent()
    outline_creator = get_outline_creator_agent(num_chapters)
    outline_editor = get_outline_editor_agent()

    # 初始化变量，存储每次迭代的最新结果
    current_story_plan = None
    current_world_details = None
    current_character_profiles = None
    current_outline_text = None
    last_editor_feedback = "无" # 用于传递给下一轮

    final_approved_outline_structure = None

    for i in range(MAX_FULL_REFINEMENT_ITERATIONS):
        print(f"\n--- 第 {i+1}/{MAX_FULL_REFINEMENT_ITERATIONS} 轮精炼 ---")
        iteration_success = True # 标记本轮是否成功

        # --- 1. 故事规划 ---
        print("步骤 1: 生成/修改故事规划...")
        planner_prompt = f"""请根据以下初始想法进行故事规划。

【用户初始想法】
{initial_prompt}

【上一轮编辑反馈】(如果适用)
{last_editor_feedback}

请严格按照你的系统提示格式输出。如果收到反馈，请重点解决反馈中与故事规划相关的问题。"""
        try:
            user_proxy.reset()
            user_proxy.initiate_chat(recipient=planner, message=planner_prompt, max_turns=1, clear_history=True) # 使用 clear_history 避免混淆
            plan_msg = user_proxy.last_message(planner)
            if plan_msg and plan_msg["content"]:
                current_story_plan = plan_msg["content"]
                save_content(f"iteration_{i+1}_story_plan.txt", current_story_plan)
                print("故事规划已更新。")
            else:
                print("错误：未能从 Story Planner 获取有效回复。")
                iteration_success = False
        except Exception as e:
            print(f"生成故事规划时出错: {e}")
            iteration_success = False

        if not iteration_success: continue # 如果本轮失败，直接进入下一轮或结束

        # --- 2. 世界设定 ---
        print("步骤 2: 生成/修改世界设定...")
        world_builder_prompt = f"""请根据以下信息构建世界设定：

【用户初始想法】
{initial_prompt}

【最新的故事规划】
{current_story_plan}

【上一轮编辑反馈】(如果适用)
{last_editor_feedback}

请严格按照你的系统提示格式输出。如果收到反馈，请重点解决反馈中与世界设定相关的问题。"""
        try:
            user_proxy.reset()
            user_proxy.initiate_chat(recipient=world_builder, message=world_builder_prompt, max_turns=1, clear_history=True)
            world_msg = user_proxy.last_message(world_builder)
            if world_msg and world_msg["content"]:
                current_world_details = world_msg["content"]
                save_content(f"iteration_{i+1}_world_details.txt", current_world_details)
                print("世界设定已更新。")
            else:
                print("错误：未能从 World Builder 获取有效回复。")
                iteration_success = False
        except Exception as e:
            print(f"生成世界设定时出错: {e}")
            iteration_success = False

        if not iteration_success: continue

        # --- 3. 角色创建 ---
        print("步骤 3: 生成/修改角色档案...")
        char_creator_prompt = f"""请根据以下信息创建角色档案：

【用户初始想法】
{initial_prompt}

【最新的故事规划】
{current_story_plan}

【最新的世界设定】
{current_world_details}

【上一轮编辑反馈】(如果适用)
{last_editor_feedback}

请严格按照你的系统提示格式输出，并确保包含【角色档案】标记。如果收到反馈，请重点解决反馈中与角色相关的问题。"""
        try:
            user_proxy.reset()
            user_proxy.initiate_chat(recipient=char_creator, message=char_creator_prompt, max_turns=1, clear_history=True)
            profile_msg = user_proxy.last_message(char_creator)
            if profile_msg and profile_msg["content"]:
                profiles_text = profile_msg["content"]
                extracted_profiles = extract_section(profiles_text, "【角色档案】")
                if extracted_profiles:
                     current_character_profiles = extracted_profiles
                     save_content(f"iteration_{i+1}_character_profiles.txt", current_character_profiles)
                     print("角色档案已更新。")
                else:
                     print("警告：Character Creator 回复中未找到【角色档案】标记，将使用完整回复。")
                     current_character_profiles = profiles_text # 使用完整回复作为档案
                     save_content(f"iteration_{i+1}_character_profiles.txt", current_character_profiles)
            else:
                print("错误：未能从 Character Creator 获取有效回复。")
                iteration_success = False
        except Exception as e:
            print(f"生成角色档案时出错: {e}")
            iteration_success = False

        if not iteration_success: continue

        # --- 4. 大纲创作 ---
        print("步骤 4: 生成/修改章节大纲...")
        outline_creator_prompt = f"""请根据以下所有最新信息，创作一个包含 {num_chapters} 章的详细大纲：

【用户初始想法】
{initial_prompt}

【最新的故事规划】
{current_story_plan}

【最新的世界设定】
{current_world_details}

【最新的角色档案】
{current_character_profiles}

【上一轮编辑反馈】(如果适用)
{last_editor_feedback}

请严格按照你的系统提示中定义的格式输出，包含【章节大纲】标记，并以【大纲完】结束。如果收到反馈，请重点解决反馈中与章节细节或大纲结构相关的问题。"""
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
                else:
                    print("错误：未能从 Outline Creator 回复中提取章节大纲。")
                    print("原始回复:", outline_msg["content"][:1000])
                    iteration_success = False
            else:
                print("错误：未能从 Outline Creator 获取有效回复。")
                iteration_success = False
        except Exception as e:
            print(f"生成章节大纲时出错: {e}")
            iteration_success = False

        if not iteration_success: continue

        # --- 5. 大纲编辑评审 ---
        print("步骤 5: 进行大纲编辑评审...")
        editor_input_prompt = f"""请审阅以下完整材料：

【用户初始想法】
{initial_prompt}

【最新故事规划】
{current_story_plan}

【最新世界设定】
{current_world_details}

【最新角色档案】
{current_character_profiles}

【最新大纲草稿】(共 {num_chapters} 章)
{current_outline_text}
【草稿结束】

请根据你的系统提示，进行全面评审，并输出 【最终批准】 或 【评审意见】+【修改建议】 或 【驳回重做】。"""
        try:
            user_proxy.reset()
            user_proxy.initiate_chat(recipient=outline_editor, message=editor_input_prompt, max_turns=1, clear_history=True)
            editor_response_msg = user_proxy.last_message(outline_editor)

            if not editor_response_msg or not editor_response_msg["content"]:
                print(f"错误：未能从 Outline Editor 获取有效回复 (第 {i+1} 轮)。")
                last_editor_feedback = "错误：编辑无响应。"
                continue # 进行下一轮迭代，保留当前结果

            editor_response_text = editor_response_msg["content"]
            save_content(f"iteration_{i+1}_editor_feedback.txt", editor_response_text)
            print(f"大纲编辑回复 (第 {i+1} 轮):\n{editor_response_text[:500]}...")

            if "【最终批准】" in editor_response_text:
                print("大纲已获编辑最终批准！流程结束。")
                # 解析最终文本
                parsed_outline = _parse_outline(current_outline_text, num_chapters)
                if parsed_outline:
                    print("最终大纲解析成功。")
                    final_approved_outline_structure = parsed_outline
                     # 保存最终版本的文件
                    save_content("final_story_plan.txt", current_story_plan)
                    save_content("final_world_details.txt", current_world_details)
                    save_content("final_character_profiles.txt", current_character_profiles)
                    save_content("final_outline.txt", current_outline_text)
                    return final_approved_outline_structure, current_character_profiles, current_world_details, current_story_plan
                else:
                     print("错误：最终大纲批准了，但解析失败。请检查 final_outline.txt")
                     # 即使解析失败，也返回文本供后续步骤尝试加载
                     return None, current_character_profiles, current_world_details, current_story_plan

            elif "【驳回重做】" in editor_response_text:
                 print("大纲被编辑驳回，建议重做。请检查反馈并调整初始输入或流程。")
                 last_editor_feedback = editor_response_text # 记录反馈
                 # 可以选择在这里终止，或者让循环继续（可能意义不大）
                 # return None, None, None, None
                 continue # 继续下一轮，让 Agent 基于驳回意见尝试

            elif "【评审意见】" in editor_response_text and "【修改建议】" in editor_response_text:
                print(f"大纲编辑提出了修改建议 (第 {i+1} 轮)，将在下一轮迭代中应用。")
                last_editor_feedback = editor_response_text # 记录反馈给下一轮
            else:
                print(f"警告：大纲编辑的回复格式无法识别 (第 {i+1} 轮)，将视为需要修改。")
                last_editor_feedback = editor_response_text # 记录反馈给下一轮

        except Exception as e:
            print(f"大纲编辑评审过程中出错 (第 {i+1} 轮): {e}")
            last_editor_feedback = f"错误：评审过程中发生异常: {e}"
            # 继续下一轮迭代

        # 每次迭代后稍作暂停
        print("暂停10秒...")
        time.sleep(10)

    # 如果循环结束仍未批准
    print(f"警告：达到最大迭代次数 ({MAX_FULL_REFINEMENT_ITERATIONS})，大纲仍未获得最终批准。")
    print("将使用最后一轮生成的结果。建议检查最后一次的编辑反馈。")
    # 解析最后一轮的大纲
    parsed_outline = _parse_outline(current_outline_text, num_chapters) if current_outline_text else None
    if parsed_outline:
        print("最后一轮大纲解析成功。")
        # 保存最后版本的文件
        save_content("final_unapproved_story_plan.txt", current_story_plan or "")
        save_content("final_unapproved_world_details.txt", current_world_details or "")
        save_content("final_unapproved_character_profiles.txt", current_character_profiles or "")
        save_content("final_unapproved_outline.txt", current_outline_text or "")
        return parsed_outline, current_character_profiles, current_world_details, current_story_plan
    else:
        print("错误：最后一轮大纲未能成功生成或解析。")
        return None, current_character_profiles, current_world_details, current_story_plan


# --- 需要在 outline_module.py 中保留或移动过来的函数 ---
def _parse_outline(outline_text: Optional[str], num_chapters: int) -> Optional[List[Dict]]:
    """解析大纲文本为结构化数据 (之前的实现)"""
    if not outline_text:
        return None

    chapters = []
    # 使用正则表达式匹配每个章节块
    chapter_pattern = re.compile(
        r"^\s*\*\*(第\s*(\d+)\s*章:\s*(.*?))\*\*\s*$"
        r"(.*?)"
        r"(?=(^\s*\*\*(第\s*\d+\s*章:|\【大纲完\】))|\Z)",
        re.MULTILINE | re.DOTALL
    )
    matches = chapter_pattern.finditer(outline_text)
    processed_indices = set()

    for match in matches:
        chapter_title_full, chapter_num_str, chapter_title = match.group(1), match.group(2), match.group(3)
        chapter_content = match.group(4).strip()
        try:
            chapter_num = int(chapter_num_str)
        except ValueError:
            print(f"警告：无法解析章节号 '{chapter_num_str}'")
            continue

        if chapter_num in processed_indices: continue

        chapter_data = {"chapter_number": chapter_num, "title": chapter_title.strip()}
        target = extract_section(chapter_content, "本章目标:")
        events_text = extract_section(chapter_content, "关键事件:")
        dev = extract_section(chapter_content, "角色发展:")
        setting = extract_section(chapter_content, "场景设定:")
        tone = extract_section(chapter_content, "基调:")
        foreshadow = extract_section(chapter_content, "引出/伏笔:")

        events = []
        if events_text:
            events = [e.strip() for e in events_text.split('\n') if e.strip().startswith('-')]

        prompt_lines = []
        if target: prompt_lines.append(f"- 本章目标: {target}")
        if events: prompt_lines.append(f"- 关键事件:\n    {chr(10).join(events)}")
        if dev: prompt_lines.append(f"- 角色发展:\n{dev}")
        if setting: prompt_lines.append(f"- 场景设定: {setting}")
        if tone: prompt_lines.append(f"- 基调: {tone}")
        if foreshadow: prompt_lines.append(f"- 引出/伏笔: {foreshadow}")

        chapter_data["prompt"] = "\n".join(prompt_lines)

        # 移除基本验证的打印，避免过多输出
        # if len(events) < 2: print(f"警告：第 {chapter_num} 章关键事件少于2个。")
        # if not all([target, dev, setting, tone]): print(f"警告：第 {chapter_num} 章缺少必要元素。")

        chapters.append(chapter_data)
        processed_indices.add(chapter_num)

    if len(chapters) != num_chapters:
        print(f"警告：解析出 {len(chapters)} 章，与要求的 {num_chapters} 章不符。")
        chapters.sort(key=lambda x: x["chapter_number"])
        # 可以添加填充或截断逻辑

    return chapters if chapters else None

# 注意：generate_initial_story_elements 和 generate_character_profiles
# 这两个函数现在的功能被包含在 generate_refined_outline 内部的循环中了，
# 可以从 outline_module.py 中移除，或者保留作为独立的辅助函数（但不被主流程直接调用）。
# 这里暂时注释掉或删除它们，避免混淆。
# def generate_initial_story_elements(...) -> ...: ...
# def generate_character_profiles(...) -> ...: ...