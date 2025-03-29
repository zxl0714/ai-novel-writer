# outline_module.py
# Implements the iterative outline refinement process, including
# batch generation for the detailed chapter outline,
# and passing previous batch context during revisions.

import autogen
from typing import Optional, List, Dict, Tuple
# Import agent retrieval functions and configurations
from agents import (
    get_story_planner_agent, get_world_builder_agent,
    get_character_creator_agent, get_outline_creator_agent,
    get_outline_editor_agent, get_user_proxy_agent
)
# Import default and reasoning LLM configurations
from config import llm_config, reasoning_llm_config_to_use, DEFAULT_AGENT_CONFIG
# Import utility functions
from utils import save_content, extract_section, OUTPUT_DIR, load_content
import os
import time
import re # For parsing

# --- Constants ---
MAX_FULL_REFINEMENT_ITERATIONS = 7 # Max refinement rounds for the whole outline process
OUTLINE_BATCH_SIZE = 5 # Number of chapters Outline Creator generates per call
MAX_BATCH_RETRIES = 1 # Max retries if a batch generation fails validation

# --- Helper Function for Parsing Outline ---
# (Assuming the _parse_outline function from previous responses is here and works)
def _parse_outline(outline_text: Optional[str], expected_chapters: Optional[int] = None, chapter_range: Optional[Tuple[int, int]] = None) -> Tuple[Optional[List[Dict]], str]:
    """
    Parses outline text into structured data (List[Dict]).
    Also returns a status string ("OK", "Parse Error", "Count Mismatch", "Sequence Error").
    Can parse the full outline or a specific range of chapters.
    """
    status = "Parse Error"
    if not outline_text:
        # print("错误：传入的 outline_text 为空，无法解析。") # Reduce noise
        return None, status

    chapters = []
    # Regex to capture chapter number, title, and content block
    chapter_pattern = re.compile(
        r"^\s*\*\*\s*第\s*(\d+)\s*章\s*[:：]\s*(.*?)\s*\*\*\s*$" # Header line
        r"([\s\S]*?)" # Content (non-greedy)
        r"(?=(?:^\s*\*\*\s*第\s*\d+\s*章\s*[:：])|(?:^\s*【大纲完】\s*$)|\Z)", # Lookahead
        re.MULTILINE
    )

    clean_outline_text = outline_text.strip()
    end_marker = "【大纲完】"
    if clean_outline_text.endswith(end_marker) and not clean_outline_text.endswith(f"\n{end_marker}"):
         clean_outline_text = clean_outline_text[:-len(end_marker)].strip()

    matches = chapter_pattern.finditer(clean_outline_text)
    processed_indices = set()
    last_match_end = 0
    parsed_numbers = []

    keys_in_order = ["本章目标", "关键事件", "角色发展", "场景设定", "基调", "引出/伏笔"]
    def safe_extract(content, start_key, keys_order):
        # ... (safe_extract implementation from previous response) ...
        start_tag = f"{start_key}:"
        start_idx = content.find(start_tag)
        if start_idx == -1: # Try without colon
            start_tag_no_colon = f"{start_key}"
            start_idx = content.find(start_tag_no_colon)
            if start_idx == -1: return None
            start_tag = start_tag_no_colon
        
        content_start = start_idx + len(start_tag)
        end_idx = len(content)
        current_key_index_in_order = keys_order.index(start_key) if start_key in keys_order else -1
        
        if current_key_index_in_order != -1:
            for next_key_index in range(current_key_index_in_order + 1, len(keys_order)):
                next_key = keys_order[next_key_index]
                next_key_tag = f"{next_key}:"
                next_key_tag_no_colon = f"{next_key}"
                
                next_indices = []
                idx1 = content.find(next_key_tag, content_start)
                if idx1 != -1: next_indices.append(idx1)
                idx2 = content.find(next_key_tag_no_colon, content_start)
                if idx2 != -1 and idx2 not in next_indices: next_indices.append(idx2)
                
                if not next_indices: continue
                
                best_next_idx = min(next_indices)
                newline_before = content.rfind('\n', content_start, best_next_idx)
                
                is_at_line_start = False
                if newline_before != -1:
                    line_fragment = content[newline_before:best_next_idx].strip()
                    if line_fragment.startswith(next_key):
                        is_at_line_start = True
                        end_idx = newline_before
                        break
                elif content[:best_next_idx].strip() == "":
                     is_at_line_start = True
                     end_idx = best_next_idx
                     break
        
        return content[content_start:end_idx].strip()


    for match in matches:
        last_match_end = match.end()
        chapter_num_str = match.group(1)
        chapter_title = match.group(2).strip()
        chapter_content = match.group(3).strip()

        try:
            chapter_num = int(chapter_num_str)
        except ValueError:
            # print(f"警告：无法解析章节号 '{chapter_num_str}'，跳过。") # Reduce noise
            continue

        if chapter_range and not (chapter_range[0] <= chapter_num <= chapter_range[1]):
            continue

        if chapter_num in processed_indices:
            # print(f"警告：重复解析到第 {chapter_num} 章，跳过。") # Reduce noise
            status = "Sequence Error"
            continue

        if not chapter_title or not chapter_content:
             # print(f"警告：第 {chapter_num} 章缺少标题或内容，跳过。") # Reduce noise
             continue

        chapter_data = {"chapter_number": chapter_num, "title": chapter_title}
        parsed_numbers.append(chapter_num)

        target = safe_extract(chapter_content, "本章目标", keys_in_order)
        events_text = safe_extract(chapter_content, "关键事件", keys_in_order)
        dev = safe_extract(chapter_content, "角色发展", keys_in_order)
        setting = safe_extract(chapter_content, "场景设定", keys_in_order)
        tone = safe_extract(chapter_content, "基调", keys_in_order)
        foreshadow = safe_extract(chapter_content, "引出/伏笔", keys_in_order)

        events = []
        if events_text:
            lines = events_text.split('\n')
            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith('-') or re.match(r'^\s*\d+\.', stripped_line):
                    event_content = re.sub(r'^[-\s*\d\.]+\s*', '', stripped_line)
                    if event_content: events.append(event_content)

        prompt_lines = []
        if target: prompt_lines.append(f"- 本章目标: {target}")
        if events:
            formatted_events = "\n".join([f"    - {e}" for e in events])
            prompt_lines.append(f"- 关键事件:\n{formatted_events}")
        if dev: prompt_lines.append(f"- 角色发展:\n{dev}")
        if setting: prompt_lines.append(f"- 场景设定: {setting}")
        if tone: prompt_lines.append(f"- 基调: {tone}")
        if foreshadow: prompt_lines.append(f"- 引出/伏笔: {foreshadow}")
        # Store the original text block as well, might be useful for revisions
        chapter_data["original_text"] = match.group(0) # Full matched block including header
        chapter_data["prompt"] = "\n".join(prompt_lines)

        chapters.append(chapter_data)
        processed_indices.add(chapter_num)

    # --- Final Validation ---
    if not chapters:
        # print("错误：未能从文本中解析出任何有效的章节信息。") # Reduce noise
        return None, "Parse Error"

    chapters.sort(key=lambda x: x["chapter_number"])
    parsed_numbers.sort()

    expected_start = chapter_range[0] if chapter_range else 1
    is_sequential = all(parsed_numbers[j] == expected_start + j for j in range(len(parsed_numbers)))
    if not is_sequential:
        print(f"警告：解析出的章节号不连续。期望起始 {expected_start}，实际解析到: {parsed_numbers}")
        status = "Sequence Error"
    else:
         status = "OK"

    parsed_count = len(chapters)
    if expected_chapters is not None:
        if parsed_count != expected_chapters:
            print(f"警告：解析出 {parsed_count} 章，与期望的 {expected_chapters} 章不符。")
            status = "Count Mismatch" if status == "OK" else status

    # print(f"解析状态: {status}, 共解析出 {parsed_count} 章: {parsed_numbers}") # Reduce noise
    return chapters, status


# --- Main Outline Refinement Function ---
def generate_refined_outline(
    initial_prompt: str,
    num_chapters: int,
) -> Tuple[Optional[List[Dict]], Optional[str], Optional[str], Optional[str]]:
    """
    Generates and refines the story outline, characters, world, and plan using
    an iterative loop, including batch generation for the chapter outline and
    passing previous context correctly for revisions.
    """
    print("\n--- 开始完整的大纲精炼流程 ---")
    if num_chapters <= 0:
         print("错误：请求的章节数必须大于 0。")
         return None, None, None, None

    # --- Agent Initialization ---
    print("Initializing agents for Outline Refinement...")
    try:
        user_proxy = get_user_proxy_agent()
        planner = get_story_planner_agent(num_chapters=num_chapters)
        world_builder = get_world_builder_agent()
        char_creator = get_character_creator_agent()
        outline_editor = get_outline_editor_agent(num_chapters=num_chapters)
        print("Agents initialized.")
    except Exception as e:
        print(f"错误：初始化 Outline Refinement Agents 时出错: {e}")
        return None, None, None, None

    # --- State Variables ---
    current_story_plan: Optional[str] = None
    current_world_details: Optional[str] = None
    current_character_profiles: Optional[str] = None
    current_outline_text: Optional[str] = None # Complete outline text for the current iteration
    last_editor_feedback: str = "无"

    previous_story_plan: Optional[str] = None
    previous_world_details: Optional[str] = None
    previous_character_profiles: Optional[str] = None
    previous_outline_text: Optional[str] = None # Full outline text from the PREVIOUS iteration
    # ** NEW: Store parsed previous outline for easy batch access **
    parsed_previous_outline: Optional[List[Dict]] = None

    final_outline_structure: Optional[List[Dict]] = None

    # --- Main Refinement Loop ---
    for i in range(MAX_FULL_REFINEMENT_ITERATIONS):
        print(f"\n--- 第 {i+1}/{MAX_FULL_REFINEMENT_ITERATIONS} 轮精炼 ---")
        iteration_success_flags = {"planner": False, "builder": False, "char_creator": False, "outline_creator": False, "editor": False}

        # --- 1. Story Planning (Includes check/update for previous state) ---
        print("步骤 1: 生成/修改故事规划...")
        planner_prompt: str
        if i == 0:
             planner_prompt = f"""... (Prompt for first iteration as before) ..."""
             planner_prompt = f"""请根据以下初始想法进行故事规划 (目标 {num_chapters} 章)。
【用户初始想法】
{initial_prompt}
请严格按照你的系统提示格式输出。"""
        else:
             prompt_previous_plan = previous_story_plan if previous_story_plan else "无"
             planner_prompt = f"""请根据以下信息，修改或完善故事规划 (目标 {num_chapters} 章)。
【用户初始想法】
{initial_prompt}
【上一轮编辑反馈】
{last_editor_feedback}
【你上一轮生成的故事规划】(请在此基础上修改)
{prompt_previous_plan}
请重点解决编辑反馈中与故事规划相关的问题，并输出修改后的**完整**故事规划，严格遵循格式。"""
        try:
            # ... (Call planner, update current_story_plan, handle fallback) ...
            user_proxy.reset()
            user_proxy.initiate_chat(recipient=planner, message=planner_prompt, max_turns=1, clear_history=True)
            plan_msg = user_proxy.last_message(planner)
            if plan_msg and plan_msg["content"]:
                current_story_plan = plan_msg["content"]
                iteration_success_flags["planner"] = True
                print("故事规划已更新。")
            else:
                print("错误：未能从 Story Planner 获取有效回复。")
                if previous_story_plan is not None: current_story_plan = previous_story_plan
        except Exception as e:
            print(f"生成故事规划时出错: {e}")
            if previous_story_plan is not None: current_story_plan = previous_story_plan

        if current_story_plan is None:
             print("错误：缺少故事规划，无法继续本轮精炼。")
             # Update previous state before continuing/breaking
             previous_story_plan = current_story_plan
             previous_world_details = current_world_details
             previous_character_profiles = current_character_profiles
             previous_outline_text = current_outline_text
             if parsed_previous_outline is None and previous_outline_text: # Try parsing if we have text
                  parsed_previous_outline, _ = _parse_outline(previous_outline_text, num_chapters)
             continue

        # --- 2. World Building (Includes check/update for previous state) ---
        print("步骤 2: 生成/修改世界设定...")
        world_builder_prompt: str
        if i == 0:
            world_builder_prompt = f"""... (Prompt for first iteration as before) ..."""
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
            # ... (Call builder, update current_world_details, handle fallback) ...
            user_proxy.reset()
            user_proxy.initiate_chat(recipient=world_builder, message=world_builder_prompt, max_turns=1, clear_history=True)
            world_msg = user_proxy.last_message(world_builder)
            if world_msg and world_msg["content"]:
                current_world_details = world_msg["content"]
                iteration_success_flags["builder"] = True
                print("世界设定已更新。")
            else:
                print("错误：未能从 World Builder 获取有效回复。")
                if previous_world_details is not None: current_world_details = previous_world_details
        except Exception as e:
             print(f"生成世界设定时出错: {e}")
             if previous_world_details is not None: current_world_details = previous_world_details

        if current_world_details is None:
             print("错误：缺少世界设定，无法继续本轮精炼。")
             # Update previous state before continuing/breaking
             previous_story_plan = current_story_plan
             previous_world_details = current_world_details
             previous_character_profiles = current_character_profiles
             previous_outline_text = current_outline_text
             if parsed_previous_outline is None and previous_outline_text:
                  parsed_previous_outline, _ = _parse_outline(previous_outline_text, num_chapters)
             continue

        # --- 3. Character Creation (Includes check/update for previous state) ---
        print("步骤 3: 生成/修改角色档案...")
        char_creator_prompt: str
        if i == 0:
             char_creator_prompt = f"""... (Prompt for first iteration as before) ..."""
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
            # ... (Call char_creator, update current_character_profiles, handle fallback) ...
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
                     current_character_profiles = profiles_text
                 iteration_success_flags["char_creator"] = True
            else:
                 print("错误：未能从 Character Creator 获取有效回复。")
                 if previous_character_profiles is not None: current_character_profiles = previous_character_profiles
        except Exception as e:
            print(f"生成角色档案时出错: {e}")
            if previous_character_profiles is not None: current_character_profiles = previous_character_profiles

        if current_character_profiles is None:
            print("错误：缺少角色档案，无法继续本轮精炼。")
            # Update previous state before continuing/breaking
            previous_story_plan = current_story_plan
            previous_world_details = current_world_details
            previous_character_profiles = current_character_profiles
            previous_outline_text = current_outline_text
            if parsed_previous_outline is None and previous_outline_text:
                  parsed_previous_outline, _ = _parse_outline(previous_outline_text, num_chapters)
            continue

        # --- 4. Batch Outline Generation ---
        print(f"步骤 4: 分批生成 {num_chapters} 章大纲 (每批 {OUTLINE_BATCH_SIZE} 章)...")
        accumulated_outline_text_this_iteration = ""
        batch_outline_texts = []
        all_batches_successful = True

        # Prepare context of previous iteration's outline (parsed) for revision reference
        previous_outline_chapters_map: Dict[int, Dict] = {}
        if i > 0 and parsed_previous_outline: # Use the parsed version from end of last iteration
             for ch_data in parsed_previous_outline:
                  if isinstance(ch_data, dict) and "chapter_number" in ch_data:
                      previous_outline_chapters_map[ch_data["chapter_number"]] = ch_data


        for batch_num, batch_start in enumerate(range(1, num_chapters + 1, OUTLINE_BATCH_SIZE)):
            batch_end = min(batch_start + OUTLINE_BATCH_SIZE - 1, num_chapters)
            print(f"  生成批次 {batch_num + 1}: 章节 {batch_start} 到 {batch_end}")

            batch_successful = False
            for attempt in range(MAX_BATCH_RETRIES + 1):
                print(f"    尝试 {attempt + 1}/{MAX_BATCH_RETRIES + 1}...")

                # --- ** 修改点：准备传递给 Outline Creator 的 Prompt ** ---
                # 基础上下文
                base_context = f"""【用户初始想法】
{initial_prompt}
【本轮最新的故事规划】
{current_story_plan}
【本轮最新的世界设定】
{current_world_details}
【本轮最新的角色档案】
{current_character_profiles}
"""
                # 本轮前面批次的上下文
                previous_batches_context = f"【先前已生成的本轮大纲部分】(供衔接参考)\n{accumulated_outline_text_this_iteration}\n" if accumulated_outline_text_this_iteration else ""

                # 上一轮编辑反馈 (针对整篇)
                editor_feedback_context = f"【上一轮编辑反馈】(请重点关注与章节 {batch_start}-{batch_end} 相关的部分)\n{last_editor_feedback}\n" if i > 0 and last_editor_feedback != "无" else ""

                # ** 修改点：获取并格式化上一轮对应的章节大纲 **
                previous_batch_text = ""
                if i > 0 and previous_outline_chapters_map:
                     prev_batch_parts = []
                     for ch_num in range(batch_start, batch_end + 1):
                          if ch_num in previous_outline_chapters_map:
                               # Extract the original text block if stored, otherwise reformat
                               original_text = previous_outline_chapters_map[ch_num].get("original_text")
                               if original_text:
                                    prev_batch_parts.append(original_text)
                               else: # Reformat if original text wasn't stored
                                    ch_data = previous_outline_chapters_map[ch_num]
                                    prev_batch_parts.append(f"**第 {ch_num} 章: {ch_data.get('title','N/A')}**\n{ch_data.get('prompt','')}")
                          else:
                               prev_batch_parts.append(f"**第 {ch_num} 章:** (上一轮未找到或解析失败)")
                     if prev_batch_parts:
                           previous_batch_text = f"【你上一轮生成的对应章节大纲】(请在此基础上修改)\n" + "\n\n".join(prev_batch_parts) + "\n"


                # 构建最终 Prompt
                outline_creator_prompt = f"""请根据以下所有最新信息，**仅生成第 {batch_start} 章到第 {batch_end} 章** 的详细章节大纲。

{base_context}
{previous_batches_context}
{editor_feedback_context}
{previous_batch_text}

【写作指令】
你的当前任务是：**只输出第 {batch_start} 章到第 {batch_end} 章的大纲内容**。
请严格遵循你系统提示中定义的格式为这几章生成内容。
确保与【先前已生成的本轮大纲部分】(如果存在) 以及整体规划保持连贯。
{f'请重点根据【上一轮编辑反馈】和【你上一轮生成的对应章节大纲】，并结合本轮最新的规划/设定/角色信息，对第 {batch_start} 章到第 {batch_end} 章进行修改和完善。' if i > 0 else ''}
**禁止**输出范围之外的章节，**禁止**使用省略号。"""

                # 添加结束标记指令（如果这是最后一批）
                if batch_end == num_chapters:
                    outline_creator_prompt += "\n请在生成完最后一章后，明确使用 **【大纲完】** 作为结束标记。"
                else:
                     outline_creator_prompt += "\n请**不要**在本批次结尾添加【大纲完】标记。"

                # --- ** 调用 Outline Creator ** ---
                try:
                    outline_creator = get_outline_creator_agent(num_chapters=num_chapters, batch_start=batch_start, batch_end=batch_end)
                    user_proxy.reset()
                    user_proxy.initiate_chat(recipient=outline_creator, message=outline_creator_prompt, max_turns=1, clear_history=True)
                    batch_msg = user_proxy.last_message(outline_creator)

                    if batch_msg and batch_msg["content"]:
                        batch_text_raw = batch_msg["content"]
                        # Extract and clean batch text
                        batch_text = extract_section(batch_text_raw, "【章节大纲】") or batch_text_raw
                        if batch_end != num_chapters and batch_text.strip().endswith("【大纲完】"):
                             batch_text = batch_text.rsplit("【大纲完】", 1)[0].strip()
                        batch_text = batch_text.strip() # Final strip

                        if not batch_text: # Handle case where LLM only outputs marker
                             print(f"    错误：批次 {batch_num + 1} LLM 返回内容为空或仅含标记。")
                             raise ValueError("Empty content from Outline Creator")

                        # Validate this batch
                        print(f"    验证批次 {batch_num + 1}...")
                        expected_batch_len = batch_end - batch_start + 1
                        parsed_batch, batch_status = _parse_outline(batch_text,
                                                                    expected_chapters=expected_batch_len,
                                                                    chapter_range=(batch_start, batch_end))

                        # Stricter validation for batch generation
                        if parsed_batch and batch_status == "OK" and len(parsed_batch) == expected_batch_len:
                             print(f"    批次 {batch_num + 1} 验证通过。")
                             batch_outline_texts.append(batch_text) # Store raw text
                             batch_successful = True
                             break # Exit retry loop
                        else:
                             print(f"    错误：批次 {batch_num + 1} 验证失败 (状态: {batch_status}, 解析到 {len(parsed_batch) if parsed_batch else 0}/{expected_batch_len} 章)。")
                             # Retry if attempts remain
                    else:
                        print(f"    错误：未能从 Outline Creator 获取批次 {batch_num + 1} 的有效回复。")

                except Exception as e_batch:
                    print(f"    生成大纲批次 {batch_num + 1} 时出错: {e_batch}")

                if not batch_successful and attempt < MAX_BATCH_RETRIES:
                    print(f"    将在 5 秒后重试批次 {batch_num + 1}...")
                    time.sleep(5)

            # End of retry loop for batch
            if not batch_successful:
                print(f"错误：生成大纲批次 {batch_num + 1} (章节 {batch_start}-{batch_end}) 失败，已达到最大重试次数。")
                all_batches_successful = False
                break # Exit the main batch generation loop

            # Update accumulated text for next batch's context
            accumulated_outline_text_this_iteration = "\n\n".join(batch_outline_texts).strip()

        # --- End Batch Outline Generation Loop ---

        if not all_batches_successful:
            print(f"错误：第 {i+1} 轮大纲批次生成未能完成。跳过本轮后续步骤。")
            last_editor_feedback = "【系统强制反馈】大纲未能完整生成，请重试。"
            previous_story_plan = current_story_plan # Still update previous for next attempt
            previous_world_details = current_world_details
            previous_character_profiles = current_character_profiles
            previous_outline_text = None # Mark outline as failed this round
            parsed_previous_outline = None
            continue

        # All batches successful, assemble full text
        current_outline_text = accumulated_outline_text_this_iteration + "\n\n【大纲完】" # Add final marker
        iteration_success_flags["outline_creator"] = True
        print("所有大纲批次生成完成。")
        save_content(f"iteration_{i+1}_outline_draft_complete.txt", current_outline_text)

        # --- 4.5. Code-Level Completeness Check (Final check on assembled text) ---
        print("步骤 4.5: 进行最终大纲完整性检查...")
        final_parsed_outline, final_status = _parse_outline(current_outline_text, num_chapters)
        # Check if exactly num_chapters were parsed AND the sequence is OK
        if final_parsed_outline and len(final_parsed_outline) == num_chapters and final_status == "OK":
             print("最终大纲完整性检查通过。")
        else:
             parsed_count = len(final_parsed_outline) if final_parsed_outline else 0
             print(f"错误：最终大纲完整性检查失败！状态: {final_status}, 解析到 {parsed_count}/{num_chapters} 章。")
             last_editor_feedback = f"【系统强制反馈】最终组装的大纲不完整或格式/序号错误 (状态: {final_status}, 解析到 {parsed_count}/{num_chapters} 章)，请重新生成所有章节。"
             previous_story_plan = current_story_plan
             previous_world_details = current_world_details
             previous_character_profiles = current_character_profiles
             previous_outline_text = current_outline_text # Keep faulty version
             parsed_previous_outline = final_parsed_outline # Keep potentially partially parsed version
             continue

        # --- 5. Outline Editor Review ---
        print("步骤 5: 调用大纲编辑进行评审...")
        # Construct editor prompt (including previous outline text for comparison in i > 0)
        editor_input_prompt = f"""请审阅以下完整材料：
【用户初始想法】
{initial_prompt}
【本轮最新的故事规划】
{current_story_plan}
【本轮最新的世界设定】
{current_world_details}
【本轮最新的角色档案】
{current_character_profiles}
【本轮最新大纲草稿】(共 {num_chapters} 章)
{current_outline_text}
【草稿结束】 implicit end marker now 【大纲完】

{"【上一轮编辑反馈】\n" + last_editor_feedback + "\n" if i > 0 and last_editor_feedback != "无" else ""}
{"【上一轮大纲草稿】(供对比参考)\n" + previous_outline_text + "\n" if i > 0 and previous_outline_text else ""}

请根据你的系统提示（首先检查完整性，然后评估逻辑、角色、设定、节奏、黄金三章、结局、常见陷阱等），输出 【最终批准】 或 【驳回重做】 或 【评审意见】+【修改建议】。"""

        try:
            user_proxy.reset()
            user_proxy.initiate_chat(recipient=outline_editor, message=editor_input_prompt, max_turns=1, clear_history=True)
            editor_response_msg = user_proxy.last_message(outline_editor)

            if not editor_response_msg or not editor_response_msg["content"]:
                 # ... (Error handling for editor response) ...
                 print(f"错误：未能从 Outline Editor 获取有效回复 (第 {i+1} 轮)。")
                 last_editor_feedback = "错误：编辑无响应。"
                 iteration_success_flags["editor"] = False
            else:
                 editor_response_text = editor_response_msg["content"]
                 save_content(f"iteration_{i+1}_editor_feedback.txt", editor_response_text)
                 print(f"大纲编辑回复 (第 {i+1} 轮):\n{editor_response_text[:500]}...")
                 iteration_success_flags["editor"] = True

                 # Process Editor Decision (Stricter check)
                 # Check for explicit approval AND absence of high-priority feedback
                 is_approved = "【最终批准】" in editor_response_text and "优先级-高" not in editor_response_text

                 if is_approved:
                     print("大纲已获编辑最终批准！流程结束。")
                     # Parse the final approved outline one last time
                     final_parsed_outline_approved, final_status_approved = _parse_outline(current_outline_text, num_chapters)
                     if final_parsed_outline_approved and final_status_approved == "OK":
                         print("最终批准的大纲解析成功。")
                         final_outline_structure = final_parsed_outline_approved
                         # Save final versions
                         if current_story_plan: save_content("final_story_plan.txt", current_story_plan)
                         if current_world_details: save_content("final_world_details.txt", current_world_details)
                         if current_character_profiles: save_content("final_character_profiles.txt", current_character_profiles)
                         if current_outline_text: save_content("final_outline.txt", current_outline_text)
                         return final_outline_structure, current_character_profiles, current_world_details, current_story_plan
                     else:
                          print(f"错误：最终大纲批准了，但最终解析失败 (状态: {final_status_approved})。请检查 final_outline.txt")
                          return None, current_character_profiles, current_world_details, current_story_plan # Return text

                 elif "【驳回重做】" in editor_response_text:
                      print("大纲被编辑驳回重做。将在下一轮尝试。")
                      last_editor_feedback = editor_response_text
                 # Any other feedback (or approval with high-prio issues) means revision needed
                 elif "【评审意见】" in editor_response_text or "【场景规划反馈】" in editor_response_text or "优先级-高" in editor_response_text:
                     print(f"大纲编辑提出了修改建议 (第 {i+1} 轮)，将在下一轮迭代中应用。")
                     last_editor_feedback = editor_response_text
                 else: # Unknown format or implicit approval without explicit tag
                     print(f"警告：大纲编辑的回复格式无法识别或未明确批准/反馈 (第 {i+1} 轮)，将视为需要修改。")
                     last_editor_feedback = editor_response_text

        except Exception as e_edit:
             print(f"大纲编辑评审过程中出错 (第 {i+1} 轮): {e_edit}")
             last_editor_feedback = f"错误：评审过程中发生异常: {e_edit}"
             iteration_success_flags["editor"] = False

        # --- End of Iteration: Update Previous State for next iteration ---
        previous_story_plan = current_story_plan
        previous_world_details = current_world_details
        previous_character_profiles = current_character_profiles
        previous_outline_text = current_outline_text # Save the full text generated in this iteration
        # Parse and store the structured outline for the next iteration's batch context
        parsed_previous_outline, _ = _parse_outline(previous_outline_text, num_chapters)

        if not all(iteration_success_flags.values()):
             print(f"警告：第 {i+1} 轮部分步骤未能成功完成。")

        print("暂停10秒...")
        time.sleep(10)

    # --- End Main Refinement Loop ---
    # ... (Handle reaching max iterations as before) ...
    print(f"警告：达到最大精炼次数 ({MAX_FULL_REFINEMENT_ITERATIONS})，大纲仍未获得最终批准。")
    print("将使用最后一轮生成的结果。建议检查最后一次的编辑反馈。")

    if current_story_plan: save_content("final_unapproved_story_plan.txt", current_story_plan)
    if current_world_details: save_content("final_unapproved_world_details.txt", current_world_details)
    if current_character_profiles: save_content("final_unapproved_character_profiles.txt", current_character_profiles)
    if current_outline_text: save_content("final_unapproved_outline.txt", current_outline_text)

    final_parsed_outline, final_status = _parse_outline(current_outline_text, num_chapters)
    if final_parsed_outline and final_status != "Parse Error":
        print("最后一轮大纲解析完成（可能不完美）。")
        return final_parsed_outline, current_character_profiles, current_world_details, current_story_plan
    else:
        print("警告：最后一轮大纲未能成功生成或解析。")
        return None, current_character_profiles, current_world_details, current_story_plan

# Note: Need to ensure get_outline_editor_agent also accepts num_chapters if its prompt needs it
# Example in agents.py:
# def get_outline_editor_agent(num_chapters: int) -> autogen.AssistantAgent:
#     prompt = OUTLINE_EDITOR_PROMPT.format(num_chapters=num_chapters)
#     return autogen.AssistantAgent(
#         name="大纲编辑",
#         system_message=prompt,
#         llm_config=reasoning_llm_config_to_use,
#         **DEFAULT_AGENT_CONFIG
#     )