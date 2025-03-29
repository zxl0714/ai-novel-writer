# book_module.py
# Implements the logic for generating book chapters using dynamic scene planning,
# scene-by-scene writing/editing, and final length adjustment.

import autogen
from typing import Optional, List, Dict, Any, Union, Tuple
# Import agent retrieval functions and configurations
from agents import (
    get_memory_keeper_agent, get_writer_agent, get_editor_agent,
    get_user_proxy_agent, get_scene_planner_agent,
    get_scene_plan_editor_agent 
)
from config import llm_config, reasoning_llm_config_to_use, DEFAULT_AGENT_CONFIG
# Import utility functions
from utils import save_content, extract_section, OUTPUT_DIR, load_content
import os
import time
import re # For parsing

# --- Constants ---
MAX_SCENE_REVISION_ROUNDS = 2 # Max revision rounds for a single scene (Draft + 2 Revisions)
TARGET_CHAPTER_LENGTH = 4000  # Target character count per chapter (Hard Requirement)
MIN_CHAPTER_LENGTH = 3800   # Minimum acceptable character count after expansion
MAX_EXPANSION_ATTEMPTS = 2  # Max attempts for the final expansion step if chapter is short
CHAPTER_GENERATION_ATTEMPTS = 2 # Max attempts to generate a whole chapter if major errors occur

# Scene Length Calculation Parameters
SCENE_LENGTH_FACTOR = 1.15  # Target scene length multiplier (e.g., 1.15 = aim 15% higher than average)
MIN_SCENE_TARGET_LENGTH = 900 # Minimum target length for any single scene calculation
SCENE_ACCEPTABLE_FACTOR = 0.75 # Minimum acceptable length for a scene relative to its target (e.g., 0.75 = 75%)

# --- Helper Function for Parsing Scene List ---
# Included here for self-containment, can be moved to utils.py
def parse_scene_list(text: str) -> Optional[List[str]]:
    """
    Parses the scene list from the Scene Planner's response.
    Expects a format like:
    【场景列表】
    1. Scene description 1...
    2. Scene description 2...
    ...
    【列表结束】
    Returns a list of scene description strings.
    """
    if not text: return None

    # 1. Extract content between markers
    scene_section = extract_section(text, "【场景列表】", "【列表结束】")

    if not scene_section:
        print("警告：未能从 Scene Planner 回复中找到【场景列表】和【列表结束】标记。尝试直接解析完整文本（可能不准确）。")
        scene_section = text # Fallback to whole text

    scenes = []
    # 2. Parse numbered list within the extracted section
    # Regex looks for lines starting with number, period, optional space, then captures the rest
    pattern = re.compile(r"^\s*(\d+)\s*\.\s*(.*)", re.MULTILINE)
    lines = scene_section.strip().split('\n')
    expected_next_num = 1
    current_scene_buffer = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line: continue

        match = pattern.match(stripped_line)
        if match:
            num = int(match.group(1))
            desc_part = match.group(2).strip()

            # If it's the expected next number, finalize the previous scene and start new
            if num == expected_next_num:
                if current_scene_buffer: # Save the previous scene description
                    scenes.append("\n".join(current_scene_buffer).strip())
                current_scene_buffer = [desc_part] # Start new scene buffer
                expected_next_num += 1
            # If number is unexpected, treat as continuation of previous scene or ignore
            elif current_scene_buffer:
                 # Assume it's part of the previous scene's description if number is wrong
                 current_scene_buffer.append(stripped_line)
            # else: ignore lines before the first valid number
        elif current_scene_buffer:
            # If not a new numbered item, append to the current scene's description
            current_scene_buffer.append(stripped_line)

    # Append the last scene description
    if current_scene_buffer:
        scenes.append("\n".join(current_scene_buffer).strip())

    if not scenes:
        print("错误：未能成功解析出任何有效的场景描述。")
        return None

    print(f"成功解析出 {len(scenes)} 个场景。")
    return scenes

# --- BookGenerator Class ---
class BookGenerator:
    """
    Orchestrates the generation of a book using dynamic scene planning,
    scene-by-scene writing/editing, and length control.
    """
    def __init__(self, final_outline: List[Dict], character_profiles: str, world_details: str):
        """
        Initializes the BookGenerator.
        Args:
            final_outline: The final approved structured outline (List[Dict]).
            character_profiles: Full text of character profiles.
            world_details: Full text of world details.
        Raises:
            TypeError: If final_outline is not a list of dictionaries.
        """
        if not isinstance(final_outline, list):
             raise TypeError("final_outline 必须是字典列表")
        if not all(isinstance(item, dict) for item in final_outline):
             raise TypeError("final_outline 必须是字典列表")

        self.final_outline = final_outline
        self.character_profiles = character_profiles if character_profiles else ""
        self.world_details = world_details if world_details else ""
        self.chapters_memory: List[str] = [] # Stores chapter summaries
        self.output_dir = OUTPUT_DIR
        self.chapters_output_dir = os.path.join(self.output_dir, "chapters")
        os.makedirs(self.chapters_output_dir, exist_ok=True)

        # Initialize Agents
        print("Initializing agents for BookGenerator...")
        try:
            self.memory_keeper = get_memory_keeper_agent()
            self.scene_planner = get_scene_planner_agent()
            self.writer = get_writer_agent()
            # Use reasoning model for Outline Editor, potentially for Chapter Editor too
            self.scene_plan_editor = get_scene_plan_editor_agent(target_chapter_length=TARGET_CHAPTER_LENGTH)
            # Decide which config for chapter editor - let's use default for now unless specified
            # self.editor = get_editor_agent(llm_config_override=reasoning_llm_config_to_use) # Option: Use reasoning
            self.editor = get_editor_agent() # Option: Use default
            self.user_proxy = get_user_proxy_agent()
            print("Agents initialized successfully.")
        except Exception as e:
            print(f"错误：初始化 Agents 时发生严重错误: {e}")
            raise # Reraise the exception to stop execution if agents can't init

        # Stores the planned scenes for the chapter currently being generated
        self.current_chapter_scenes: Optional[List[str]] = None

    def _get_chapter_context(self, chapter_number: int) -> str:
        """Prepares context from previous chapters via Memory Keeper."""
        context_parts = []
        if self.chapters_memory:
            context_parts.append("【先前章节概要回顾 (由记忆守护者提供)】")
            max_prev_chapters = 7 # Limit context length
            start_index = max(0, len(self.chapters_memory) - max_prev_chapters)
            for i, summary in enumerate(self.chapters_memory[start_index:], start=start_index + 1):
                context_parts.append(f"第 {i} 章概要:\n{summary}")
            context_parts.append("-" * 20)
        # TODO: Consider dynamically calling Memory Keeper here for more tailored context if needed
        return "\n".join(context_parts)

    def _write_and_edit_scene(
        self,
        chapter_number: int,
        scene_index: int,
        num_scenes_planned: int,
        scene_description: str, # Now assuming this is a string description
        chapter_outline_prompt: str,
        chapter_context: str,
        previous_scene_summary: Optional[str]
        ) -> Optional[str]:
        """
        Handles the writing, editing, and revision loop for a single scene.
        Returns the approved scene content (str) or None if failed.
        """
        print(f"    场景 {scene_index + 1}/{num_scenes_planned}: 开始生成...")
        scene_content: Optional[str] = None
        editor_feedback: str = ""

        # --- Calculate Target and Reference Lengths ---
        if num_scenes_planned <= 0: base_avg_length = 900
        else: base_avg_length = TARGET_CHAPTER_LENGTH / num_scenes_planned
        # Writer's target is inflated
        target_scene_length = max(MIN_SCENE_TARGET_LENGTH, int(base_avg_length * SCENE_LENGTH_FACTOR))
        # Editor's reference baseline (e.g., simple average or minimum target)
        editor_reference_length = max(MIN_SCENE_TARGET_LENGTH, int(base_avg_length))
        # Minimum acceptable length for a scene after editing
        min_acceptable_scene_length = int(target_scene_length * SCENE_ACCEPTABLE_FACTOR)

        print(f"    场景 {scene_index + 1}: Writer 目标约 {target_scene_length} 字, Editor 参考基准约 {editor_reference_length} 字 (最低接受 {min_acceptable_scene_length} 字)。")

        # --- Revision Loop ---
        for revision_round in range(MAX_SCENE_REVISION_ROUNDS + 1):
            is_revision = revision_round > 0
            action = f"修改稿({revision_round})" if is_revision else "初稿"

            # --- Prepare Writer Prompt ---
            # Include full context without truncation
            writer_prompt_context = f"""【章节整体上下文概要】
{chapter_context if chapter_context else "无"}

【角色档案参考】(完整)
{self.character_profiles}

【世界设定参考】(完整)
{self.world_details}

【本章大纲要求】(完整)
{chapter_outline_prompt}

【先前场景概要】(如果不是第一个场景)
{previous_scene_summary if previous_scene_summary else "无"}
"""
            # Add special instructions for first/last scenes
            special_instructions = ""
            is_first_scene = (scene_index == 0)
            is_last_scene = (scene_index == num_scenes_planned - 1)
            if is_first_scene:
                special_instructions += "\n**特别注意：这是本章的第一个场景。** 请确保开头能顺畅衔接【先前章节概要回顾】（如果是第一章则需有力开局），并快速建立本章的初始情境和氛围。\n"
                if chapter_number == 1:
                     special_instructions += "**这是全书开篇！必须极度抓人眼球，迅速引入核心设定或冲突！**\n"
            if is_last_scene:
                special_instructions += "\n**特别注意：这是本章的最后一个场景。** 请在完成场景核心内容后，务必设计一个强有力的结尾（如小高潮解决、总结反思、或强烈悬念钩子），引导读者期待下一章。确保收束本章主要线索或达成章节目标。\n"

            # Assemble the specific prompt for this round
            if not is_revision:
                writer_prompt = f"""{writer_prompt_context}
【当前场景要求 ({scene_index + 1}/{num_scenes_planned})】
{scene_description}
{special_instructions}
【写作指令】
请根据以上所有信息，创作**当前场景**的内容。注意与先前场景的衔接。
目标长度：约 {target_scene_length} 字。
输出格式：【场景草稿】...【草稿完】"""
            else:
                if not scene_content or not editor_feedback:
                    print(f"错误：缺少上一稿内容或编辑反馈，无法进行场景 {scene_index + 1} 的修改。")
                    return None
                writer_prompt = f"""{writer_prompt_context}
【当前场景要求 ({scene_index + 1}/{num_scenes_planned})】
{scene_description}
{special_instructions}
【编辑反馈】(请根据这些建议修改你上一稿的内容)
{editor_feedback}
【你上一稿的场景内容】(供参考)
{scene_content}
【写作指令】
请根据【编辑反馈】修改【你上一稿的场景内容】，输出修改后的**完整当前场景**。
目标长度：约 {target_scene_length} 字。
输出格式：【场景草稿】...【草稿完】"""

            # --- Call Writer ---
            print(f"    场景 {scene_index + 1}: 调用 Writer 生成 {action}...")
            current_scene_len = 0
            try:
                self.user_proxy.reset()
                self.user_proxy.initiate_chat(recipient=self.writer, message=writer_prompt, max_turns=1, clear_history=True)
                writer_msg = self.user_proxy.last_message(self.writer)
                if writer_msg and writer_msg["content"]:
                    extracted_content = extract_section(writer_msg["content"], "【场景草稿】", "【草稿完】")
                    if extracted_content:
                        scene_content = extracted_content
                        current_scene_len = len(scene_content)
                        print(f"    场景 {scene_index + 1}: {action} 生成成功 (实际长度: {current_scene_len} 字)。")
                    else:
                        print(f"错误：未能从 Writer 回复中提取场景 {scene_index + 1} 的 {action}。")
                        return None
                else:
                    print(f"错误：未能从 Writer 获取场景 {scene_index + 1} 的 {action} 回复。")
                    return None
            except Exception as e:
                print(f"生成场景 {scene_index + 1} {action} 时出错: {e}")
                return None

            # --- Call Editor ---
            print(f"    场景 {scene_index + 1}: 调用 Editor 评审 {action} (实际长度: {current_scene_len} 字)...")
            # Use full context for editor as well
            editor_prompt_context = writer_prompt_context # Reuse context part
            # Add special checks for first/last scenes
            editor_special_checks = ""
            if is_first_scene:
                editor_special_checks += "\n**评审重点：** 请特别检查此开篇场景与上文的衔接是否流畅自然？是否有效开启了本章内容？"
                if chapter_number == 1:
                    editor_special_checks += " **此为全书开篇，吸引力是否足够强？**"
            if is_last_scene:
                editor_special_checks += "\n**评审重点：** 请特别检查此结尾场景是否有效收束了本章或达到了预期的结尾效果（如悬念、高潮）？是否为下一章留下了合适的接口？"

            editor_prompt = f"""请评审以下**单个场景**的草稿。

{editor_prompt_context}

【当前场景要求 ({scene_index + 1}/{num_scenes_planned})】
{scene_description}

【系统提示：当前场景草稿实际长度约 {current_scene_len} 字。单场景内容充实度参考基准约 {editor_reference_length} 字。】

【当前场景草稿】
{scene_content}
【草稿结束】

请根据你的系统提示（评审流程、标准），判断此场景是否合格。
{editor_special_checks}
输出【编辑通过】或【编辑反馈】+【修改建议】。"""

            try:
                self.user_proxy.reset()
                self.user_proxy.initiate_chat(recipient=self.editor, message=editor_prompt, max_turns=1, clear_history=True)
                editor_msg = self.user_proxy.last_message(self.editor)
                if editor_msg and editor_msg["content"]:
                    editor_feedback = editor_msg["content"]
                    print(f"    场景 {scene_index + 1}: 编辑评审完成。")

                    # Process Editor Feedback
                    if "【编辑通过】" in editor_feedback:
                        # Final check on minimum acceptable length even if editor approved
                        if current_scene_len < min_acceptable_scene_length and revision_round < MAX_SCENE_REVISION_ROUNDS:
                             print(f"    场景 {scene_index + 1}: 编辑通过但长度 ({current_scene_len} 字) 低于最低接受值 ({min_acceptable_scene_length} 字)，强制要求扩写。")
                             # Force next round by creating feedback asking for expansion
                             editor_feedback = f"【编辑反馈】\n总体评价：内容尚可但过于简略。\n【修改建议】\n- 针对【内容充实度/长度问题】: 当前场景长度 {current_scene_len} 字 不足 {min_acceptable_scene_length} 字，请扩充至 {target_scene_length} 字 左右，增加更多细节。"
                             # Continue to the next iteration of the loop
                        else:
                             print(f"    场景 {scene_index + 1}: 编辑通过 (最终长度 {current_scene_len} 字)。")
                             return scene_content # Scene approved, return content

                    elif "【编辑反馈】" in editor_feedback and revision_round < MAX_SCENE_REVISION_ROUNDS:
                         print(f"    场景 {scene_index + 1}: 编辑提出修改意见，将进行修改 ({revision_round + 1}/{MAX_SCENE_REVISION_ROUNDS})。")
                         # Keep editor_feedback and continue to the next iteration
                    else: # Max revisions reached or unknown feedback format
                         if revision_round >= MAX_SCENE_REVISION_ROUNDS:
                              print(f"    场景 {scene_index + 1}: 达到最大修改次数，采用当前版本 (长度 {current_scene_len} 字)。")
                         else:
                              print(f"    场景 {scene_index + 1}: 编辑反馈格式未知或未要求修改，采用当前版本 (长度 {current_scene_len} 字)。")
                         return scene_content # Use current version
                else:
                     print(f"错误：未能从 Editor 获取场景 {scene_index + 1} 的评审回复。采用当前版本。")
                     return scene_content # Use current version if editor fails
            except Exception as e:
                 print(f"评审场景 {scene_index + 1} 时出错: {e}。采用当前版本。")
                 return scene_content # Use current version if exception during edit

        # Should ideally return within the loop
        print(f"警告：场景 {scene_index + 1} 的写作/编辑循环意外结束。返回最后生成的内容。")
        return scene_content

    def _generate_chapter(self, chapter_data: Dict) -> Optional[str]:
        """
        Generates a single chapter's content using dynamic scene planning,
        scene-by-scene writing/editing, and final length adjustment.
        """
        chapter_number = chapter_data.get("chapter_number")
        chapter_title = chapter_data.get("title", "无标题")
        chapter_outline_prompt = chapter_data.get("prompt", "")

        if chapter_number is None or not chapter_outline_prompt:
            print(f"错误：章节数据不完整 (编号: {chapter_number}, 大纲: {'存在' if chapter_outline_prompt else '缺失'})。")
            return None

        print(f"\n--- 开始生成 第 {chapter_number} 章: {chapter_title} ---")
        print("步骤 1: 进行动态场景规划...")
        self.current_chapter_scenes = None # Reset for the chapter

        # --- 1. Call Scene Planner ---
        chapter_context = self._get_chapter_context(chapter_number) # Get previous chapter summaries
        scene_planner_prompt = f"""请根据以下【本章大纲要求】，并参考【角色档案】和【世界设定】，将其细化分解为一系列（通常 4-5 个）逻辑连贯的场景。
最终目标是让这些场景组合起来能够支撑起一章大约 {TARGET_CHAPTER_LENGTH} 字的内容。

【本章大纲要求】
{chapter_outline_prompt}

【角色档案参考】(完整)
{self.character_profiles}

【世界设定参考】(完整)
{self.world_details}

【先前章节概要回顾】(供参考)
{chapter_context if chapter_context else "无"}

请严格按照你的系统提示格式输出【场景列表】，并以【列表结束】标记结尾。确保场景划分考虑了角色行为逻辑和环境因素。"""

        scenes_text: Optional[str] = None
        try:
            self.user_proxy.reset()
            self.user_proxy.initiate_chat(
                recipient=self.scene_planner,
                message=scene_planner_prompt,
                max_turns=1,
                clear_history=True
            )
            planner_msg = self.user_proxy.last_message(self.scene_planner)
            if planner_msg and planner_msg["content"]:
                scenes_text = planner_msg["content"]
            else:
                 print(f"错误：未能从场景规划师获取回复 (第 {chapter_number} 章)。")
                 return None
        except Exception as e:
             print(f"场景规划时出错 (第 {chapter_number} 章): {e}")
             return None

        # --- Parse Scene List ---
        scenes = parse_scene_list(scenes_text) if scenes_text else None
        if not scenes:
            print(f"错误：场景规划师未能生成有效的场景列表 (第 {chapter_number} 章)。")
            # Fallback: Maybe try writing the whole chapter at once? Or just fail.
            return None # Fail for now

        self.current_chapter_scenes = scenes
        save_content(f"chapter_{chapter_number:03d}_scenes_planned.txt", "\n".join([f"{i+1}. {s}" for i, s in enumerate(scenes)]))
        print(f"场景规划完成，共 {len(scenes)} 个场景。")

        # --- 1.5. Review Scene Plan ---
        print("步骤 1.5: 审核场景规划...")
        scene_list_for_review = "\n".join([f"{i+1}. {s}" for i, s in enumerate(self.current_chapter_scenes)])
        scene_review_prompt = f"""请评审以下针对某章节大纲规划出的【场景列表】。

【本章大纲要求】
{chapter_outline_prompt}

【角色档案参考】(完整)
{self.character_profiles}

【世界设定参考】(完整)
{self.world_details}

【规划的场景列表】
{scene_list_for_review}
【列表结束】

请评估此场景列表是否：
1. 逻辑连贯，能有效推进章节情节，并与角色设定、世界规则相符？
2. 完整覆盖了【本章大纲要求】的关键内容？
3. 场景划分和描述是否清晰、合理，便于【写手】创作？
4. 基于场景描述，预估能否支撑约 {TARGET_CHAPTER_LENGTH} 字的章节内容？

请输出【场景规划批准】或【场景规划反馈】+【修改建议】。"""

        final_scenes_to_use = self.current_chapter_scenes # Default to using initially planned scenes
        try:
            self.user_proxy.reset()
            if not hasattr(self, 'scene_plan_editor'):
                # 这个检查理论上不再需要，因为 __init__ 中已初始化
                print("错误：Scene Plan Editor 未在 BookGenerator 中初始化！")
                raise AttributeError("Scene Plan Editor not initialized")
            self.user_proxy.initiate_chat(recipient=self.scene_plan_editor, message=scene_review_prompt, max_turns=1, clear_history=True)
            review_msg = self.user_proxy.last_message(self.scene_plan_editor)

            if review_msg and review_msg["content"]:
                review_feedback = review_msg["content"]
                save_content(f"chapter_{chapter_number:03d}_scenes_review.txt", review_feedback)
                print(f"场景规划评审完成: {review_feedback}...")            
                print(f"场景规划评审完成: {review_feedback[:100]}...")

                if "【场景规划批准】" in review_feedback:
                    print("场景规划获批。")
                    # final_scenes_to_use remains self.current_chapter_scenes
                elif "【场景规划反馈】" in review_feedback:
                    print("场景规划收到修改建议，尝试修订一次...")
                    # --- Attempt Scene Plan Revision ---
                    revision_prompt = f"""请根据以下【评审反馈】修改你之前生成的场景列表。

【评审反馈】
{review_feedback}

【原始章节大纲要求】
{chapter_outline_prompt}
【角色档案参考】(完整)
{self.character_profiles}
【世界设定参考】(完整)
{self.world_details}

【你上一轮生成的场景列表】(供参考)
{scene_list_for_review}

请输出修改后的【场景列表】，并以【列表结束】标记结尾。"""
                    try:
                        self.user_proxy.reset()
                        self.user_proxy.initiate_chat(recipient=self.scene_planner, message=revision_prompt, max_turns=1, clear_history=True)
                        revised_planner_msg = self.user_proxy.last_message(self.scene_planner)
                        if revised_planner_msg and revised_planner_msg["content"]:
                            revised_scenes_text = revised_planner_msg["content"]
                            revised_scenes = parse_scene_list(revised_scenes_text)
                            if revised_scenes:
                                print("场景规划已根据反馈修订。")
                                final_scenes_to_use = revised_scenes # Use revised scenes
                                save_content(f"chapter_{chapter_number:03d}_scenes_revised.txt", "\n".join([f"{i+1}. {s}" for i, s in enumerate(revised_scenes)]))
                            else:
                                print("警告：场景规划修订失败，将使用原始版本。")
                        else:
                                print("警告：未能获取场景规划的修订结果，将使用原始版本。")
                    except Exception as rev_e:
                            print(f"修订场景规划时出错: {rev_e}。将使用原始版本。")
                    # --- End Revision Attempt ---
                else:
                    print("警告：场景规划评审反馈格式未知，将使用原始版本。")
            else:
                print("警告：未能获取场景规划评审结果，将使用原始版本。")
        except Exception as review_e:
            print(f"审核场景规划时出错: {review_e}。将使用原始版本。")

        # Update self.current_chapter_scenes with the final list to use
        self.current_chapter_scenes = final_scenes_to_use
        if not self.current_chapter_scenes:
             print("错误：没有可用的场景列表，无法继续生成章节。")
             return None

        # --- 2. Scene-by-Scene Writing Loop ---
        print(f"步骤 2: 逐一生成和编辑 {len(self.current_chapter_scenes)} 个场景...")
        accumulated_chapter_content = ""
        previous_scene_summary = None

        for i, scene_desc in enumerate(self.current_chapter_scenes):
            scene_content = self._write_and_edit_scene(
                chapter_number=chapter_number,
                scene_index=i,
                num_scenes_planned=len(self.current_chapter_scenes),
                scene_description=scene_desc,
                chapter_outline_prompt=chapter_outline_prompt,
                chapter_context=chapter_context,
                previous_scene_summary=previous_scene_summary
            )

            if scene_content:
                # Use strip() before adding separators
                stripped_scene = scene_content.strip()
                if stripped_scene: # Only append if not empty
                     accumulated_chapter_content += stripped_scene + "\n\n" # Ensure separation
                     # Update summary based on the actual content added
                     summary_len = 200
                     previous_scene_summary = stripped_scene[-summary_len:]
                     print(f"    场景 {i + 1} 完成并拼接。")
                else:
                     print(f"警告：场景 {i + 1} 内容为空，已跳过拼接。")
                     previous_scene_summary = "(上一个场景内容为空)"

            else:
                print(f"错误：场景 {i + 1} 未能成功生成，章节可能不完整！")
                previous_scene_summary = "(上一个场景生成失败)"

            time.sleep(5) # Pause between scenes

        # --- 3. Final Length Check & Expansion ---
        print("步骤 3: 进行最终长度检查...")
        final_content_stripped = accumulated_chapter_content.strip()
        final_length = len(final_content_stripped)
        print(f"章节拼接完成，当前总长度: {final_length} 字。目标: {TARGET_CHAPTER_LENGTH} 字 (最低 {MIN_CHAPTER_LENGTH} 字)。")

        if final_length < MIN_CHAPTER_LENGTH and final_length > 0:
            print(f"警告：当前长度 {final_length} 不足 {MIN_CHAPTER_LENGTH}，尝试进行最终扩写...")
            expansion_attempts_done = 0
            while expansion_attempts_done < MAX_EXPANSION_ATTEMPTS and final_length < MIN_CHAPTER_LENGTH:
                 expansion_attempts_done += 1
                 print(f"  扩写尝试 {expansion_attempts_done}/{MAX_EXPANSION_ATTEMPTS}...")
                 needed_chars = max(100, MIN_CHAPTER_LENGTH - final_length)
                 target_addition_chars = max(needed_chars, int(needed_chars * 1.2))

                 expansion_context = f"""【章节整体上下文概要】
{chapter_context if chapter_context else "无"}

【角色档案参考】(完整)
{self.character_profiles}

【世界设定参考】(完整)
{self.world_details}

【本章大纲要求】(完整)
{chapter_outline_prompt}
"""
                 expansion_prompt = f"""{expansion_context}

【当前章节内容】(当前长度 {final_length} 字)
{final_content_stripped}
【内容结束】

【指令】
以上是本章已有的内容，长度为 {final_length} 字，距离目标 {TARGET_CHAPTER_LENGTH} 字 (最低 {MIN_CHAPTER_LENGTH}) 还需补充约 {needed_chars} 字。
请在**不破坏现有情节和结局氛围**的前提下，生成一段**补充性的文字**（目标约 {target_addition_chars} 字），用于添加到现有内容的末尾或合适的段落之间（优先考虑添加到末尾形成总结或过渡）。补充内容可以是：
1. 对某个场景的细节描写（环境、动作、心理活动）进行深化。
2. 增加一个简短的结尾段落，升华主题、营造氛围或留下余韵。
请**仅仅输出你需要【补充】的文字内容**，不要重复原文，并用 `【补充内容】` 和 `【补充结束】` 将其包裹起来。"""

                 try:
                     self.user_proxy.reset()
                     self.user_proxy.initiate_chat(recipient=self.writer, message=expansion_prompt, max_turns=1, clear_history=True)
                     writer_msg = self.user_proxy.last_message(self.writer)
                     if writer_msg and writer_msg["content"]:
                          addition_content_section = extract_section(writer_msg["content"], "【补充内容】", "【补充结束】")
                          if addition_content_section:
                               addition_content = addition_content_section.strip()
                               addition_length = len(addition_content)
                               if addition_length > 0:
                                    print(f"  获取到补充内容，长度: {addition_length} 字。")
                                    # Append the addition
                                    final_content_stripped += "\n\n" + addition_content
                                    final_length = len(final_content_stripped) # Update total length
                                    accumulated_chapter_content = final_content_stripped # Update main variable
                                    print(f"  追加补充内容后，新总长度: {final_length} 字。")
                               else:
                                    print("  警告：获取到的补充内容为空。")


                               if final_length >= MIN_CHAPTER_LENGTH:
                                    print("  扩写后达到最低长度要求。")
                                    break
                               elif expansion_attempts_done >= MAX_EXPANSION_ATTEMPTS:
                                    print(f"  达到最大扩写次数 {MAX_EXPANSION_ATTEMPTS}。")
                                    break
                               else:
                                    print("  扩写后仍不足最低长度，继续尝试...")
                          else:
                               print("  错误：未能从 Writer 的扩写回复中提取【补充内容】。扩写失败。")
                               break
                     else:
                          print("  错误：未能从 Writer 获取扩写回复。扩写失败。")
                          break
                 except Exception as e:
                      print(f"  扩写时出错: {e}。扩写失败。")
                      break

            if final_length < MIN_CHAPTER_LENGTH:
                 print(f"警告：扩写后长度仍为 {final_length}，未达到最低要求 {MIN_CHAPTER_LENGTH}。将使用当前内容。")

        elif final_length == 0:
             print("错误：章节内容为空，无法进行扩写或保存。")
             return None
        else:
            print("当前长度已满足或超过最低要求，无需扩写。")

        # --- 4. Update Chapter Memory ---
        if final_content_stripped:
            memory_summary_prompt = f"""请根据你作为【故事连续性维护者】的系统指令，为以下刚刚完成的第 {chapter_number} 章内容，生成一份供第 {chapter_number + 1} 章参考的结构化摘要。

【刚刚完成的章节内容】
{final_content_stripped}
【内容结束】

请严格按照你系统提示中定义的格式（核心进展、状态变更、新增信息、结尾悬念、连续性提醒）输出。"""

            try:
                # 确保 memory_keeper 存在
                if not hasattr(self, 'memory_keeper'): raise AttributeError("Memory Keeper not initialized")

                # **修改点 2：使用 generate_reply 并准备解析新格式**
                # 注意：直接调用 agent.generate_reply 可能不会使用完整的聊天历史（如果需要的话）
                # 但对于这种基于单次输入生成摘要的任务通常是足够的。
                summary_response_raw = self.memory_keeper.generate_reply(
                    messages=[{"role": "user", "content": memory_summary_prompt}]
                )

                if isinstance(summary_response_raw, str) and summary_response_raw.strip():
                    # **修改点 3：尝试用新 Prompt 定义的精确 Heder 来提取**
                    summary_header = f"【记忆更新：第 {chapter_number} 章总结 (供第 {chapter_number + 1} 章参考)】"
                    summary = extract_section(summary_response_raw, summary_header)

                    if not summary: # 如果带章节号的 Header 没匹配到，尝试不带章节号的通用 Header
                         summary_header_generic = "【记忆更新：" # 更宽松的匹配
                         summary_section = extract_section(summary_response_raw, summary_header_generic)
                         if summary_section:
                              # 清理可能包含的子标题
                              if summary_section.startswith(f"第 {chapter_number} 章总结"):
                                   summary = re.sub(r"^\s*第\s*\d+\s*章总结\s*\(供第\s*\d+\s*章参考\)\s*[:：]?\s*", "", summary_section).strip()
                              else:
                                   summary = summary_section.strip()
                         else: # 如果连通用 Header 都找不到，使用原始回复（去除可能的对话前缀）
                              summary = summary_response_raw.strip()
                              # 移除可能的 "好的，这是..." 等对话性开头
                              summary = re.sub(r"^\s*(好的|没问题|这是您要的)[，：:,]?\s*", "", summary).strip()

                    # 清理提取出的 summary（可能仍包含外层标记或额外换行）
                    summary = summary.strip()

                    if summary:
                        # **修改点 4：更健壮的记忆列表填充逻辑**
                        # 确保 memory list 长度至少达到 chapter_number - 1
                        while len(self.chapters_memory) < chapter_number - 1:
                             missing_chap_num = len(self.chapters_memory) + 1
                             print(f"警告：检测到缺失的章节 {missing_chap_num} 概要，补充占位符。")
                             self.chapters_memory.append(f"第 {missing_chap_num} 章概要：(因先前错误跳过或未生成)")

                        # 添加或覆盖当前章节概要
                        if len(self.chapters_memory) == chapter_number - 1:
                            self.chapters_memory.append(summary)
                            print(f"第 {chapter_number} 章概要已存入记忆库。")
                        elif len(self.chapters_memory) >= chapter_number:
                             print(f"警告：覆盖已存在的第 {chapter_number} 章记忆。可能由于重试导致。")
                             self.chapters_memory[chapter_number - 1] = summary # 覆盖
                        else: # 理论上不会到这里
                             print(f"错误：记忆库状态异常 ({len(self.chapters_memory)} vs {chapter_number})，无法为第 {chapter_number} 章添加概要。")
                             # 尝试追加，但不保证序号正确
                             self.chapters_memory.append(summary)


                        # (可选) 保存本次生成的概要，方便调试
                        save_content(f"chapter_{chapter_number:03d}_summary.txt", summary)

                    else: # 如果解析和清理后 summary 为空
                         print(f"警告：Memory Keeper 返回的概要内容为空或格式不符。")
                         # 添加占位符
                         while len(self.chapters_memory) < chapter_number -1: self.chapters_memory.append(f"第 {len(self.chapters_memory)+1} 章概要：(未知错误导致缺失)")
                         if len(self.chapters_memory) == chapter_number - 1:
                              self.chapters_memory.append(f"第 {chapter_number} 章概要：(摘要内容为空) {final_content_stripped[:200]}...")
                else:
                    print(f"警告：未能从 Memory Keeper 获取第 {chapter_number} 章的有效概要字符串。")
                    # 添加占位符
                    while len(self.chapters_memory) < chapter_number -1: self.chapters_memory.append(f"第 {len(self.chapters_memory)+1} 章概要：(未知错误导致缺失)")
                    if len(self.chapters_memory) == chapter_number - 1:
                         self.chapters_memory.append(f"第 {chapter_number} 章概要：(摘要获取失败) {final_content_stripped[:200]}...")
            except Exception as mem_e:
                print(f"调用 Memory Keeper 生成概要时出错：{mem_e}")
                # 添加占位符
                while len(self.chapters_memory) < chapter_number -1: self.chapters_memory.append(f"第 {len(self.chapters_memory)+1} 章概要：(未知错误导致缺失)")
                if len(self.chapters_memory) == chapter_number - 1:
                     self.chapters_memory.append(f"第 {chapter_number} 章概要：(摘要生成异常) {final_content_stripped[:200]}...")
        else:
             # ... (处理章节内容为空的情况，这里的占位符逻辑也需要健壮) ...
             print(f"错误：第 {chapter_number} 章最终内容为空，无法更新记忆库。")
             while len(self.chapters_memory) < chapter_number -1: self.chapters_memory.append(f"第 {len(self.chapters_memory)+1} 章概要：(未知错误导致缺失)")
             if len(self.chapters_memory) == chapter_number - 1:
                 self.chapters_memory.append(f"第 {chapter_number} 章生成失败，内容缺失。")


        return final_content_stripped if final_content_stripped else None


    def generate_book(self):
        """Generates all chapters sequentially using the refined process."""
        print("\n--- 开始生成书籍章节 (动态场景模式 + 长度保证 + 场景审核) ---")
        if not self.final_outline:
             print("错误：没有提供有效的大纲，无法生成书籍。")
             return
             
        total_chapters = len(self.final_outline)
        print(f"计划生成 {total_chapters} 章。")

        for i, chapter_data in enumerate(self.final_outline):
            chapter_number = chapter_data.get("chapter_number")
            if chapter_number is None:
                 print(f"错误：大纲数据索引 {i} 缺少 'chapter_number'。跳过此条目。")
                 continue
                 
            # Ensure memory list padding matches chapter number before starting
            while len(self.chapters_memory) < chapter_number - 1:
                missing_chap_num = len(self.chapters_memory) + 1
                print(f"警告：检测到跳过的章节 {missing_chap_num}，补充占位符到记忆库。")
                self.chapters_memory.append(f"第 {missing_chap_num} 章概要：(因先前错误跳过或未生成)")


            print(f"\n处理中：第 {chapter_number}/{total_chapters} 章")

            attempt = 0
            generated_content: Optional[str] = None
            while attempt < CHAPTER_GENERATION_ATTEMPTS:
                attempt += 1
                print(f"第 {chapter_number} 章 - 尝试次数 {attempt}/{CHAPTER_GENERATION_ATTEMPTS}")
                try:
                    # Reset chapter-specific state if retrying
                    self.current_chapter_scenes = None
                    generated_content = self._generate_chapter(chapter_data)
                except Exception as chapter_e:
                     print(f"第 {chapter_number} 章尝试 {attempt} 发生严重错误: {chapter_e}")
                     generated_content = None # Ensure content is None on error

                if generated_content is not None: # Check for None explicitly
                    final_len = len(generated_content)
                    # Check final length against minimum requirement
                    if final_len >= MIN_CHAPTER_LENGTH:
                        filename = f"chapter_{chapter_number:03d}.txt"
                        full_content_to_save = f"第 {chapter_number} 章: {chapter_data.get('title', '无标题')}\n\n{generated_content}"
                        try:
                            relative_path = os.path.join("chapters", filename)
                            save_content(relative_path, full_content_to_save)
                            print(f"第 {chapter_number} 章生成并保存成功 (最终长度: {final_len} 字)。")
                            break # Success, exit retry loop
                        except Exception as save_e:
                             print(f"错误：保存第 {chapter_number} 章时出错: {save_e}。视为生成失败。")
                             generated_content = None # Treat save error as generation failure for retry
                    else:
                         print(f"警告：第 {chapter_number} 章尝试 {attempt} 最终内容长度 ({final_len} 字) 不足 {MIN_CHAPTER_LENGTH} 字，视为失败。")
                         generated_content = None # Length failure, trigger retry if attempts remain
                         time.sleep(10) # Pause before retry on length failure
                else:
                    print(f"第 {chapter_number} 章尝试 {attempt} 失败 (生成内容为空或内部错误)。")
                    time.sleep(15) # Pause longer after complete failure

            # After all attempts for the chapter
            if generated_content is None:
                print(f"错误：生成第 {chapter_number} 章失败，已达到最大尝试次数 ({CHAPTER_GENERATION_ATTEMPTS})。")
                print("跳过此章节，继续下一章...")
                # Ensure memory placeholder is added *only if* it wasn't added by _generate_chapter
                if len(self.chapters_memory) == chapter_number - 1:
                     self.chapters_memory.append(f"第 {chapter_number} 章生成失败，内容缺失。")

            time.sleep(10) # Pause between chapters

        print("\n--- 书籍章节生成完成 ---")
        self._combine_chapters()

    def _combine_chapters(self):
         """Merges all generated chapter files into a single book file."""
         print("\n--- 合并所有章节 ---")
         full_book_content = []
         if not os.path.exists(self.chapters_output_dir):
              print(f"错误：找不到章节目录 {self.chapters_output_dir}")
              return
         try:
             # Ensure correct numerical sorting of chapter files
             files = sorted(
                 [f for f in os.listdir(self.chapters_output_dir) if f.startswith("chapter_") and f.endswith(".txt")],
                 key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else float('inf')
                 )
             if not files:
                 print("错误：在 chapters 目录下没有找到任何章节文件。")
                 return

             for filename in files:
                 filepath = os.path.join(self.chapters_output_dir, filename)
                 try:
                     with open(filepath, "r", encoding="utf-8") as f:
                         full_book_content.append(f.read())
                     print(f"已加载: {filename}")
                 except IOError as e:
                     print(f"错误：无法读取章节文件 {filepath}: {e}")

             if full_book_content:
                 save_content("full_book_combined.txt", "\n\n---\n\n".join(full_book_content))
                 print(f"所有章节已合并到 full_book_combined.txt")
             else:
                 print("没有成功加载任何章节内容进行合并。")

         except Exception as e:
              print(f"合并章节时发生错误: {e}")