# main.py (修改后)

from config import llm_config
from outline_module import generate_refined_outline # 只导入主函数
from book_module import BookGenerator
from utils import save_content, load_content, OUTPUT_DIR # 导入 OUTPUT_DIR
import json
import os

# --- 配置区 ---
NUM_CHAPTERS = 10
INITIAL_PROMPT = """
请创作一个故事，核心元素如下：
故事背景设定在近未来的上海，一家名为“幻境科技”的VR游戏公司内部。
主角是一位名叫林潇的年轻女程序员，她非常有才华但性格内向，不善交际。她独立开发了一款具有革命性情感交互系统的VR游戏demo。
主要冲突：公司内部的技术总监（李峰，一个野心勃勃的中年男人）想要窃取她的成果据为己有，而另一位资深但正直的游戏策划师（陈忶，女性，林潇的潜在导师）则发现了李峰的企图，试图帮助林潇。
故事线：林潇在秘密完善demo -> 李峰开始打压并试图获取代码 -> 陈忶介入调查与帮助 -> 在公司年度发布会前的高潮对决 -> 林潇最终保护了自己的成果并获得认可（或选择离开创业）。
风格：现代都市职场、科技惊悚、略带成长的励志元素。视角主要跟随林潇。
"""

# --- 主流程 ---
def run_generation():
    print("--- AutoGen Book Generator (循环精炼版) ---")

    # --- 阶段 1 & 2 合并: 大纲、角色、设定等的生成与精炼 ---
    # 这个函数现在包含了完整的迭代循环
    final_outline_structure, final_character_profiles, final_world_details, final_story_plan = \
        generate_refined_outline(
            initial_prompt=INITIAL_PROMPT,
            num_chapters=NUM_CHAPTERS
        )

    # 检查 generate_refined_outline 的结果
    if final_outline_structure is None:
        print("错误：未能生成或批准最终大纲结构，流程终止。")
        print(f"请检查 '{OUTPUT_DIR}/' 目录下的 iteration_*.txt 文件了解详情。")
        # 即使大纲结构解析失败，后续步骤可能仍需要文本文件
        if not final_character_profiles:
             print("角色档案也缺失，无法继续。")
             return
        else:
             print("警告：大纲结构解析失败，但将尝试使用最终文本文件继续生成章节。")
             # 尝试从文件加载文本大纲
             final_outline_text = load_content("final_outline.txt") or load_content("final_unapproved_outline.txt")
             if not final_outline_text:
                  print("错误：无法加载任何最终大纲文本文件，流程终止。")
                  return
             # 尝试在 book_module 中重新解析
             # 注意: _parse_outline 需要可访问，或者将解析逻辑移到 book_module
             from outline_module import _parse_outline
             final_outline_structure = _parse_outline(final_outline_text, NUM_CHAPTERS)
             if not final_outline_structure:
                 print("错误：无法从最终文本文件解析大纲，流程终止。")
                 return

    # 确保其他必要信息也加载了 (generate_refined_outline 应该已保存最终版本)
    if not final_character_profiles:
        final_character_profiles = load_content("final_character_profiles.txt") or \
                                   load_content("final_unapproved_character_profiles.txt")
        if not final_character_profiles:
             print("错误：无法加载角色档案，无法继续。")
             return

    if not final_world_details:
        final_world_details = load_content("final_world_details.txt") or \
                              load_content("final_unapproved_world_details.txt")
        if not final_world_details:
             print("警告：无法加载世界设定。")
             final_world_details = "" # 使用空文本继续

    print("\n--- 大纲精炼完成（或达到最大迭代次数） ---")
    print(f"最终使用的角色档案文件: {'已加载' if final_character_profiles else '未找到'}")
    print(f"最终使用的世界设定文件: {'已加载' if final_world_details else '未找到'}")
    print(f"最终使用的大纲结构: {'已生成' if final_outline_structure else '解析失败'}")

    # 保存最终结构化大纲为 JSON (如果成功解析)
    if final_outline_structure:
        try:
            save_content("final_outline_structure.json", json.dumps(final_outline_structure, ensure_ascii=False, indent=4))
        except Exception as e:
            print(f"保存最终结构化大纲 JSON 时出错: {e}")


    # --- 阶段 3: 书籍章节生成 ---
    # 初始化书籍生成器
    book_generator = BookGenerator(
        final_outline=final_outline_structure,
        character_profiles=final_character_profiles,
        world_details=final_world_details
    )

    # 开始生成章节
    book_generator.generate_book()

    print("\n--- 全部流程结束 ---")

if __name__ == "__main__":
    run_generation()