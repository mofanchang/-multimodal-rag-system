import os
import time
import operator
from typing import Annotated, List, TypedDict, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# --- 專業 LangGraph 組件 ---
try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

load_dotenv()

# --- 1. 定義專業 State ---
class AgentState(TypedDict):
    # 使用 Annotated[list, operator.add] 是 LangGraph 處理訊息流的標準做法
    messages: Annotated[List[BaseMessage], operator.add]
    plan: List[str]
    code: str
    review_comments: List[str]
    metrics: Dict[str, Dict]
    approved: bool

# --- 2. 定義原子 Tools (Agent Skills) ---
@tool
def fetch_guidelines(topic: str) -> str:
    """從企業知識庫中檢索特定的編碼規範。"""
    try:
        with open("guidelines.md", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "找不到相關規範。"

@tool
def security_scanner(code: str) -> str:
    """對代碼進行靜態安全掃描，檢查密鑰洩漏。"""
    if "api_key" in code.lower():
        return "❌ 警告：檢測到硬編碼密鑰！"
    return "✅ 安全掃描通過。"

tools = [fetch_guidelines, security_scanner]
tool_node = ToolNode(tools)

# --- 3. 定義 Agents ---

def research_agent(state: AgentState):
    print("\n🔍 [Agent: Research] 正在檢索企業規範...")
    # 在真實場景中，這裡會調用 model.bind_tools(tools)
    content = fetch_guidelines.invoke({"topic": "python_async"})
    return {
        "messages": [AIMessage(content=f"已獲取規範：{content[:50]}...")],
        "metrics": {"Research": {"tokens": 200, "cost": "$0.0020"}}
    }

def planner_agent(state: AgentState):
    print("📝 [Agent: Planner] 制定開發計劃...")
    plan = ["1. 建立異步結構", "2. 實作安全檢查邏輯"]
    return {
        "plan": plan,
        "messages": [AIMessage(content="計劃已制定。")],
        "metrics": {"Planner": {"tokens": 150, "cost": "$0.0015"}}
    }

def coder_agent(state: AgentState):
    print("💻 [Agent: Coder] 撰寫實作代碼...")
    code = "async def get_data():\n    return {'status': 'ok'}"
    return {
        "code": code,
        "messages": [AIMessage(content="代碼已生成。")],
        "metrics": {"Coder": {"tokens": 600, "cost": "$0.0060"}}
    }

def reviewer_agent(state: AgentState):
    print("⚖️ [Agent: Reviewer] 執行 RAG 審查與安全掃描...")
    # 調用安全工具
    scan_result = security_scanner.invoke({"code": state["code"]})
    return {
        "review_comments": [scan_result, "✅ 符合異步開發規範。"],
        "messages": [AIMessage(content="審查完成。")],
        "metrics": {"Reviewer": {"tokens": 400, "cost": "$0.0040"}}
    }

def reporter_node(state: AgentState):
    print("📊 [Agent: Reporter] 彙整最終報告...")
    total_cost = sum(float(m["cost"].replace("$", "")) for m in state["metrics"].values())
    report = f"""
=====================================================
🚀 PR DRAFT & OBSERVABILITY REPORT (Enterprise Ver.)
=====================================================
【任務代碼】:
{state['code']}

【審核意見】:
{chr(10).join(state['review_comments'])}

-----------------------------------------------------
📈 成本監控 (Observability)
-----------------------------------------------------
總 Token 預估: {sum(m['tokens'] for m in state['metrics'].values())}
總計費用: ${total_cost:.5f}
=====================================================
"""
    print(report)
    return state

# --- 4. 構建狀態機 (Graph) ---

workflow = StateGraph(AgentState)

workflow.add_node("research", research_agent)
workflow.add_node("planner", planner_agent)
workflow.add_node("coder", coder_agent)
workflow.add_node("reviewer", reviewer_agent)
workflow.add_node("reporter", reporter_node)

workflow.set_entry_point("research")
workflow.add_edge("research", "planner")
workflow.add_edge("planner", "coder")
workflow.add_edge("coder", "reviewer")
workflow.add_edge("reviewer", "reporter")
workflow.add_edge("reporter", END)

app = workflow.compile()

if __name__ == "__main__":
    # 初始化一個專業的 State
    inputs = {
        "messages": [HumanMessage(content="請幫我開發一個企業數據接口")],
        "plan": [],
        "code": "",
        "review_comments": [],
        "metrics": {},
        "approved": False
    }
    app.invoke(inputs)
