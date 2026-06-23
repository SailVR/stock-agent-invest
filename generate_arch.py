#!/usr/bin/env python3
"""生成本地架构图 ARCHITECTURE.png（使用 matplotlib，支持中文）。"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis("off")

# 颜色
C = {
    "primary": "#4f46e5", "green": "#10b981", "orange": "#f59e0b",
    "red": "#ef4444", "gray": "#6b7280", "text": "#1f2937", "label": "#9ca3af",
}

def bg_box(x, y, w, h, title, color):
    c = mcolors.to_rgba(color, 0.06)
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=c, edgecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.3, title, ha="center", va="top",
            fontsize=11, fontweight="bold", color=color)

def nd(x, y, label, sub="", color="#4f46e5"):
    w, h = 2.8, 0.65 if not sub else 0.85
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.1",
                          facecolor="white", edgecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y + (0.12 if sub else 0), label, ha="center", va="center",
            fontsize=9, fontweight="bold", color=color)
    if sub:
        ax.text(x, y - 0.22, sub, ha="center", va="top",
                fontsize=7, color=C["label"])

def ar(x1, y1, x2, y2, label="", color="#d1d5db"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.12, label, ha="center", va="bottom",
                fontsize=7, color=C["label"], style="italic")

# ── 标题 ──
ax.text(9, 13.5, "Portfolio Agent - 系统架构图", ha="center", va="center",
        fontsize=16, fontweight="bold", color=C["text"])

# ── Layer 1: 用户交互 ──
bg_box(0.5, 11.5, 17, 1.5, "1. 用户交互层", C["primary"])
nd(3, 12.3, "上传持仓 CSV")
nd(6, 12.3, "聊天输入")
nd(9, 12.3, "刷新 / 清空")

# ── Layer 2: Web ──
bg_box(0.5, 9.8, 17, 1.5, "2. Web 层 (Flask)", C["primary"])
nd(3, 10.5, "/analyze", "SSE 流式分析")
nd(6, 10.5, "/chat", "SSE 流式聊天")
nd(9, 10.5, "/result", "分析结果页")
nd(12, 10.5, "/api/memory", "偏好/历史接口")
ar(3, 12.0, 3, 11.15); ar(6, 12.0, 6, 11.15)

# ── Layer 3: Agent ──
bg_box(0.5, 5.0, 17, 4.8, "3. Agent 层 (LangGraph)", C["green"])

nd(9, 9.0, "Portfolio Agent", "主流程编排")

bg_box(1.5, 6.8, 15, 1.6, "三路并行执行", C["label"])
nd(3, 7.8, "Market Agent", "指数+涨停+情绪", C["green"])
nd(6, 7.8, "Company Agent", "财务+新闻+评级", C["orange"])
nd(9, 7.8, "Risk Agent", "ST+停牌检测", C["red"])

nd(9, 6.3, "merge_analysis", "屏障节点(全部完成才放行)", C["gray"])
nd(5.5, 5.5, "Strategy Agent", "策略建议")
nd(12.5, 5.5, "Report Agent", "投资日报")

ar(9, 8.7, 3, 8.15, "并行"); ar(9, 8.7, 6, 8.15, "并行"); ar(9, 8.7, 9, 8.15, "并行")
ar(3, 7.45, 9, 6.65); ar(6, 7.45, 9, 6.65); ar(9, 7.45, 9, 6.65)
ar(9, 5.95, 5.5, 5.85); ar(9, 5.95, 12.5, 5.85)

# ── Layer 4: 数据源 ──
bg_box(0.5, 2.8, 17, 1.8, "4. 数据源层", C["orange"])
nd(3, 3.6, "Tushare Pro", "财务/指数/停牌")
nd(7, 3.6, "AKShare", "涨停池/新闻/ST")
nd(11, 3.6, "千问 / DeepSeek", "意图识别/分析/生成")
ar(3, 5.0, 3, 3.95); ar(7, 5.0, 7, 3.95); ar(11, 5.0, 11, 3.95)

# ── Layer 5: 持久化 ──
bg_box(0.5, 1.2, 17, 1.5, "5. 持久化层", C["gray"])
nd(4, 1.9, "SQLite 记忆库", "持仓/分析/偏好")
nd(8, 1.9, "FAISS + BM25", "混合检索+JSONL")
nd(12, 1.9, "Logs", "滚动日志文件")
ar(12, 5.15, 12, 2.25); ar(6, 5.15, 4, 2.25); ar(6, 5.15, 8, 2.25)

# ── 图例 ──
lx, ly = 0.8, 0.3
for i, (l, c) in enumerate([
    ("用户/Web", C["primary"]), ("Agent", C["green"]),
    ("数据源", C["orange"]), ("持久化", C["gray"])]):
    ax.plot([lx + i*3.8, lx + i*3.8 + 0.4], [ly, ly], color=c, linewidth=3)
    ax.text(lx + i*3.8 + 0.5, ly - 0.03, l, fontsize=7, color=C["label"], va="center")

plt.savefig("ARCHITECTURE.png", dpi=150, bbox_inches="tight",
            facecolor="#f8f9fa", edgecolor="none")
print("OK - ARCHITECTURE.png")
