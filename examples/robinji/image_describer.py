"""YOLO + DeepSeek 图片描述应用。

ultralytics 采用惰性导入（仅在 get_model 运行时导入），
保证纯函数单元测试无需安装 torch 即可运行。
"""
import html
import os

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image


def load_env_file(path):
    """加载 .env 键值文件到环境变量（不覆盖已存在的环境变量）。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def summarize_results(results):
    """汇总 YOLO 检测结果为 {class, count, avg_conf} 列表。"""
    counts = {}
    confs = {}
    names = {}
    for result in results:
        names = result.names
        if result.boxes is None:
            continue
        for cls_id, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            cls_id = int(cls_id)
            counts[cls_id] = counts.get(cls_id, 0) + 1
            confs.setdefault(cls_id, []).append(float(conf))
    summary = []
    for cls_id, count in counts.items():
        summary.append(
            {"class": names.get(cls_id, str(cls_id)), "count": count, "avg_conf": round(sum(confs[cls_id]) / len(confs[cls_id]), 3)}
        )
    summary.sort(key=lambda d: d["count"], reverse=True)
    return summary


def draw_detections(image_bgr, results):
    """在图片副本上绘制 YOLO 检测框与标签，返回标注后的 BGR 图像。"""
    annotated = image_bgr.copy()
    for result in results:
        if result.boxes is None:
            continue
        for xyxy, cls_id, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{result.names.get(int(cls_id), str(int(cls_id)))} {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - th - baseline), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


def build_prompt(detections):
    """将检测结果拼成 DeepSeek 提示词。detections 为空时提示未检测到目标。"""
    if detections:
        det_text = ", ".join(
            f"{d['class']} x{d['count']}（平均置信度 {d['avg_conf']}）" for d in detections
        )
    else:
        det_text = "未检测到常见目标"
    return (
        "你是一个专业的图片描述助手。下面是 YOLO 目标检测模型对一张图片的识别结果"
        "（类别、数量、平均置信度）：\n"
        f"{det_text}\n"
        "请用一段自然、流畅的中文描述这张图片的内容，包括画面中的主体、数量和场景氛围，"
        "不要生硬罗列数据，也不要编造检测结果之外的具体细节。"
    )


def call_deepseek(prompt, api_key, temperature=0.7, max_tokens=300):
    """调用 DeepSeek chat completions 接口，返回生成的文本。"""
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是图片描述助手，请用中文回答。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek API 返回错误 (HTTP {resp.status_code}): {resp.text[:200]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


@st.cache_resource
def get_model(model_path):
    """加载并缓存 YOLO 模型。"""
    from ultralytics import YOLO

    return YOLO(model_path)


CSS = """
<style>
.stApp { background: #f8fafc; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
.banner { background: linear-gradient(135deg, #eef4ff 0%, #e6f7f5 100%); border: 1px solid #e2e8f0; border-radius: 18px; padding: 26px 30px; margin-bottom: 20px; }
.banner h1 { color: #1e293b; font-size: 1.9rem; margin: 0 0 6px 0; font-weight: 700; }
.banner p { color: #64748b; margin: 0; font-size: 1.02rem; }
.section-label { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #94a3b8; margin: 16px 0 6px 0; }
[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 12px 16px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04); }
[data-testid="stMetricLabel"] { color: #64748b; }
[data-testid="stMetricValue"] { color: #1e293b; font-weight: 700; }
.stButton > button { background: linear-gradient(135deg, #4f8cff, #22c1c3); color: #ffffff; border: none; border-radius: 12px; padding: 0.55rem 1.4rem; font-weight: 600; font-size: 1.02rem; transition: all 0.2s ease; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(79, 140, 255, 0.35); color: #ffffff; }
.stButton > button:focus { color: #ffffff; }
[data-testid="stFileUploaderDropzone"] { background: #ffffff; border: 2px dashed #c7d6f5; border-radius: 16px; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: #4f8cff; }
.desc-card { background: linear-gradient(135deg, #f0f7ff 0%, #f0fbf8 100%); border: 1px solid #dbeafe; border-radius: 16px; padding: 20px 24px; font-size: 1.04rem; line-height: 1.85; color: #1e293b; }
.empty-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px 20px; color: #64748b; text-align: center; }
[data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 14px; overflow: hidden; }
</style>
"""


def main():
    load_env_file(os.path.join(os.path.dirname(__file__), ".env"))
    st.set_page_config(page_title="YOLO 图片描述", page_icon="🖼️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="banner"><h1>🖼️ YOLO 图片描述</h1>'
        "<p>上传一张图片 · YOLO 识别目标 · DeepSeek 生成自然语言描述</p></div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('<div class="section-label">检测配置</div>', unsafe_allow_html=True)
        model_name = st.selectbox("检测模型", ["yolov8n.pt", "yolo11n.pt"], index=0)
        conf_thres = st.slider("置信度阈值", 0.05, 1.0, 0.25, 0.05)
        st.markdown('<div class="section-label">生成配置</div>', unsafe_allow_html=True)
        temperature = st.slider("temperature", 0.0, 1.5, 0.7, 0.1)
        max_tokens = st.number_input("max_tokens", 50, 1000, 300, 50)
        st.markdown('<div class="section-label">API Key</div>', unsafe_allow_html=True)
        api_key_input = st.text_input(
            "DeepSeek API Key",
            type="password",
            help="留空则读取 .env 文件或环境变量 DEEPSEEK_API_KEY",
        )

    uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        st.markdown('<div class="empty-card">👆 请上传一张图片，开始识别与描述</div>', unsafe_allow_html=True)
        return

    image = Image.open(uploaded)
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    model = get_model(model_name)
    results = model(image_bgr, conf=conf_thres)

    detections = summarize_results(results)
    annotated = draw_detections(image_bgr, results)

    total = sum(d["count"] for d in detections)
    n_classes = len(detections)
    avg_conf = round(sum(d["avg_conf"] for d in detections) / n_classes, 3) if n_classes else 0.0
    m1, m2, m3 = st.columns(3)
    m1.metric("🎯 检测目标总数", total)
    m2.metric("🏷️ 目标类别数", n_classes)
    m3.metric("📊 平均置信度", f"{avg_conf:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**原图**")
            st.image(image_bgr, channels="BGR", width="stretch")
    with col2:
        with st.container(border=True):
            st.markdown("**YOLO 标注**")
            st.image(annotated, channels="BGR", width="stretch")

    st.markdown("#### 识别结果")
    if detections:
        st.dataframe(
            detections,
            column_config={
                "class": st.column_config.TextColumn("类别"),
                "count": st.column_config.NumberColumn("数量"),
                "avg_conf": st.column_config.ProgressColumn("平均置信度", min_value=0.0, max_value=1.0, format="%.3f"),
            },
            hide_index=True,
            width="stretch",
        )
    else:
        st.markdown('<div class="empty-card">未检测到常见目标</div>', unsafe_allow_html=True)

    api_key = (
        api_key_input.strip()
        or os.environ.get("DEEPSEEK_API_KEY", "")
        or st.secrets.get("DEEPSEEK_API_KEY", "")
    )
    if not api_key:
        st.error("未配置 DeepSeek API Key。请在 .env / 环境变量 / Streamlit Secrets 中设置 DEEPSEEK_API_KEY，或在侧边栏输入。")
        return

    if st.button("✨ 生成图片描述", width="stretch"):
        with st.spinner("DeepSeek 正在生成描述..."):
            try:
                prompt = build_prompt(detections)
                description = call_deepseek(prompt, api_key, temperature=temperature, max_tokens=int(max_tokens))
            except Exception as e:
                st.error(f"生成描述失败：{e}")
                return
        st.markdown("#### 图片描述")
        st.markdown(f'<div class="desc-card">{html.escape(description)}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
