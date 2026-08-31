# YOLO + DeepSeek 图片描述应用 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 Streamlit 网页应用：上传图片后，用 YOLO 识别目标并调用 DeepSeek 生成一段中文图片描述。

**Architecture:** 单文件 Streamlit 应用 `examples/robinji/image_describer.py`，内部按职责拆分为纯函数（`summarize_results` / `draw_detections` / `build_prompt` / `call_deepseek`）与 UI（`main`）。纯函数可独立单测，UI 只做编排与展示。

**Tech Stack:** Streamlit、Ultralytics YOLO、OpenCV、requests、pytest（测试）

**Spec:** `docs/superpowers/specs/2026-08-31-image-describer-design.md`

## Global Constraints

- 仅新增 `examples/robinji/image_describer.py` 与 `examples/robinji/test_image_describer.py`；不修改 `requirements.txt`。
- API Key 仅从环境变量 `DEEPSEEK_API_KEY` 或侧边栏输入读取，禁止写入代码。
- LLM 请求仅发往 `https://api.deepseek.com`，模型名固定 `deepseek-chat`。
- 检测模型固定使用 `examples/robinji/` 目录内已存在的 `yolov8n.pt` / `yolo11n.pt`。
- 测试运行目录与命令：`cd examples/robinji && pytest test_image_describer.py -v`。
- 本仓库约定：除非用户明确要求，不执行 git commit（计划中的 Commit 步骤默认跳过）。

---

### Task 1: 检测结果工具函数（summarize_results + draw_detections）

**Files:**
- Create: `examples/robinji/image_describer.py`
- Test: `examples/robinji/test_image_describer.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `summarize_results(results) -> list[dict]`，返回 `[{"class": str, "count": int, "avg_conf": float}, ...]`，按 count 降序；`results` 为 ultralytics `Results` 对象列表（`.boxes.cls` / `.boxes.conf` / `.boxes.xyxy` 为 tensor，`.names` 为 `{cls_id: name}` 字典，`boxes` 可能为 `None`）。
  - `draw_detections(image_bgr: np.ndarray, results) -> np.ndarray`，返回画框后的 BGR 图像副本。

- [ ] **Step 1: 创建模块骨架与失败测试**

`examples/robinji/image_describer.py`（先只写 import，函数体在 Step 3 写入，保证 TDD 的 RED 阶段真实可测；`streamlit`/`ultralytics` 惰性导入，避免单测依赖 torch）：

```python
"""YOLO + DeepSeek 图片描述应用。

streamlit 与 ultralytics 采用惰性导入（仅在运行时使用处导入），
保证纯函数单元测试无需安装 torch 即可运行。
"""
import os

import cv2
import numpy as np
import requests
from PIL import Image
```

`examples/robinji/test_image_describer.py`：

```python
"""Tests for image_describer."""
from unittest import mock

import numpy as np

import image_describer as app


class FakeBoxes:
    def __init__(self, cls, conf, xyxy=None):
        self.cls = cls
        self.conf = conf
        self.xyxy = xyxy


class FakeResults:
    def __init__(self, cls=None, conf=None, names=None, xyxy=None):
        self.names = names or {}
        if cls is None:
            self.boxes = None
        else:
            self.boxes = FakeBoxes(cls, conf, xyxy)


def make_detections():
    return [
        {"class": "person", "count": 2, "avg_conf": 0.85},
        {"class": "car", "count": 1, "avg_conf": 0.7},
    ]


def test_summarize_results_counts_and_confidences():
    results = [
        FakeResults(
            cls=np.array([0, 0, 1]),
            conf=np.array([0.9, 0.8, 0.7]),
            names={0: "person", 1: "car"},
        )
    ]
    assert app.summarize_results(results) == make_detections()


def test_summarize_results_empty_boxes():
    results = [FakeResults()]
    assert app.summarize_results(results) == []


def test_draw_detections_returns_annotated_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    results = [
        FakeResults(
            cls=np.array([0]),
            conf=np.array([0.9]),
            names={0: "person"},
            xyxy=np.array([[10, 10, 50, 50]]),
        )
    ]
    annotated = app.draw_detections(img, results)
    assert annotated.shape == img.shape
    assert annotated.any()  # 画了框，像素有变化
```

- [ ] **Step 2: 运行测试确认失败（函数未定义）**

Run: `cd examples/robinji && pytest test_image_describer.py -v`
Expected: FAIL（模块尚未包含函数，ImportError 或 AttributeError）

- [ ] **Step 3: 实现两个函数**

在 `image_describer.py` 的 import 之后追加：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd examples/robinji && pytest test_image_describer.py -v`
Expected: 3 个测试全部 PASS

- [ ] **Step 5: Commit（默认跳过，除非用户明确要求）**

---

### Task 2: build_prompt（提示词构建）

**Files:**
- Modify: `examples/robinji/image_describer.py`（追加 `build_prompt` 函数）
- Modify: `examples/robinji/test_image_describer.py`（追加测试）

**Interfaces:**
- Consumes: `summarize_results` 的输出格式 `list[{"class", "count", "avg_conf"}]`
- Produces: `build_prompt(detections: list[dict]) -> str`，供 `call_deepseek` 作为 user 消息内容。

- [ ] **Step 1: 写失败测试**

在 `test_image_describer.py` 追加：

```python
def test_build_prompt_contains_detections():
    prompt = app.build_prompt(make_detections())
    assert "person" in prompt
    assert "car" in prompt
    assert "0.85" in prompt


def test_build_prompt_empty_detections():
    prompt = app.build_prompt([])
    assert "未检测到常见目标" in prompt
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd examples/robinji && pytest test_image_describer.py::test_build_prompt_contains_detections test_image_describer.py::test_build_prompt_empty_detections -v`
Expected: FAIL（AttributeError: module has no attribute 'build_prompt'）

- [ ] **Step 3: 实现 build_prompt**

在 `image_describer.py` 追加：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd examples/robinji && pytest test_image_describer.py -v`
Expected: 5 个测试全部 PASS

- [ ] **Step 5: Commit（默认跳过，除非用户明确要求）**

---

### Task 3: call_deepseek（DeepSeek API 客户端）

**Files:**
- Modify: `examples/robinji/image_describer.py`（追加 `call_deepseek` 函数）
- Modify: `examples/robinji/test_image_describer.py`（追加测试）

**Interfaces:**
- Consumes: `build_prompt` 的输出（str）；调用方传入 `api_key: str`、`temperature: float`、`max_tokens: int`
- Produces: `call_deepseek(prompt: str, api_key: str, temperature: float = 0.7, max_tokens: int = 300) -> str`；HTTP 非 2xx 时抛出 `RuntimeError`（消息含状态码与响应片段）。

- [ ] **Step 1: 写失败测试**

在 `test_image_describer.py` 追加：

```python
def test_call_deepseek_returns_content():
    fake_resp = mock.Mock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"choices": [{"message": {"content": " 这是一张图片。  "}}]}
    with mock.patch("image_describer.requests.post", return_value=fake_resp) as post:
        text = app.call_deepseek("prompt", "key")
    assert text == "这是一张图片。"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer key"
    assert post.call_args.kwargs["json"]["model"] == "deepseek-chat"


def test_call_deepseek_raises_on_error():
    fake_resp = mock.Mock()
    fake_resp.status_code = 401
    fake_resp.text = "unauthorized"
    with mock.patch("image_describer.requests.post", return_value=fake_resp):
        try:
            app.call_deepseek("prompt", "bad-key")
            assert False, "应当抛出异常"
        except RuntimeError as e:
            assert "401" in str(e)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd examples/robinji && pytest test_image_describer.py::test_call_deepseek_returns_content test_image_describer.py::test_call_deepseek_raises_on_error -v`
Expected: FAIL（AttributeError: module has no attribute 'call_deepseek'）

- [ ] **Step 3: 实现 call_deepseek**

在 `image_describer.py` 追加：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd examples/robinji && pytest test_image_describer.py -v`
Expected: 7 个测试全部 PASS

- [ ] **Step 5: Commit（默认跳过，除非用户明确要求）**

---

### Task 4: Streamlit 界面（main）与冒烟验证

**Files:**
- Modify: `examples/robinji/image_describer.py`（追加 `get_model` 缓存与 `main`）

**Interfaces:**
- Consumes: Task 1-3 的全部函数
- Produces: 可运行的 Streamlit 应用（`streamlit run image_describer.py`）

- [ ] **Step 1: 追加 UI 代码**

在 `image_describer.py` 末尾追加：

```python
@st.cache_resource
def get_model(model_path):
    """加载并缓存 YOLO 模型。"""
    return YOLO(model_path)


def main():
    st.set_page_config(page_title="YOLO 图片描述", page_icon="🖼️")
    st.title("YOLO + DeepSeek 图片描述")
    st.write("上传一张图片，应用会用 YOLO 识别目标，并调用 DeepSeek 生成一段文字描述。")

    with st.sidebar:
        st.header("配置")
        model_name = st.selectbox("检测模型", ["yolov8n.pt", "yolo11n.pt"], index=0)
        conf_thres = st.slider("置信度阈值", 0.05, 1.0, 0.25, 0.05)
        temperature = st.slider("temperature", 0.0, 1.5, 0.7, 0.1)
        max_tokens = st.number_input("max_tokens", 50, 1000, 300, 50)
        api_key_input = st.text_input("DeepSeek API Key（可选，覆盖环境变量）", type="password")

    uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        st.info("请上传一张图片开始。")
        return

    image = Image.open(uploaded)
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    model = get_model(model_name)
    results = model(image_bgr, conf=conf_thres)

    detections = summarize_results(results)
    annotated = draw_detections(image_bgr, results)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原图")
        st.image(image_bgr, channels="BGR")
    with col2:
        st.subheader("YOLO 标注")
        st.image(annotated, channels="BGR")

    st.subheader("识别结果")
    if detections:
        st.dataframe(detections)
    else:
        st.write("未检测到常见目标。")

    api_key = api_key_input.strip() or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        st.error("未配置 DeepSeek API Key。请设置环境变量 DEEPSEEK_API_KEY，或在侧边栏输入。")
        return

    if st.button("生成图片描述"):
        with st.spinner("DeepSeek 正在生成描述..."):
            try:
                prompt = build_prompt(detections)
                description = call_deepseek(prompt, api_key, temperature=temperature, max_tokens=int(max_tokens))
            except Exception as e:
                st.error(f"生成描述失败：{e}")
                return
        st.subheader("图片描述")
        st.markdown(description)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 导入冒烟检查**

Run: `cd examples/robinji && python -c "import image_describer; print('import ok')"`
Expected: 输出 `import ok`

- [ ] **Step 3: 无头启动冒烟检查**

Run:
```bash
cd examples/robinji && streamlit run image_describer.py --server.headless true --server.port 8501 &
sleep 8
curl -s http://localhost:8501/healthz
kill %1
```
Expected: curl 返回 `ok`，应用可启动。

- [ ] **Step 4: 人工验证清单（交给用户执行）**

在浏览器打开 `http://localhost:8501`，用仓库根目录 `tests/` 下的 jpg 图片验证：
1. 上传后原图与 YOLO 标注图并排显示。
2. 识别结果表格正确显示类别/数量/置信度。
3. 未设置 API Key 时显示错误提示。
4. 设置 `DEEPSEEK_API_KEY`（或侧边栏输入）后点击"生成图片描述"，得到一段中文描述。

- [ ] **Step 5: Commit（默认跳过，除非用户明确要求）**

---

## Self-Review

- **Spec coverage:** 上传（Task 4）、YOLO 检测与标注（Task 1/4）、识别表格（Task 4）、DeepSeek 调用（Task 3）、描述展示（Task 4）、侧边栏配置（Task 4）、错误处理——无 Key/无目标/API 错误/非图片格式（Task 3/4）、安全（env-only Key，Task 4）、验证（Task 4）。全部覆盖。
- **Placeholder scan:** 无 TBD/TODO，所有步骤含完整代码。
- **Type consistency:** `summarize_results` 输出 `{"class","count","avg_conf"}` 与 `build_prompt` 消费一致；`call_deepseek(prompt, api_key, temperature, max_tokens)` 签名在 Task 3 定义、Task 4 调用一致；测试中的 mock 结构（`.boxes.cls/.conf/.xyxy/.names`）与实现访问路径一致。
