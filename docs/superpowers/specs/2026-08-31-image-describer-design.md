# 图片描述应用（YOLO + DeepSeek）设计文档

- 日期：2026-08-31
- 状态：已批准
- 作者：robinji（与 AI 协作）

## 目标

构建一个 Streamlit 网页应用：用户上传一张图片，应用使用 YOLO（Ultralytics）识别图中的目标，并将识别结果发送给 DeepSeek 大模型，生成一段自然流畅的中文图片描述。

## 需求

- 用户通过浏览器上传图片（支持 jpg / png 等常见格式）。
- 应用用 YOLO 模型检测目标，并在图片副本上绘制检测框与标签。
- 展示识别结果（类别、数量、平均置信度）。
- 调用 DeepSeek（文本模型 `deepseek-chat`）生成一段自然语言描述。
- 界面同时展示：原图、标注图、识别结果表格、文字描述。

## 技术选型

| 项 | 选择 | 理由 |
|---|---|---|
| UI 框架 | Streamlit | 与用户已有项目一致，开发快 |
| 检测模型 | Ultralytics YOLO（`yolov8n.pt` / `yolo11n.pt`） | 模型文件已在 `examples/robinji/` 本地存在，无需下载 |
| 大模型 | DeepSeek `deepseek-chat`（OpenAI 兼容接口） | 用户选择，国内可直连、便宜 |
| LLM 调用方式 | `requests` POST 到 `https://api.deepseek.com/chat/completions` | 无新增依赖（requests 已是 ultralytics 依赖） |

## 架构

单文件应用 `examples/robinji/image_describer.py`，内部拆分为职责单一的函数：

```
image_describer.py
├── run_detection(model, image) -> (annotated_img, detections)
│     运行 YOLO，绘制标注图，返回检测列表（class, count, avg_conf）
├── build_prompt(detections) -> str
│     将检测结果拼成 DeepSeek 提示词
├── call_deepseek(prompt, api_key, temperature, max_tokens) -> str
│     调用 DeepSeek API，返回生成的描述
└── main()
     Streamlit 界面：上传 → 检测 → 表格 → 描述
```

## 数据流

1. `st.file_uploader` 接收图片 → PIL → OpenCV BGR。
2. `YOLO(model_path)` 检测（model 用 `@st.cache_resource` 缓存加载）。
3. `run_detection` 在图片副本上画框 + 标签 + 置信度，并汇总 `{class: 名称, count: 数量, avg_conf: 平均置信度}`。
4. `build_prompt` 生成提示词（见下）。
5. `call_deepseek` POST 请求，解析 `choices[0].message.content`。
6. 界面输出：`st.image` 原图与标注图并排（`st.columns`）、`st.dataframe` 识别表格、`st.write`/`st.markdown` 描述。

## 提示词设计

系统提示词示例：

> 你是一个专业的图片描述助手。下面是 YOLO 目标检测模型对一张图片的识别结果（类别、数量、平均置信度）：{detections}。请用一段自然、流畅的中文描述这张图片的内容，包括画面中的主体、数量和场景氛围，不要生硬罗列数据，也不要编造检测结果之外的具体细节。

## 侧边栏配置

- 模型选择：`yolov8n.pt` / `yolo11n.pt`（默认 `yolov8n.pt`）
- 置信度阈值：slider，默认 0.25
- temperature：slider，默认 0.7
- max_tokens：number_input，默认 300
- API Key：password 输入框（可选，覆盖环境变量）

## 错误处理

- 未设置 API Key（`.env` 文件 / 环境变量 `DEEPSEEK_API_KEY` 均为空且未在侧边栏输入）：`st.error` 提示设置方式，不发起请求。
- 未检测到任何目标：界面提示"未检测到常见目标"，仍可调用 LLM 生成兜底描述。
- DeepSeek API 返回错误（HTTP 非 2xx / 网络异常）：`st.error` 显示友好错误信息与状态码。
- 非图片文件上传：`st.error` 提示格式要求。

## 安全

- API Key 从 `.env` 文件（`examples/robinji/.env`，已被 .gitignore 忽略）或环境变量 `DEEPSEEK_API_KEY` 读取，或用户在侧边栏临时输入（secrets 不写入代码、不提交仓库）。`.env.example` 提供模板。
- 仅请求固定外部域名 `https://api.deepseek.com`，无 SSRF 风险。

## 验证

- 用 `tests/` 目录下现有 jpg 测试图片人工验证：上传后能显示标注图、识别表格和 LLM 描述。
- 无 Key 场景验证错误提示。
- 运行方式：`streamlit run image_describer.py`（需在 `examples/robinji/` 目录下）。

## 涉及文件

- 新增：`examples/robinji/image_describer.py`
- 不改动：`examples/robinji/requirements.txt`（streamlit / opencv-python / ultralytics 均已覆盖）

## 明确不做（Out of scope）

- LLM 流式输出（打字机效果）——后续可加。
- 视频 / 摄像头实时描述。
- 检测结果缓存（`st.cache_data`）——当前规模收益不大，后续可加。
- 部署到云端。
