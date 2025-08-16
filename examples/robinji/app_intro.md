# 🚀 动手实践：用 YOLOv8 和 Streamlit 打造行人检测报警系统

大家好！今天我想分享一个超酷的小项目——**行人检测报警系统**。这个项目结合了计算机视觉和实时音频反馈，非常适合初学者入门 AI 和 Python 开发。跟着我一步步来，你也可以轻松实现！

## 🌟 项目简介

这个应用使用 **YOLOv8**（一个强大的目标检测模型）和 **Streamlit**（一个快速构建数据应用的框架）来检测摄像头画面中的行人。当检测到行人时，系统会发出持续的报警音（`beep.mp3`），直到你手动确认停止。

## 🛠️ 技术栈

- **YOLOv8**：用于实时目标检测。
- **Streamlit**：快速构建交互式 Web 应用。
- **OpenCV**：处理摄像头画面。
- **Pygame**：播放报警音。

## 🚀 如何运行

1. **克隆代码**：确保你有 `app.py`、`yolov8n.pt`（YOLOv8 模型文件）和 `beep.mp3`（报警音文件）。
2. **安装依赖**：
   ```bash
   pip install streamlit opencv-python ultralytics pygame
   ```
3. **启动应用**：
   ```bash
   streamlit run app.py
   ```
4. **体验功能**：
   - 打开摄像头后，系统会自动检测行人。
   - 检测到行人时，会播放报警音。
   - 点击 **确认并停止检测** 按钮可以停止报警。

## 💡 项目亮点

- **实时检测**：YOLOv8 的高效检测能力。
- **交互式界面**：Streamlit 让开发变得简单。
- **音频反馈**：增强用户体验。

## 📌 适合人群

- 对 AI 和计算机视觉感兴趣的初学者。
- 想快速上手 Streamlit 的开发者。
- 喜欢动手实践的编程爱好者。

## 🔗 代码仓库

你可以在这里找到完整的代码和资源文件：[GitHub 链接]（待补充）

## 📝 代码解析

以下是 `app.py` 的核心代码逻辑解析：

1. **初始化音频和模型**：
   ```python
   pygame.mixer.init()
   model = YOLO("yolov8n.pt")
   ```
   - 使用 `pygame` 初始化音频系统，加载报警音文件 `beep.mp3`。
   - 加载 YOLOv8 模型 `yolov8n.pt`。

2. **Streamlit 界面**：
   ```python
   st.title("行人检测报警系统")
   st.write("当摄像头检测到行人时，会发出持续的报警音。")
   ```
   - 设置标题和描述，提供用户交互按钮。

3. **摄像头和检测循环**：
   ```python
   cap = cv2.VideoCapture(0)
   while True:
       ret, frame = cap.read()
       results = model(frame)
   ```
   - 打开摄像头并实时读取画面。
   - 使用 YOLOv8 检测行人，绘制边界框。

4. **报警逻辑**：
   ```python
   if detected and not confirmed:
       pygame.mixer.music.play(-1)
   else:
       pygame.mixer.music.stop()
   ```
   - 检测到行人时循环播放报警音，用户确认后停止。

5. **资源释放**：
   ```python
   cap.release()
   cv2.destroyAllWindows()
   ```
   - 关闭摄像头和窗口，释放资源。

快来试试吧！如果有任何问题，欢迎留言讨论～ 🎉

---

**#AI #计算机视觉 #Python #Streamlit #YOLOv8 #开源项目**