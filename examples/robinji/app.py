import os
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pygame

# Initialize pygame for audio
pygame.mixer.init()
audio_path = "beep.mp3"

# Check if file exists and is not empty
if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
    st.error("音频文件无效或为空，请提供有效的 beep.mp3 文件。")
else:
    try:
        pygame.mixer.music.load(audio_path)
    except Exception as e:
        st.error(f"加载音频文件失败: {e}")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Streamlit app
def main():
    st.title("行人检测报警系统")
    st.write("当摄像头检测到行人时，会发出持续的报警音。")

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("无法打开摄像头！")
        return

    # Placeholder for the video frame
    frame_placeholder = st.empty()
    
    # 添加确认按钮
    confirmed = False
    if st.button("确认并停止检测"):
        confirmed = True
        pygame.mixer.music.stop()
        st.success("已确认，停止检测和报警")

    # Detection loop
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("无法读取摄像头画面！")
            break
            
        # Perform detection
        results = model(frame)
        detected = False

        # Draw bounding boxes and check for pedestrians
        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0:  # Class 0 is 'person' in YOLO
                    detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Play alarm if pedestrian detected and not confirmed
        if detected and not confirmed:
            try:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)  # Loop the beep sound
            except:
                pass  # Ignore errors in playing sound
        else:
            pygame.mixer.music.stop()

        # Display the frame
        frame_placeholder.image(frame, channels="BGR")

        # Break the loop if 'q' is pressed (in non-streamlit context)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()