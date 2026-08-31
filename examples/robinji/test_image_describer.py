"""Tests for image_describer."""
import os
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


def test_build_prompt_contains_detections():
    prompt = app.build_prompt(make_detections())
    assert "person" in prompt
    assert "car" in prompt
    assert "0.85" in prompt


def test_build_prompt_empty_detections():
    prompt = app.build_prompt([])
    assert "未检测到常见目标" in prompt


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


def test_load_env_file_sets_variables(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# 注释行\nDEEPSEEK_API_KEY=sk-test-123\n\n", encoding="utf-8")
    app.load_env_file(str(env_file))
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-test-123"


def test_load_env_file_does_not_override_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "existing")
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=from-file\n", encoding="utf-8")
    app.load_env_file(str(env_file))
    assert os.environ["DEEPSEEK_API_KEY"] == "existing"


def test_load_env_file_missing_file_is_noop(tmp_path):
    app.load_env_file(str(tmp_path / "nonexistent.env"))  # 不应抛异常
