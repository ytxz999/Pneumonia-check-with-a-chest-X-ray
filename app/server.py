"""
server.py —— 胸片肺炎智能检测系统（自研 Web 前端版）

技术栈：FastAPI + 原生 HTML/CSS/JS 单页应用
为什么自研前端而不用 Gradio：
  1. 界面完全可控，能做出医疗级精美 UI（玻璃拟态、环形仪表盘、动效）
  2. 功能可扩展：阈值滑块、热力图叠加控制、历史记录等
  3. 答辩时「自己写的前端」比拖拽式 Gradio 更有说服力

运行方式：
    python app/server.py
    浏览器打开 http://127.0.0.1:7860
"""

import io
import sys
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

# Windows 控制台 GBK 兼容
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pth"
CLASSES_CN = ["正常", "肺炎"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------- 加载模型（启动时加载一次） ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES_CN))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

app = FastAPI(title="胸片肺炎智能检测系统")


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/predict")
async def predict(payload: dict):
    """接收 base64 图片 + 阈值 + 叠加透明度，返回预测、热力图、统计信息。"""
    import base64

    img_b64: str = payload.get("image", "")
    threshold: float = float(payload.get("threshold", 0.5))
    overlay_alpha: float = float(payload.get("overlay_alpha", 0.45))
    if not img_b64:
        return JSONResponse({"error": "未收到图片"}, status_code=400)

    # base64 → PIL（去掉 data:image/xxx;base64, 前缀）
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")

    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0].cpu().numpy()
    p_pneu = float(probs[1])

    # 业务规则：肺炎概率 >= 阈值 判为肺炎
    pred_idx = 1 if p_pneu >= threshold else 0

    # Grad-CAM（对「肺炎」类计算，展示模型寻找病灶的区域）
    rgb_np = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
    with GradCAM(model=model, target_layers=[model.layer4]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])[0]
    heatmap = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)

    # 热力图与原图叠加（alpha 控制热力图浓度）
    base = np.array(img.resize((224, 224))).astype(np.float32)
    blended = (base * (1 - overlay_alpha) + heatmap.astype(np.float32) * overlay_alpha).astype(np.uint8)

    hm_b64 = base64.b64encode(png_bytes(Image.fromarray(blended))).decode()
    orig_b64 = base64.b64encode(png_bytes(img.resize((448, 448)))).decode()

    return {
        "pred": pred_idx,
        "pred_cn": CLASSES_CN[pred_idx],
        "p_normal": round(float(probs[0]), 4),
        "p_pneumonia": round(p_pneu, 4),
        "threshold": threshold,
        "heatmap": hm_b64,
        "original": orig_b64,
        "device": str(device),
        "model": "ResNet18 (Transfer Learning)",
    }


def _find_test_image(kind: str) -> Image.Image:
    """从测试集中按类别各取一张固定示例图。"""
    test_dir = ROOT / "data" / "chest_xray" / "test"
    if kind == "pneu":
        path = test_dir / "PNEUMONIA" / "person100_bacteria_475.jpeg"
    else:
        path = test_dir / "NORMAL" / "IM-0003-0001.jpeg"
    if not path.exists():
        # 兜底：取该类别文件夹第一张图
        folder = test_dir / ("PNEUMONIA" if kind == "pneu" else "NORMAL")
        files = sorted(folder.glob("*"))
        if not files:
            raise FileNotFoundError(f"测试集目录下没有图片: {folder}")
        path = files[0]
    return Image.open(path).convert("RGB")


@app.get("/api/sample")
def sample(kind: str = "normal"):
    """返回一张示例胸片的 base64（完整分辨率，用于载入分析）。"""
    img = _find_test_image(kind)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    import base64
    return {"image": base64.b64encode(buf.getvalue()).decode()}


@app.get("/api/samples_preview")
def samples_preview():
    """返回两张示例图的缩略图 base64（用于前端示例按钮）。"""
    import base64
    out = {}
    for kind in ("normal", "pneu"):
        try:
            img = _find_test_image(kind).resize((160, 160))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            out[kind] = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            out[kind] = ""
    return out


if __name__ == "__main__":
    import gradio  # noqa: F401  # 确保依赖仍在 requirements 中

    print("=" * 60)
    print("  胸片肺炎智能检测系统 已启动")
    print("  请用浏览器打开:  http://127.0.0.1:7860")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
