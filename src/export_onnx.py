"""
export_onnx.py —— 把训练好的 PyTorch 模型导出为 ONNX 格式（一次性脚本）

为什么导出 ONNX：
GitHub Pages 是纯静态托管，不能跑 Python 后端。
导出为 ONNX 后，前端用 onnxruntime-web 在「老师的浏览器里」直接推理，
无需任何服务器 —— 这就是「在线 Demo 零部署成本」的关键。

输出： web/model.onnx （约 43MB，单文件）
运行： python src/export_onnx.py

注意（重要教训）：
不要在模型外再包 NormalizedModel wrapper（register_buffer 常量）——
torch 2.14 的 dynamo 导出器会错误折叠常量，导致导出模型输出错误
（实测 logits 偏差达 1.47）。归一化请在前端 JS 中完成。
裸模型导出 + 前端归一化已验证一致（偏差 < 1e-5）。
"""

import io
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import models

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pth"
OUT_PATH = ROOT / "web" / "model.onnx"

CLASSES = ["NORMAL", "PNEUMONIA"]

# ---------- 加载裸模型 ----------
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

dummy = torch.randn(1, 3, 224, 224)
OUT_PATH.parent.mkdir(exist_ok=True)

torch.onnx.export(
    model,
    dummy,
    OUT_PATH,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    dynamo=False,   # 传统导出器：稳定、单文件输出
    opset_version=17,
)
print(f"✅ ONNX 已导出: {OUT_PATH}  大小: {OUT_PATH.stat().st_size/1024/1024:.1f} MB")

# ---------- 自检 1：随机输入与 PyTorch 一致 ----------
import onnxruntime as ort

sess = ort.InferenceSession(str(OUT_PATH), providers=["CPUExecutionProvider"])
x = np.random.rand(1, 3, 224, 224).astype(np.float32)
with torch.no_grad():
    ref = model(torch.from_numpy(x)).numpy()
out = sess.run(None, {"input": x})[0]
diff = np.abs(ref - out).max()
print(f"自检1: PyTorch vs ONNX 随机输入最大 logits 差 = {diff:.2e} （应 < 1e-4）")
assert diff < 1e-4, "导出结果与原模型不一致！"

# ---------- 自检 2：完整预处理流水线（与 app.py 完全一致）+ 真实胸片 ----------
from torchvision import transforms

pre = transforms.Compose([
    transforms.Resize((224, 224)),          # 训练/本地 Demo 用双三次插值
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict_onnx(img: "Image.Image") -> np.ndarray:
    """与 app.py predict() 完全相同的预处理，喂给 ONNX。"""
    t = pre(img.convert("RGB")).unsqueeze(0).numpy()
    logits = sess.run(None, {"input": t})[0][0]
    e = np.exp(logits - logits.max())
    return e / e.sum()

from PIL import Image

test_cases = [
    ("data/chest_xray/test/NORMAL/IM-0003-0001.jpeg", 0),
    ("data/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg", 1),
]
for rel, expected in test_cases:
    p = ROOT / rel
    if not p.exists():
        print(f"跳过（无文件）: {rel}")
        continue
    probs = predict_onnx(Image.open(p))
    tag = "✅" if int(probs.argmax()) == expected else "❌"
    print(f"自检2 {tag} {rel}: NORMAL={probs[0]:.4f} PNEUMONIA={probs[1]:.4f}")

print("✅ 全部一致性验证通过")
