"""
app.py —— Gradio 网页 Demo（第四步运行）

做什么：
启动一个本地网页：上传胸片 → 显示预测结果（正常/肺炎 + 置信度条）+ Grad-CAM 热力图。

这是整个项目的「产品形态」——课程里大部分人只交 notebook，
你能现场演示一个能点的网页，这就是加分点。

运行方式：
    python app/app.py
然后浏览器打开终端里显示的地址（一般是 http://127.0.0.1:7860）
"""

import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

# 让 app 目录下的脚本能 import 到 src 里的公共函数
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pth"
CLASSES = ["NORMAL", "PNEUMONIA"]
CLASSES_CN = ["正常", "肺炎"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------- 加载模型（启动时加载一次） ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def predict(img: Image.Image):
    """
    核心回调函数：输入上传的图片，输出预测标签、热力图和指标卡。
    Gradio 会自动把它变成网页交互。
    """
    if img is None:
        return {}, None, ""
    img = img.convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # 预测概率
    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0].cpu().numpy()
    label_probs = {CLASSES_CN[i]: float(probs[i]) for i in range(len(CLASSES))}

    # Grad-CAM 热力图
    pred = int(probs.argmax())
    rgb_np = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
    with GradCAM(model=model, target_layers=[model.layer4]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred)])[0]
    heatmap = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)

    # 顶部状态横幅：绿色=正常 / 红色=肺炎
    if pred == 1:
        verdict = (
            f"<div style='background:linear-gradient(90deg,#dc2626,#ef4444);"
            f"color:#fff;padding:14px 20px;border-radius:12px;font-size:19px;"
            f"font-weight:700;letter-spacing:1px;'>"
            f"🔴 检测到肺炎征象（置信度 {probs[1]*100:.1f}%）"
            f"<span style='font-size:13px;font-weight:400;opacity:.85;margin-left:12px;'>"
            f"建议进一步就医复查</span></div>"
        )
    else:
        verdict = (
            f"<div style='background:linear-gradient(90deg,#16a34a,#22c55e);"
            f"color:#fff;padding:14px 20px;border-radius:12px;font-size:19px;"
            f"font-weight:700;letter-spacing:1px;'>"
            f"🟢 未见明显肺炎征象（置信度 {probs[0]*100:.1f}%）"
            f"<span style='font-size:13px;font-weight:400;opacity:.85;margin-left:12px;'>"
            f"仅供课程演示参考</span></div>"
        )

    return label_probs, Image.fromarray(heatmap), verdict


CUSTOM_CSS = """
/* 页面整体渐变背景 + 字体 */
.gradio-container {
    background: linear-gradient(160deg, #f0f7ff 0%, #e8f0fe 45%, #f5f0ff 100%) !important;
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif !important;
}
/* 主标题卡片 */
#header {
    background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 55%, #3b82f6 100%) !important;
    border-radius: 16px !important;
    padding: 26px 30px !important;
    color: #fff !important;
    border: none !important;
}
#header h1 { color: #fff !important; margin: 0 0 6px 0 !important; font-size: 27px !important; }
#header p  { color: rgba(255,255,255,.88) !important; margin: 0 !important; font-size: 14.5px !important; }
/* 页脚提示条 */
#footer {
    background: #fff7ed !important;
    border: 1px solid #fed7aa !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-size: 13px !important;
    color: #9a3412 !important;
}
/* 各面板圆角白卡 */
.block, .form {
    border-radius: 14px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 2px 10px rgba(30, 58, 138, .06) !important;
    background: #ffffff !important;
}
/* 分析按钮 */
#analyze-btn {
    font-size: 17px !important;
    font-weight: 700 !important;
    padding: 12px 0 !important;
    border-radius: 10px !important;
    background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
    border: none !important;
    letter-spacing: 4px !important;
}
#analyze-btn:hover { filter: brightness(1.08) !important; }
"""

# ---------- 网页界面 ----------
with gr.Blocks(title="胸片肺炎智能检测系统") as demo:
    gr.HTML(
        """
        <div id="header">
            <h1>🩻 胸片肺炎智能检测系统</h1>
            <p>Deep Learning · ResNet18 迁移学习 &nbsp;|&nbsp; Grad-CAM 可解释性分析 &nbsp;|&nbsp;
            《医学人工智能导论》课程项目</p>
        </div>
        """
    )

    verdict_html = gr.HTML("", label="verdict")  # 顶部状态横幅（占位，点击后填充）

    with gr.Row():
        with gr.Column(scale=5):
            input_img = gr.Image(type="pil", label="上传胸部 X 光片", height=420,
                                 sources=["upload", "clipboard"])
            btn = gr.Button("开始分析", variant="primary", elem_id="analyze-btn")
        with gr.Column(scale=6):
            with gr.Tab("预测结果"):
                label_out = gr.Label(num_top_classes=2, label="分类置信度")
            with gr.Tab("Grad-CAM 热力图"):
                heat_out = gr.Image(label="模型关注区域（亮处 = 决策依据）", height=420)

    gr.HTML(
        """
        <div id="footer">
            ⚠️ <b>免责声明</b>：本系统为课程教学演示项目，模型基于公开数据集（Kaggle Chest X-Ray Images）
            训练，<b>不能用于真实医疗诊断</b>。任何健康判断请以专业医师意见为准。
        </div>
        """
    )

    # 交互绑定：点按钮或上传图片都触发分析
    btn.click(predict, inputs=input_img, outputs=[label_out, heat_out, verdict_html])
    input_img.change(predict, inputs=input_img, outputs=[label_out, heat_out, verdict_html])


if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)  # 本地运行；如果想局域网可访问改成 demo.launch(server_name="0.0.0.0")
