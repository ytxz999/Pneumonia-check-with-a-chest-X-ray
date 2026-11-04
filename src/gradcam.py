"""
gradcam.py —— Grad-CAM 可解释性可视化（第三步运行）

做什么：
Grad-CAM（Gradient-weighted Class Activation Mapping，梯度加权类激活映射）
能回答「模型看这张胸片的哪个区域得出的结论」。

原理（一句话版）：
最后一个卷积层的特征图 × 该类别的梯度权重 → 加权求和 → 上采样叠回原图。
亮的地方 = 模型决策时最关注的区域。

对胸片来说，理想情况是热力图亮在肺部病灶处；
如果亮在图片角落或文字区域，说明模型学到了捷径（数据泄漏的信号）——
这也是报告中值得写的一个分析点。

输出：
results/gradcam_*.png —— 每张测试图的 [原图 | 热力图叠加] 对比

运行方式：
    python src/gradcam.py                # 默认可视化 6 张测试图
    python src/gradcam.py --num 10       # 可视化 10 张
"""

import argparse
import io
import sys
from pathlib import Path

# Windows 控制台默认 GBK 编码，强制改成 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

# ============ 配置区 ============
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "chest_xray" / "test"
MODEL_PATH = ROOT / "models" / "best_model.pth"
RESULTS_DIR = ROOT / "results"
# ResNet18 的最后一个卷积层（Grad-CAM 的标准目标层）
TARGET_LAYER = ["layer4"]
CLASSES = ["NORMAL", "PNEUMONIA"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# ================================


def load_model(device):
    """加载训练好的模型权重"""
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval().to(device)
    return model


def preprocess(img: Image.Image) -> torch.Tensor:
    """单张图预处理，和训练时保持一致"""
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(img).unsqueeze(0)  # [C,H,W] -> [1,C,H,W]，凑成 batch 维度


def main(num_images: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    # 从每个类别各取一半图（保证两种情况都看到）
    normal_imgs = sorted((DATA_DIR / "NORMAL").glob("*.jpeg"))[: num_images // 2]
    pneumo_imgs = sorted((DATA_DIR / "PNEUMONIA").glob("*.jpeg"))[: num_images - num_images // 2]
    samples = [(p, "NORMAL") for p in normal_imgs] + [(p, "PNEUMONIA") for p in pneumo_imgs]

    RESULTS_DIR.mkdir(exist_ok=True)

    with GradCAM(model=model, target_layers=[getattr(model, TARGET_LAYER[0])]) as cam:
        for img_path, true_label in samples:
            img = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(img).to(device)

            # 预测
            with torch.no_grad():
                probs = torch.softmax(model(input_tensor), dim=1)[0]
            pred = int(probs.argmax())

            # 生成热力图（针对预测出的那个类别）
            rgb_img = transforms.functional.resize(img, (224, 224))
            rgb_img = transforms.functional.to_pil_image(
                transforms.functional.pil_to_tensor(rgb_img).float() / 255.0
            )
            rgb_np = np_img(rgb_img)
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred)])[0]
            visualization = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)

            # 拼图：左原图 | 右热力图
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(rgb_np); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(visualization)
            axes[1].set_title(f"True: {true_label} | Pred: {CLASSES[pred]} ({probs[pred]*100:.1f}%)")
            axes[1].axis("off")
            plt.tight_layout()
            out = RESULTS_DIR / f"gradcam_{img_path.stem[:30]}.png"
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"✅ {img_path.name}  真实={true_label}  预测={CLASSES[pred]}  置信度={probs[pred]*100:.1f}%  → {out.name}")

    print(f"\nGrad-CAM 可视化完成！下一步运行:  python app/app.py")


def np_img(pil_img: Image.Image):
    """PIL 图转 numpy 数组（0~1 浮点），show_cam_on_image 需要"""
    import numpy as np
    return np.array(pil_img).astype(np.float32) / 255.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=6, help="可视化几张测试图")
    args = parser.parse_args()
    main(args.num)
