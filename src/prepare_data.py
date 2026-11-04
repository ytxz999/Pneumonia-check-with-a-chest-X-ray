"""
prepare_data.py —— 数据准备脚本（第一步运行）

作用：
1. 检查 Kaggle 胸片数据集是否已放到 data/ 目录下
2. 原始验证集只有 16 张图（太小，评估不可靠），
   所以从训练集中重新切出 10% 作为新的验证集

数据集下载方法：
1. 注册 Kaggle 账号 https://www.kaggle.com
2. 打开 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
3. 点击 Download 下载 zip（约 1.2 GB）
4. 解压后把里面的 chest_xray 文件夹放到  data/  目录下
   最终结构应该是：
   data/chest_xray/train/NORMAL/*.jpeg
   data/chest_xray/train/PNEUMONIA/*.jpeg
   data/chest_xray/test/...  val/...

运行方式：
    python src/prepare_data.py
"""

import random
import shutil
import sys
import io
from pathlib import Path

# Windows 控制台默认 GBK 编码，强制改成 UTF-8 以支持中文和 emoji 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ============ 配置区（按需修改） ============
DATA_DIR = Path(__file__).parent.parent / "data" / "chest_xray"
# 新验证集占原训练集的比例
VAL_RATIO = 0.1
# 随机种子：保证每次运行切分结果一致（写报告时要说明，保证可复现）
SEED = 42
# ===========================================


def check_dataset() -> bool:
    """检查数据集目录结构是否完整"""
    required = [
        DATA_DIR / "train" / "NORMAL",
        DATA_DIR / "train" / "PNEUMONIA",
        DATA_DIR / "test" / "NORMAL",
        DATA_DIR / "test" / "PNEUMONIA",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("❌ 数据集不完整，缺少以下目录：")
        for p in missing:
            print(f"   {p}")
        print("\n请按脚本开头的说明下载并解压 Kaggle 数据集到 data/ 目录。")
        return False

    # 打印各类别图片数量
    print("✅ 数据集检查通过，各类别图片数量：")
    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            d = DATA_DIR / split / cls
            n = len(list(d.glob("*.jpeg"))) + len(list(d.glob("*.jpg"))) if d.exists() else 0
            print(f"   {split}/{cls}: {n} 张")
    return True


def resplit_validation():
    """
    重新切分验证集：
    - 原始 val/ 只有 16 张，统计上没有意义
    - 从 train/ 每个类别随机抽 VAL_RATIO 比例的图片，移到 val/ 下
    - 先备份原 val 到 val_original_16/
    """
    val_dir = DATA_DIR / "val"
    backup_dir = DATA_DIR / "val_original_16"

    # 已经切分过就跳过（可重复运行）
    if backup_dir.exists():
        print("ℹ️ 检测到已切分过验证集，跳过（如需重新切分请删除 val_original_16 文件夹）")
        return

    # 备份原始 val
    shutil.move(str(val_dir), str(backup_dir))
    val_dir.mkdir()

    random.seed(SEED)
    total_moved = 0
    for cls in ["NORMAL", "PNEUMONIA"]:
        train_cls_dir = DATA_DIR / "train" / cls
        images = sorted(train_cls_dir.glob("*.jpeg")) + sorted(train_cls_dir.glob("*.jpg"))
        n_val = int(len(images) * VAL_RATIO)
        # random.sample 保证无重复抽取
        val_images = random.sample(images, n_val)

        (val_dir / cls).mkdir(parents=True, exist_ok=True)
        for img in val_images:
            shutil.move(str(img), str(val_dir / cls / img.name))
        total_moved += n_val
        print(f"   {cls}: 移动 {n_val} 张到新验证集")

    print(f"✅ 验证集重新切分完成，共 {total_moved} 张（随机种子={SEED}，可复现）")


if __name__ == "__main__":
    print("=" * 50)
    print("步骤 1/4：数据准备")
    print("=" * 50)

    if not check_dataset():
        exit(1)  # 数据没准备好，退出

    resplit_validation()
    print("\n数据准备完成！下一步运行:  python src/train.py")
