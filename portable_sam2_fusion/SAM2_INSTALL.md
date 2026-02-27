# SAM2 权重下载指南

## 方法1：使用HuggingFace下载（推荐）

```bash
# 安装huggingface_hub
pip install huggingface_hub

# 使用Python脚本下载
python << 'EOF'
from huggingface_hub import hf_hub_download

# 下载 sam2.1_hiera_base_plus
checkpoint_path = hf_hub_download(
    repo_id="facebook/sam2.1-hiera-base-plus",
    filename="sam2.1_hiera_base_plus.pt",
    local_dir="."
)
print(f"Downloaded to: {checkpoint_path}")
EOF
```

## 方法2：使用国内镜像

```bash
# 设置环境变量使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 然后使用方法1下载
```

## 方法3：使用cURL（带User-Agent）

```bash
cd checkpoints
curl -L -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
  -o sam2.1_hiera_base_plus.pt \
  "https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt"
```

## 方法4：从其他镜像站

```bash
# 使用ModelScope（国内）
pip install modelscope
python << 'EOF'
from modelscope.hub.snapshot_download import snapshot_download
cache_dir = snapshot_download('facebook/sam2.1-hiera-base-plus', cache_dir='./model_cache')
print(f"Downloaded to: {cache_dir}")
EOF
```

## SAM2.1 可用模型列表

| 模型 | 大小 | 推荐场景 |
|------|------|---------|
| sam2.1_hiera_tiny.pt | 38.9M | 资源受限环境 |
| sam2.1_hiera_small.pt | 46M | 平衡性能与速度 |
| sam2.1_hiera_base_plus.pt | 80.8M | **推荐用于小数据集** |
| sam2.1_hiera_large.pt | 224.4M | 追求最高精度 |

## 安装SAM2代码

下载完权重后，还需要安装SAM2代码：

```bash
# 克隆并安装
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .

# 如果遇到CUDA编译错误，可以忽略警告继续
```
