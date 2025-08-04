import os
from pathlib import Path
from optimum.exporters.onnx import main_export

# 环境设置，避免导出 fused attention 报错
os.environ["PYTORCH_ENABLE_SDPA"] = "0"
os.environ["TORCH_FORCE_FALLBACK_SDPA"] = "1"

model_dir = Path("/deltadisk/guestnju/tpu/Qwen3-Embedding-0.6B")
output_dir = Path("./onnx_output")
output_dir.mkdir(parents=True, exist_ok=True)

export_args = {
    "model_name_or_path": str(model_dir),
    "task": "default",
    "opset": 14,  # 注意切回较稳定的 opset
    "framework": "pt",
    "output": str(output_dir),
    "cache_dir": str(model_dir / "cache"),
    "torch_dtype": "float32",
}

main_export(**export_args)