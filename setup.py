from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"
save_dir = "./qwen2_vl_offline_weights"
LOCAL_ONLY = True

processor = AutoProcessor.from_pretrained(
    MODEL_PATH, backend="torchvision", local_files_only=LOCAL_ONLY,
)

dtype = torch.float16
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype
)
processor.save_pretrained(save_dir)
model.save_pretrained(save_dir)
print("Download complete. You can now use this folder for offline evaluation.")