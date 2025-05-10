import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载模型和处理器
# model, vis_processors, txt_processors = load_model_and_preprocess(
#             name="blip2_t5",
#             model_type="pretrain_flant5xl",
#             device=device,
#             is_eval=True
#         )

model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption",
    model_type="base_coco",
    is_eval=True,
    device=device
)

# 输入图像
image = Image.open("./src/image_processor/me.png").convert("RGB")

image = vis_processors["eval"](image).unsqueeze(0).to(device)

caption = model.generate({"image": image})[0]
print("Caption:", caption)
