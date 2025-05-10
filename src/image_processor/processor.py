from typing import List, Dict, Union
import torch
from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
import clip
from lavis.models import load_model_and_preprocess

class ImageProcessor:
    """图像处理器，支持BLIP2和CLIP模型"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['image_processor']['device'])
        self.model_name = config['image_processor']['model']
        self.batch_size = config['image_processor']['batch_size']
        self.image_size = config['image_processor']['image_size']
        
        # 初始化模型
        if self.model_name == "blip2":
            self._init_blip2()
        elif self.model_name == "clip":
            self._init_clip()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _init_blip2(self):
        """初始化BLIP2模型"""
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="blip2_t5",
            model_type="pretrain_flant5xl",
            device=self.device,
            is_eval=True
        )
    
    def _init_clip(self):
        """初始化CLIP模型"""
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
    
    def process_image(self, image: Image.Image) -> Dict:
        """处理单张图片"""
        if self.model_name == "blip2":
            return self._process_blip2(image)
        else:
            return self._process_clip(image)
    
    def process_batch(self, images: List[Image.Image]) -> List[Dict]:
        """批量处理图片"""
        results = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            if self.model_name == "blip2":
                results.extend(self._process_batch_blip2(batch))
            else:
                results.extend(self._process_batch_clip(batch))
        return results
    
    def _process_blip2(self, image: Image.Image) -> Dict:
        """使用BLIP2处理图片"""
        # 预处理图片
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        
        # 生成图片描述
        with torch.no_grad():
            caption = self.model.generate({"image": image})[0]
        
        # 获取图片特征
        with torch.no_grad():
            image_features = self.model.extract_features({"image": image})
        
        return {
            "caption": caption,
            "features": image_features.image_embeds.cpu().numpy()
        }
    
    def _process_clip(self, image: Image.Image) -> Dict:
        """使用CLIP处理图片"""
        # 预处理图片
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # 获取图片特征
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        
        return {
            "features": image_features.cpu().numpy()
        }
    
    def _process_batch_blip2(self, images: List[Image.Image]) -> List[Dict]:
        """批量使用BLIP2处理图片"""
        # 预处理图片
        processed_images = torch.stack([
            self.vis_processors["eval"](img) for img in images
        ]).to(self.device)
        
        # 生成图片描述
        with torch.no_grad():
            captions = self.model.generate({"image": processed_images})
        
        # 获取图片特征
        with torch.no_grad():
            image_features = self.model.extract_features({"image": processed_images})
        
        return [
            {
                "caption": caption,
                "features": features.cpu().numpy()
            }
            for caption, features in zip(captions, image_features.image_embeds)
        ]
    
    def _process_batch_clip(self, images: List[Image.Image]) -> List[Dict]:
        """批量使用CLIP处理图片"""
        # 预处理图片
        processed_images = torch.stack([
            self.preprocess(img) for img in images
        ]).to(self.device)
        
        # 获取图片特征
        with torch.no_grad():
            image_features = self.model.encode_image(processed_images)
        
        return [
            {"features": features.cpu().numpy()}
            for features in image_features
        ] 