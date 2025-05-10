from typing import List, Dict, Union, Optional
import numpy as np
import faiss
import json
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

class VectorStore:
    """向量存储类，使用FAISS进行向量索引和检索"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dimension = config['vector_store']['dimension']
        self.index_type = config['vector_store']['index_type']
        self.nlist = config['vector_store']['nlist']
        self.nprobe = config['vector_store']['nprobe']
        
        # 初始化向量索引
        self._init_index()
        
        # 初始化文本编码器
        self._init_encoder()
        
        # 存储文档元数据
        self.metadata = []
    
    def _init_index(self):
        """初始化FAISS索引"""
        if self.index_type == "L2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IP":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # 如果使用IVF索引
        if self.nlist > 0:
            self.index = faiss.IndexIVFFlat(
                self.index, self.dimension, self.nlist, faiss.METRIC_L2
            )
            self.index.nprobe = self.nprobe
    
    def _init_encoder(self):
        """初始化文本编码器"""
        # 使用BGE模型进行文本编码
        self.encoder = SentenceTransformer('BAAI/bge-large-zh')
    
    def add_documents(self, documents: List[Dict]):
        """添加文档到向量存储"""
        # 提取文本和元数据
        texts = []
        for doc in documents:
            # 组合文本内容
            text = ""
            if "text" in doc:
                text += doc["text"]
            if "caption" in doc:
                text += f" [图片描述] {doc['caption']}"
            if "table" in doc:
                text += f" [表格内容] {json.dumps(doc['table'], ensure_ascii=False)}"
            
            texts.append(text)
            self.metadata.append({
                "source": doc.get("source", ""),
                "type": doc.get("type", "text"),
                "page": doc.get("page", 0)
            })
        
        # 编码文本
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        
        # 添加到索引
        self.index.add(embeddings.astype(np.float32))
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相似文档"""
        # 编码查询
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        
        # 搜索最相似的文档
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), top_k
        )
        
        # 准备结果
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # 有效的索引
                results.append({
                    "score": float(1 / (1 + distance)),  # 转换为相似度分数
                    "metadata": self.metadata[idx]
                })
        
        return results
    
    def save(self, path: str):
        """保存向量存储到磁盘"""
        # 创建目录
        os.makedirs(path, exist_ok=True)
        
        # 保存索引
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        # 保存元数据
        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """从磁盘加载向量存储"""
        # 加载索引
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        
        # 加载元数据
        with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as f:
            self.metadata = json.load(f) 