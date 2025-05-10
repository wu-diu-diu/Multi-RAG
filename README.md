# Multimodal RAG System

一个支持多模态文档检索和问答的系统，支持处理Word、Excel、PPT、PDF等格式的文档。

## 功能特点

- 📄 多格式文档处理：支持Word、Excel、PPT、PDF等格式
- 🎨 图像理解：使用BLIP2/CLIP进行图像语义理解
- 📊 表格分析：支持表格结构化分析和问答
- 🔍 智能检索：基于FAISS的高效向量检索
- 💬 智能问答：支持多种大语言模型
- 🌐 交互式界面：支持图文高亮和来源引用

## 项目结构

```
.
├── Multimodal_data/          # 数据目录
├── src/                      # 源代码
│   ├── document_processor/   # 文档处理模块
│   ├── image_processor/      # 图像处理模块
│   ├── table_processor/      # 表格处理模块
│   ├── vector_store/         # 向量存储模块
│   ├── llm/                  # 大语言模型模块
│   └── web/                  # Web界面模块
├── tests/                    # 测试代码
├── requirements.txt          # 项目依赖
└── README.md                 # 项目说明
```

## 安装

1. 克隆项目
```bash
git clone [repository_url]
cd Multimodal-RAG
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 安装Tesseract OCR
- Windows: 下载安装包并添加到PATH
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

## 使用说明

1. 启动Web服务
```bash
python src/web/app.py
```

2. 访问 http://localhost:8000 使用系统

## 配置说明

在 `config.yaml` 中配置：
- 模型参数
- 向量存储设置
- API密钥
- 其他系统参数

## 许可证

MIT License 