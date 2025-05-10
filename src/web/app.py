from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import uvicorn
import os
import yaml
import json
from typing import List, Optional
import shutil
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入自定义模块
from src.document_processor.base import PDFProcessor, WordProcessor, ExcelProcessor
from src.image_processor.processor import ImageProcessor
from src.table_processor.processor import TableProcessor
from src.vector_store.store import VectorStore
from src.llm.qa import LLMQA

app = FastAPI(title="Multimodal RAG System")

# 加载配置
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 创建必要的目录
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs(config["system"]["temp_dir"], exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 设置模板
templates = Jinja2Templates(directory="templates")

# 初始化处理器
document_processors = {
    "pdf": PDFProcessor(config),
    "docx": WordProcessor(config),
    "xlsx": ExcelProcessor(config)
}
image_processor = ImageProcessor(config)
table_processor = TableProcessor(config)
vector_store = VectorStore(config)
llm_qa = LLMQA(config)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """渲染主页"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件并处理"""
    # 检查文件类型
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in config["document_processor"]["supported_formats"]:
        raise HTTPException(400, "Unsupported file format")
    
    # 保存文件
    file_path = os.path.join(config["system"]["temp_dir"], file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        # 处理文档
        processor = document_processors.get(file_ext)
        if not processor:
            raise HTTPException(400, "No processor available for this file type")
        
        result = processor.process(file_path)
        
        # 处理图片
        if "images" in result:
            image_results = image_processor.process_batch(result["images"])
            result["image_analysis"] = image_results
        
        # 处理表格
        if "tables" in result:
            table_results = table_processor.batch_process_tables(result["tables"])
            result["table_analysis"] = table_results
        
        # 添加到向量存储
        vector_store.add_documents([{
            "text": result.get("text", ""),
            "images": result.get("image_analysis", []),
            "tables": result.get("table_analysis", []),
            "source": file.filename,
            "type": file_ext
        }])
        
        return {"message": "File processed successfully", "result": result}
    
    except Exception as e:
        raise HTTPException(500, str(e))
    
    finally:
        # 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/query")
async def query(query: str = Form(...)):
    """处理用户查询"""
    try:
        # 检索相关文档
        search_results = vector_store.search(query)
        
        # 生成答案
        answer = llm_qa.generate_answer(query, search_results)
        
        return {
            "answer": answer["answer"],
            "sources": search_results
        }
    
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/sources")
async def get_sources():
    """获取所有文档源"""
    return {"sources": vector_store.metadata}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config["web"]["host"],
        port=config["web"]["port"],
        reload=config["web"]["debug"]
    ) 