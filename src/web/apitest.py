from fastapi import FastAPI
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

app = FastAPI(title="Multimodal RAG System")

templates = Jinja2Templates(directory="src/web/templates")

app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """渲染主页"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

## 定义一个路径带变量的接口
## 用户在浏览器输入：http://localhost:8000/say/hello?q=123
## 会触发say函数，并返回{"data": "hello", "item": 123}
@app.get("/say/{data}")
def say(data: str,q: int):
    return {"data": data, "item": q}

if __name__ == "__main__":
    ## uvicorn是Web服务启动器，运行API服务
    uvicorn.run(app, host="0.0.0.0", port=8000)