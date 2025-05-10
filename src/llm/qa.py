from typing import List, Dict, Optional
import json
import zhipuai
from dashscope import Generation
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

class LLMQA:
    """大语言模型问答类，支持多种模型"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config['llm']['default_model']
        self.temperature = config['llm']['temperature']
        self.max_tokens = config['llm']['max_tokens']
        
        # 初始化API密钥
        self._init_api_keys()
        
        # 初始化模型
        self._init_model()
    
    def _init_api_keys(self):
        """初始化各模型的API密钥"""
        api_keys = self.config['llm']['api_keys']
        
        # OpenAI
        if api_keys.get('openai'):
            openai.api_key = api_keys['openai']
        
        # 智谱AI
        if api_keys.get('zhipu'):
            zhipuai.api_key = api_keys['zhipu']
        
        # 通义千问
        if api_keys.get('qwen'):
            Generation.set_api_key(api_keys['qwen'])
    
    def _init_model(self):
        """初始化选定的模型"""
        if self.model_name == "chatglm3":
            self.model = self._init_chatglm3()
        elif self.model_name == "qwen":
            self.model = self._init_qwen()
        elif self.model_name == "gpt4":
            self.model = self._init_gpt4()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _init_chatglm3(self):
        """初始化ChatGLM3模型"""
        return zhipuai.ZhipuAI()
    
    def _init_qwen(self):
        """初始化通义千问模型"""
        return Generation()
    
    def _init_gpt4(self):
        """初始化GPT-4模型"""
        return ChatOpenAI(
            model_name="gpt-4",
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def generate_answer(self, query: str, context: List[Dict]) -> Dict:
        """生成答案"""
        # 准备上下文
        context_text = self._prepare_context(context)
        
        # 根据不同的模型生成答案
        if self.model_name == "chatglm3":
            return self._generate_chatglm3(query, context_text)
        elif self.model_name == "qwen":
            return self._generate_qwen(query, context_text)
        else:
            return self._generate_gpt4(query, context_text)
    
    def _prepare_context(self, context: List[Dict]) -> str:
        """准备上下文文本"""
        context_text = ""
        for item in context:
            if "text" in item:
                context_text += f"文本内容：{item['text']}\n"
            if "caption" in item:
                context_text += f"图片描述：{item['caption']}\n"
            if "table" in item:
                context_text += f"表格内容：{json.dumps(item['table'], ensure_ascii=False)}\n"
            context_text += f"来源：{item.get('source', '')}\n"
            context_text += "---\n"
        return context_text
    
    def _generate_chatglm3(self, query: str, context: str) -> Dict:
        """使用ChatGLM3生成答案"""
        prompt = f"""基于以下参考信息回答问题。如果参考信息不足以回答问题，请说明无法回答。

参考信息：
{context}

问题：{query}

请给出详细、准确的回答，并标注信息来源。"""
        
        response = self.model.chat.completions.create(
            model="chatglm3",
            messages=[
                {"role": "system", "content": "你是一个专业的问答助手，擅长基于给定信息回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return {
            "answer": response.choices[0].message.content,
            "model": "chatglm3"
        }
    
    def _generate_qwen(self, query: str, context: str) -> Dict:
        """使用通义千问生成答案"""
        prompt = f"""基于以下参考信息回答问题。如果参考信息不足以回答问题，请说明无法回答。

参考信息：
{context}

问题：{query}

请给出详细、准确的回答，并标注信息来源。"""
        
        response = self.model.call(
            model="qwen-max",
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return {
            "answer": response.output.text,
            "model": "qwen"
        }
    
    def _generate_gpt4(self, query: str, context: str) -> Dict:
        """使用GPT-4生成答案"""
        messages = [
            SystemMessage(content="你是一个专业的问答助手，擅长基于给定信息回答问题。"),
            HumanMessage(content=f"""基于以下参考信息回答问题。如果参考信息不足以回答问题，请说明无法回答。

参考信息：
{context}

问题：{query}

请给出详细、准确的回答，并标注信息来源。""")
        ]
        
        response = self.model(messages)
        
        return {
            "answer": response.content,
            "model": "gpt4"
        } 