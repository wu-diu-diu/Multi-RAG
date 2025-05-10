from typing import List, Dict, Union
import pandas as pd
import torch
from transformers import TapasTokenizer, TapasForQuestionAnswering
import numpy as np

class TableProcessor:
    """表格处理器，使用TAPAS模型进行表格问答"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_rows = config['table_processor']['max_rows']
        self.max_columns = config['table_processor']['max_columns']
        
        # 初始化TAPAS模型
        self.tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
        self.model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")
        self.model.to(self.device)
    
    def process_table(self, table_data: List[List[str]], question: str = None) -> Dict:
        """处理表格数据，可选进行问答"""
        # 转换为DataFrame
        df = pd.DataFrame(table_data[1:], columns=table_data[0])
        
        # 如果表格太大，进行截断
        if len(df) > self.max_rows:
            df = df.head(self.max_rows)
        if len(df.columns) > self.max_columns:
            df = df.iloc[:, :self.max_columns]
        
        result = {
            "table": df.to_dict(orient="records"),
            "columns": df.columns.tolist(),
            "shape": df.shape
        }
        
        # 如果提供了问题，进行问答
        if question:
            answer = self.answer_question(df, question)
            result["qa"] = answer
        
        return result
    
    def answer_question(self, df: pd.DataFrame, question: str) -> Dict:
        """使用TAPAS模型回答关于表格的问题"""
        # 准备输入
        inputs = self.tokenizer(
            table=df,
            queries=question,
            return_tensors="pt",
            truncation=True
        )
        
        # 将输入移到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 获取预测
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 处理预测结果
        predicted_answer_coordinates, predicted_aggregation_indices = self.model.convert_logits_to_predictions(
            inputs, outputs.logits, outputs.logits_aggregation
        )
        
        # 获取答案
        answers = []
        for coordinates in predicted_answer_coordinates:
            if len(coordinates) == 1:
                # 单个单元格答案
                row, col = coordinates[0]
                answers.append(str(df.iloc[row, col]))
            else:
                # 多个单元格答案
                cell_values = [str(df.iloc[row, col]) for row, col in coordinates]
                answers.append(", ".join(cell_values))
        
        # 获取聚合操作
        aggregation_operations = ["NONE", "SUM", "AVERAGE", "COUNT"]
        aggregation = aggregation_operations[predicted_aggregation_indices[0]]
        
        return {
            "answer": answers[0] if len(answers) == 1 else answers,
            "aggregation": aggregation,
            "coordinates": predicted_answer_coordinates[0].tolist()
        }
    
    def batch_process_tables(self, tables: List[List[List[str]]], questions: List[str] = None) -> List[Dict]:
        """批量处理多个表格"""
        results = []
        for i, table in enumerate(tables):
            question = questions[i] if questions else None
            result = self.process_table(table, question)
            results.append(result)
        return results 