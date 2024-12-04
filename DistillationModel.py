# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, List
from helpers.BaseReader import BaseReader  # 确保 BaseReader 已经导入
from models.BaseModel import BaseModel  # 确保 BaseModel 已经导入
import numpy as np
from tqdm import tqdm


class DistillationModel(BaseModel):
    """
    DistillationModel继承自BaseModel，并增加了对知识蒸馏（Distillation）的支持。
    学生模型通过与教师模型的输出进行对比，计算蒸馏损失。
    """

    @staticmethod
    def parse_model_args(parser):
        """
        增加解析参数以支持蒸馏损失权重。
        """
        parser.add_argument('--distillation_alpha', type=float, default=0.5,
                            help='The weight for distillation loss in total loss calculation.')
        parser.add_argument('--teacher_model_path', type=str, default=None,
                            help='Path to the pretrained teacher model.')
        parser.add_argument('--freeze_teacher', action='store_true',
                            help='If set, freeze the parameters of the teacher model.')
        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus: BaseReader, teacher_model=None):
        """
        初始化DistillationModel，包含学生模型和教师模型。
        :param args: 解析的输入参数
        :param corpus: 数据读取器，BaseReader的实例
        :param teacher_model: 预训练的教师模型（可选）
        """
        super(DistillationModel, self).__init__(args, corpus)
        
        # 加载教师模型
        if args.teacher_model_path:
            self.teacher_model = torch.load(args.teacher_model_path)
            self.teacher_model.eval()  # 设置为评估模式
            if args.freeze_teacher:
                for param in self.teacher_model.parameters():
                    param.requires_grad = False  # 冻结教师模型参数
        else:
            self.teacher_model = teacher_model
        
        self.distillation_alpha = args.distillation_alpha  # 蒸馏损失的权重系数

    def forward(self, feed_dict: dict) -> dict:
        """
        执行前向传播，包含学生模型和教师模型的输出。
        :param feed_dict: 模型的输入字典
        :return: 模型的输出字典，包括学生模型和教师模型的嵌入输出
        """
        # 学生模型的前向传播
        out_dict = super().forward(feed_dict)

        # 如果有教师模型，执行教师模型的前向传播
        if self.teacher_model:
            with torch.no_grad():  # 教师模型的输出不参与梯度计算
                teacher_out_dict = self.teacher_model(feed_dict)
            out_dict['teacher_embedding'] = teacher_out_dict.get('embedding', None)
            out_dict['teacher_prediction'] = teacher_out_dict.get('prediction', None)

        return out_dict

    def loss(self, out_dict: dict) -> torch.Tensor:
        """
        计算总损失，包含学生模型的损失和蒸馏损失。
        :param out_dict: 模型的输出字典，包含学生模型和教师模型的输出
        :return: 总的损失值
        """
        # 学生模型的原始任务损失
        student_loss = super().loss(out_dict)

        # 计算蒸馏损失
        if 'teacher_embedding' in out_dict and 'teacher_prediction' in out_dict:
            teacher_embedding = out_dict['teacher_embedding']
            teacher_prediction = out_dict['teacher_prediction']
            student_embedding = out_dict['embedding']  # 假设学生模型也返回 'embedding'
            student_prediction = out_dict['prediction']  # 假设学生模型返回 'prediction'

            # 嵌入级别蒸馏损失
            embedding_loss = F.mse_loss(student_embedding, teacher_embedding)

            # 预测级别蒸馏损失（KL散度）
            prediction_loss = F.kl_div(
                F.log_softmax(student_prediction, dim=-1),
                F.softmax(teacher_prediction, dim=-1),
                reduction="batchmean"
            )

            # 总蒸馏损失
            distillation_loss = embedding_loss + prediction_loss

            # 综合总损失
            total_loss = self.distillation_alpha * distillation_loss + (1 - self.distillation_alpha) * student_loss
        else:
            total_loss = student_loss

        return total_loss

    def customize_parameters(self) -> list:
        """
        自定义优化器参数的设置，例如不同的学习率或权重衰减。
        """
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        return [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]


class DistillationDataset(BaseModel.Dataset):
    """
    DistillationDataset继承自BaseModel的Dataset类，并增加了支持教师模型输出的能力。
    """

    def __init__(self, model, corpus, phase, teacher_outputs=None):
        """
        初始化DistillationDataset，支持加载教师模型的输出。
        :param model: 学生模型的实例
        :param corpus: 数据读取器
        :param phase: 数据集的阶段（train, dev, test）
        :param teacher_outputs: 预先计算好的教师模型输出
        """
        super().__init__(model, corpus, phase)
        self.teacher_outputs = teacher_outputs  # 教师模型的输出

    def _get_feed_dict(self, index: int) -> dict:
        """
        构建数据输入字典，包括学生模型和教师模型的输入。
        :param index: 数据索引
        :return: 返回包含学生和教师模型输入的字典
        """
        # 获取基本的学生模型输入
        feed_dict = super()._get_feed_dict(index)

        # 如果有教师模型输出，将其添加到 feed_dict 中
        if self.teacher_outputs is not None:
            feed_dict['teacher_embedding'] = self.teacher_outputs[index]

        return feed_dict

    def actions_before_epoch(self):
        """
        在每个训练epoch之前调用，用于准备负采样数据。
        """
        super().actions_before_epoch()
