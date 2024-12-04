# -*- coding: UTF-8 -*-

import os
import logging
import numpy as np
import pandas as pd
from utils import utils
from helpers.BaseReader import BaseReader  # 继承BaseReader

class DistillationReader(BaseReader):
    @staticmethod
    def parse_data_args(parser):
        parser = BaseReader.parse_data_args(parser)
        parser.add_argument('--teacher_output_path', type=str, default=None,
                            help='Path to load teacher model outputs (e.g., embeddings or predictions).')
        return parser

    def __init__(self, args):
        super(DistillationReader, self).__init__(args)  # 调用父类的初始化方法
        self.teacher_output_path = args.teacher_output_path
        if self.teacher_output_path:
            self._load_teacher_outputs()  # 加载教师模型输出

    def _load_teacher_outputs(self):
        """加载教师模型的输出，例如嵌入或者预测结果"""
        logging.info(f'Loading teacher model outputs from {self.teacher_output_path}...')
        self.teacher_outputs = dict()
        for key in ['train', 'dev', 'test']:
            file_path = os.path.join(self.teacher_output_path, key + '_teacher_outputs.npy')
            self.teacher_outputs[key] = np.load(file_path)
        logging.info('Teacher model outputs loaded.')

    def _read_data(self):
        """读取数据和教师模型输出"""
        super(DistillationReader, self)._read_data()  # 调用父类的读取数据逻辑
        if self.teacher_output_path:
            self._verify_teacher_outputs()  # 验证教师模型输出是否与数据匹配

    def _verify_teacher_outputs(self):
        """确保教师模型输出和训练数据集的用户和项目ID匹配"""
        for key in ['train', 'dev', 'test']:
            data = self.data_df[key]
            teacher_output = self.teacher_outputs[key]
            assert len(data) == len(teacher_output), f"Teacher outputs and data size mismatch in {key} set."
            logging.info(f'Teacher outputs verified for {key} set: {len(teacher_output)} examples.')

    def get_teacher_outputs(self, key):
        """根据数据集（train/dev/test）返回教师模型的输出"""
        return self.teacher_outputs[key]
