import gc
import torch
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List
from helpers import BaseRunner
from models.BaseModel import BaseModel  # 假设BaseModel在models文件夹中
from utils import utils  # 你的utils函数，例如batch_to_gpu

class DistillationRunner(BaseRunner):
    def __init__(self, args, teacher_model, student_model):
        super(DistillationRunner, self).__init__(args)
        self.teacher_model = teacher_model  # GNN Teacher model
        self.student_model = student_model  # MLP Student model

    def train(self, data_dict: Dict[str, BaseModel.Dataset]):
        teacher_model = self.teacher_model
        student_model = self.student_model
        main_metric_results, dev_results = list(), list()
        self._check_time(start=True)
        try:
            for epoch in range(self.epoch):
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()

                # 教师模型预测
                teacher_outputs = self.fit_teacher(data_dict['train'], epoch=epoch + 1)

                # 学生模型通过蒸馏进行学习
                student_loss = self.fit_student(data_dict['train'], teacher_outputs, epoch=epoch + 1)

                if np.isnan(student_loss):
                    logging.info("Loss is Nan. Stop training at %d." % (epoch + 1))
                    break
                training_time = self._check_time()

                # 记录和保存模型等
                self.save_and_log(dev_results, epoch, student_loss, main_metric_results, training_time)

                if self.early_stop > 0 and self.eval_termination(main_metric_results):
                    logging.info("Early stop at %d based on dev result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Training stopped manually.")

        # 加载最好的模型
        best_epoch = main_metric_results.index(max(main_metric_results))
        self.student_model.load_model()

    def fit_teacher(self, dataset: BaseModel.Dataset, epoch=-1) -> Dict[str, torch.Tensor]:
        # GNN教师模型训练
        teacher_model = dataset.model
        if teacher_model.optimizer is None:
            teacher_model.optimizer = self._build_optimizer(teacher_model)
        dataset.actions_before_epoch()

        teacher_model.train()
        loss_lst = []
        teacher_outputs = []
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, desc='Teacher Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, teacher_model.device)
            teacher_model.optimizer.zero_grad()
            out_dict = teacher_model(batch)
            teacher_outputs.append(out_dict['embedding'])
            loss = teacher_model.loss(out_dict)
            loss.backward()
            teacher_model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())
        return {"embedding": torch.cat(teacher_outputs)}

    def fit_student(self, dataset: BaseModel.Dataset, teacher_outputs: Dict[str, torch.Tensor], epoch=-1) -> float:
        # MLP学生模型通过蒸馏进行学习
        student_model = dataset.model
        if student_model.optimizer is None:
            student_model.optimizer = self._build_optimizer(student_model)
        dataset.actions_before_epoch()

        student_model.train()
        loss_lst = []
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, desc='Student Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, student_model.device)
            student_model.optimizer.zero_grad()
            out_dict = student_model(batch)
            distillation_loss = self.compute_distillation_loss(out_dict['embedding'], teacher_outputs['embedding'])
            distillation_loss.backward()
            student_model.optimizer.step()
            loss_lst.append(distillation_loss.detach().cpu().data.numpy())
        return np.mean(loss_lst).item()

    def compute_distillation_loss(self, student_embedding, teacher_embedding):
        # L2 loss用于蒸馏学生模型的嵌入
        loss = torch.nn.functional.mse_loss(student_embedding, teacher_embedding)
        return loss
