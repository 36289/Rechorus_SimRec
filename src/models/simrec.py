import torch
import torch.nn as nn
import torch.nn.functional as F


class SimRec(nn.Module):
    # 一个简单的知识蒸馏推荐模型，包含GNN教师和MLP学生
    def __init__(self, input_dim, hidden_dim, n_layers=2, dropout=0.2):
        super(SimRec, self).__init__()
        self.n_layers = n_layers
        
        # 学生模型 (MLP)
        self.student_layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(n_layers)
        ])
        self.student_bn = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)
        ])
        self.student_dropout = nn.Dropout(dropout)
        
        # 教师模型 (LightGCN)
        self.teacher_embedding = nn.Linear(input_dim, hidden_dim)
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def gnn_teacher(self, adj_matrix, features):
        # 计算归一化邻接矩阵
        deg = torch.sum(adj_matrix, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        norm_adj = torch.mul(
            torch.mul(adj_matrix, deg_inv_sqrt.unsqueeze(1)),
            deg_inv_sqrt.unsqueeze(0)
        )
        
        x = self.teacher_embedding(features)
        emb_list = [x]
        
        for _ in range(self.n_layers):
            x = torch.sparse.mm(norm_adj, x) if adj_matrix.is_sparse else torch.mm(norm_adj, x)
            emb_list.append(x)
            
        return torch.stack(emb_list, dim=0).mean(dim=0)

    def mlp_student(self, features):
        x = features
        for i in range(self.n_layers):
            x = self.student_layers[i](x)
            x = self.student_bn[i](x)
            x = F.relu(x)
            x = self.student_dropout(x)
        return x

    def forward(self, adj_matrix, features):
        teacher_embeddings = self.gnn_teacher(adj_matrix, features)
        student_embeddings = self.mlp_student(features)
        return teacher_embeddings, student_embeddings

    def get_predictions(self, embeddings):
        return self.predictor(embeddings)


def prediction_distillation_loss(student_preds, teacher_preds, tau=1.0):
    # 预测层面的蒸馏损失
    student_soft = F.softmax(student_preds / tau, dim=1)
    teacher_soft = F.softmax(teacher_preds / tau, dim=1)
    return F.kl_div(student_soft.log(), teacher_soft, reduction="batchmean") * (tau ** 2)


def embedding_distillation_loss(student_embeddings, teacher_embeddings, tau=0.1):
    # 嵌入层面的对比损失
    student_norm = F.normalize(student_embeddings, dim=1)
    teacher_norm = F.normalize(teacher_embeddings, dim=1)
    cosine_similarity = torch.mm(student_norm, teacher_norm.t())
    labels = torch.arange(student_embeddings.size(0)).to(student_embeddings.device)
    return F.cross_entropy(cosine_similarity / tau, labels)


class SimRecLoss(nn.Module):
    # 组合损失函数
    def __init__(self, pred_weight=1.0, emb_weight=1.0, pred_tau=1.0, emb_tau=0.1):
        super().__init__()
        self.pred_weight = pred_weight
        self.emb_weight = emb_weight
        self.pred_tau = pred_tau
        self.emb_tau = emb_tau
        
    def forward(self, student_embeddings, teacher_embeddings, student_preds=None, teacher_preds=None):
        loss = 0
        if self.emb_weight > 0:
            loss += self.emb_weight * embedding_distillation_loss(
                student_embeddings, teacher_embeddings, tau=self.emb_tau
            )
        if self.pred_weight > 0 and student_preds is not None and teacher_preds is not None:
            loss += self.pred_weight * prediction_distillation_loss(
                student_preds, teacher_preds, tau=self.pred_tau
            )
        return loss