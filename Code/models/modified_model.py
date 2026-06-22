import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    MAX_RANK,
    ORDER_LOSS_WEIGHT,
    SPARSITY_LAMBDA_INIT,
    SUM_CONSTRAINT_WEIGHT,
)


class ModifiedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, N,
                 max_rank=MAX_RANK, initial_sparsity_lambda=SPARSITY_LAMBDA_INIT):
        super(ModifiedModel, self).__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.max_rank = max_rank
        self.sparsity_lambda = nn.Parameter(torch.tensor(initial_sparsity_lambda), requires_grad=False)

        self.lstm_feature_dim = hidden_dim * 2
        self.mf_feature_dim = hidden_dim * 2
        self.total_feature_dim = self.lstm_feature_dim * N * 2

        self.lstm_dropout = 0.5
        self.mf_dropout = 0.4

        self.lstm_sad = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.lstm_dropout
        )
        self.lstm_neutral = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.lstm_dropout
        )

        self.attention = nn.MultiheadAttention(
            hidden_dim * 2,
            num_heads=4,
            dropout=self.lstm_dropout,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        self.fusion_weights = nn.Parameter(torch.ones(2))

        self.feature_transform_sad = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.mf_dropout)
        )

        self.feature_transform_neutral = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.mf_dropout)
        )

        self.U = nn.Parameter(torch.randn(hidden_dim * 2, max_rank) * 0.02)
        self.V = nn.Parameter(torch.randn(max_rank, hidden_dim * 2) * 0.02)

        self.rank_predictor_sad = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, max_rank),
            nn.Sigmoid()
        )
        nn.init.constant_(self.rank_predictor_sad[-2].bias, 0.5)

        self.rank_predictor_neutral = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, max_rank),
            nn.Sigmoid()
        )
        nn.init.constant_(self.rank_predictor_neutral[-2].bias, 0.5)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc_layers = nn.Sequential(
            nn.BatchNorm1d(self.total_feature_dim),
            nn.Linear(self.total_feature_dim, self.total_feature_dim // 2),
            nn.BatchNorm1d(self.total_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.lstm_dropout),
            nn.Linear(self.total_feature_dim // 2, self.total_feature_dim // 4),
            nn.BatchNorm1d(self.total_feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.lstm_dropout),
            nn.Linear(self.total_feature_dim // 4, output_dim)
        )

    def matrix_factorization_sad(self, x):
        x_transformed = self.feature_transform_sad(x)
        global_features = x_transformed.mean(dim=1)
        alpha = self.rank_predictor_sad(global_features)

        U = self.U.to(x.device)
        V = self.V.to(x.device)

        batch_size = x.size(0)
        U = U.unsqueeze(0)
        V = V.unsqueeze(0)
        alpha_expanded = alpha.view(batch_size, 1, self.max_rank)

        U_weighted = U * alpha_expanded
        V_weighted = V * alpha_expanded.transpose(1, 2)

        x_proj = torch.matmul(x_transformed, U_weighted)
        reconstructed = torch.matmul(x_proj, V_weighted)

        out = x_transformed + reconstructed
        out = self.pool(out.transpose(1, 2)).squeeze(-1)

        return out, alpha

    def matrix_factorization_neutral(self, x):
        x_transformed = self.feature_transform_neutral(x)
        global_features = x_transformed.mean(dim=1)
        alpha = self.rank_predictor_neutral(global_features)

        U = self.U.to(x.device)
        V = self.V.to(x.device)

        batch_size = x.size(0)
        U = U.unsqueeze(0)
        V = V.unsqueeze(0)
        alpha_expanded = alpha.view(batch_size, 1, self.max_rank)

        U_weighted = U * alpha_expanded
        V_weighted = V * alpha_expanded.transpose(1, 2)

        x_proj = torch.matmul(x_transformed, U_weighted)
        reconstructed = torch.matmul(x_proj, V_weighted)

        out = x_transformed + reconstructed
        out = self.pool(out.transpose(1, 2)).squeeze(-1)

        return out, alpha

    def update_sparsity_lambda(self, new_value):
        device = next(self.parameters()).device
        self.sparsity_lambda.data = torch.tensor(new_value, device=device)

    def regularization_loss(self, alpha):
        sparsity_loss = self.sparsity_lambda * torch.mean(torch.abs(alpha))

        rank_weights = torch.arange(1, self.max_rank + 1, device=alpha.device).float()
        order_loss = ORDER_LOSS_WEIGHT * torch.mean(alpha * rank_weights)

        sum_constraint = SUM_CONSTRAINT_WEIGHT * torch.abs(torch.sum(alpha, dim=-1) - 1.0).mean()

        return sparsity_loss + order_loss + sum_constraint

    def forward(self, data, training=True):
        total_reg_loss = 0.0

        # 静态特征仍由 data pipeline 提供，但模型决策不使用
        sad_trajs, neutral_trajs, *_ = data
        batch_size = sad_trajs.size(0)
        device = next(self.parameters()).device

        if training:
            sad_trajs = F.dropout(sad_trajs, p=0.1, training=True)
            neutral_trajs = F.dropout(neutral_trajs, p=0.1, training=True)

        fusion_weights = F.softmax(self.fusion_weights, dim=0)

        sad_features = []
        for i in range(self.N):
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim, device=device)
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim, device=device)

            sad_out, _ = self.lstm_sad(sad_trajs[:, i], (h0, c0))
            attn_out, _ = self.attention(sad_out, sad_out, sad_out)
            attn_out = self.layer_norm(attn_out)

            forward_feature = attn_out[:, -1, :self.hidden_dim]
            backward_feature = attn_out[:, 0, self.hidden_dim:]
            lstm_feature = torch.cat([forward_feature, backward_feature], dim=-1)

            mf_feature, mf_alpha = self.matrix_factorization_sad(sad_trajs[:, i])
            total_reg_loss += self.regularization_loss(mf_alpha)

            if training:
                lstm_feature = F.normalize(lstm_feature, p=2, dim=1)
                mf_feature = F.normalize(mf_feature, p=2, dim=1)

            fused_feature = fusion_weights[0] * lstm_feature + fusion_weights[1] * mf_feature
            sad_features.append(fused_feature)

        neutral_features = []
        for i in range(self.N):
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim, device=device)
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim, device=device)

            neutral_out, _ = self.lstm_neutral(neutral_trajs[:, i], (h0, c0))
            attn_out, _ = self.attention(neutral_out, neutral_out, neutral_out)
            attn_out = self.layer_norm(attn_out)

            forward_feature = attn_out[:, -1, :self.hidden_dim]
            backward_feature = attn_out[:, 0, self.hidden_dim:]
            lstm_feature = torch.cat([forward_feature, backward_feature], dim=-1)

            mf_feature, mf_alpha = self.matrix_factorization_neutral(neutral_trajs[:, i])
            total_reg_loss += self.regularization_loss(mf_alpha)

            if training:
                lstm_feature = F.normalize(lstm_feature, p=2, dim=1)
                mf_feature = F.normalize(mf_feature, p=2, dim=1)

            fused_feature = fusion_weights[0] * lstm_feature + fusion_weights[1] * mf_feature
            neutral_features.append(fused_feature)

        sad_combined = torch.cat([f.unsqueeze(1) for f in sad_features], dim=1)
        neutral_combined = torch.cat([f.unsqueeze(1) for f in neutral_features], dim=1)

        all_features = torch.cat([
            sad_combined.view(batch_size, -1),
            neutral_combined.view(batch_size, -1),
        ], dim=1)

        out = self.fc_layers(all_features)
        return out, total_reg_loss
