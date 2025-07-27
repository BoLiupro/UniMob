from transformers import Trainer
import torch
import torch.nn.functional as F
from torch import nn

class TimeDistanceLoss(nn.Module):
    def __init__(self, max_slot=48, alpha=2, temperature=0.75):
        """
        max_slot: 分类数量（时间 slot 总数）
        alpha: 对时间差的惩罚力度（越大越强调远距离误差）
        temperature: 控制 soft label 的分布宽度
        """
        super().__init__()
        self.max_slot = max_slot
        self.alpha = alpha
        self.temperature = temperature
        self.kl = nn.KLDivLoss(reduction="batchmean")

        # 构造代价矩阵 [slot_i - slot_j]^2
        i = torch.arange(max_slot).unsqueeze(1)
        j = torch.arange(max_slot).unsqueeze(0)
        self.distance_matrix = ((i - j).float().abs()) ** 2  # [max_slot, max_slot]

    def forward(self, logits, target):
        """
        logits: [B, 48]
        target: [B] (int label in [0, 47])
        """
        B = logits.size(0)
        device = logits.device

        # Step 1: softmax logits → log-probs
        log_probs = F.log_softmax(logits, dim=1)  # [B, 48]

        # Step 2: 构造 soft target 分布，惩罚远离真实值的位置
        soft_targets = torch.zeros_like(logits)  # [B, 48]
        for b in range(B):
            true_idx = target[b]
            # 构造 soft target 分布：exp(-distance / T)
            cost_row = self.distance_matrix[true_idx].to(device)
            weight = torch.exp(-cost_row / self.temperature)  # [48]
            soft_targets[b] = weight / weight.sum()  # 归一化，作为 soft label

        # Step 3: KLDivLoss(log_probs, soft_targets)
        loss = self.kl(log_probs, soft_targets.detach())

        return loss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)

        poi_logits = outputs["poi_pred"]           # [B, P]
        duration_logits = outputs["duration_pred"]   # [B, 48]
        labels = outputs["labels"]               # [B, P+1]
        duration_gt = labels[:, 0]                   # [B]
        poi_gt = labels[:, 1:]                   # [B, P]

        # ===== POI KLDiv Loss =====
        loss_fct_poi = torch.nn.KLDivLoss(reduction='batchmean')
        loss_poi = loss_fct_poi(torch.log_softmax(poi_logits,dim=-1), poi_gt)

        # ===== Stay Classification Loss =====
        # 将连续值 stay_gt 映射到 [1, 48] 的整数类别索引
        criterion = TimeDistanceLoss()
        duration_gt_cls = (duration_gt / 0.5).long().clamp(1, 48)-1  # [B]
        loss_duration = criterion(duration_logits, duration_gt_cls)
        pred_class = torch.argmax(duration_logits, dim=-1)
        true_class = ((duration_gt / 0.5).long().clamp(1, 48) - 1)
        acc = (pred_class == true_class).float().mean()

        # 总损失
        loss = loss_poi + 0.05*loss_duration
        print(f"loss_poi: {loss_poi.item():.4f},loss_duration:{0.05* loss_duration.item():.4f}, total_loss: {loss.item():.4f}")
        print("Accuracy:", acc.item())

        return (loss, outputs) if return_outputs else loss

