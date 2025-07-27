import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

class TrajectoryHead(nn.Module):
    def __init__(self, hidden_size=4096, num_poi=14):
        super().__init__()
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self.poi_head = nn.Linear(hidden_size, num_poi)
        self.duration_head = nn.Linear(hidden_size, 48)

    def forward(self, last_hidden_state, attention_mask=None):
        scores = self.attn_pool(last_hidden_state).squeeze(-1)  # [B, T]
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=1)  # [B, T]
        pooled = torch.sum(last_hidden_state * weights.unsqueeze(-1), dim=1) 
        poi_logits = self.poi_head(pooled)           # [B, 14]
        duration_logits = self.duration_head(pooled) # [B, 48]
        return poi_logits, duration_logits

class temporalTripModel(nn.Module):
    def __init__(self, base_model_path, use_peft=True, hidden_size=4096, num_poi=14):
        super().__init__()
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        if base_model_path is not None:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=self.quant_config,
                device_map="auto",
                cache_dir="checkpt/models/",
            )
            base_model.config.use_cache = False
            base_model.config.pretraining_tp = 1

            if use_peft:
                peft_config = LoraConfig(
                    lora_alpha=16,
                    lora_dropout=0.1,
                    r=64,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                self.llm = get_peft_model(base_model, peft_config)
            else:
                self.llm = base_model
        else:
            self.llm = None

        self.trajectory_head = TrajectoryHead(hidden_size, num_poi)

    def forward(self, input_ids, attention_mask, labels=None, city_id = None, return_dict=True):
        output = self.llm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        last_hidden = output.hidden_states[-1]  # [B, T, D]
        if labels is not None:
            city_id = labels[:, 0].long()  # [B, T]，城市 ID
            labels = labels[:, 1:]  # [B, T, P+1]，去掉城市 ID
            poi_logits, duration_logits = self.trajectory_head(last_hidden)
            return {
                "poi_pred": poi_logits,
                "duration_pred": duration_logits,
                "labels": labels
            }
        else:
            poi_logits, duration_logits = self.trajectory_head(last_hidden)
            return {
                "poi_pred": poi_logits,
                "duration_pred": duration_logits,
            }

    def save_pretrained(self, save_path):
        self.llm.save_pretrained(save_path)


    @classmethod
    def load_model(cls, llm_path, trajectory_head):
        # 加载量化 LLM + PEFT
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            quantization_config=quant_config,
            device_map="auto",
            cache_dir="checkpt/models/",
        )
        llm.config.use_cache = False
        llm.config.pretraining_tp = 1

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM"
        )
        llm = get_peft_model(llm, peft_config)

        # 构建模型实例
        model = cls(base_model_path=None,use_peft=False)
        model.llm = llm  # 替换 LLM
        model.trajectory_head = trajectory_head  # 替换头部
        return model