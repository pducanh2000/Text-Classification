import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings


class FeedBackModel(nn.Module):
    def __init__(self, cfg):
        super(FeedBackModel, self).__init__()
        self.config = AutoConfig.from_pretrained(cfg["model_name"])
        # self.model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], config=self.config)
        self.model = AutoModel.from_pretrained(cfg["model_name"])
        self.meanpool = MeanPooling()
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.config.hidden_size, cfg["num_classes"])

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=False)
        out = self.meanpool(out.last_hidden_state, mask)
        out = self.dropout(out)
        out = self.linear(out)
        return out