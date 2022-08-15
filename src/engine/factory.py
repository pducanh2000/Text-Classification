import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from transformers import AutoTokenizer
from src.models.debertav3_base import FeedBackModel
from src.engine.train_functions import train_model
from src.engine.infer_functions import get_submission_csv


class Engine(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = FeedBackModel(self.cfg["model_name"])
        self.model.to(cfg["device"])
        self.optimizer = AdamW(self.model.parameters(), lr=self.cfg["lr"])
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg["T_max"],
                                                        eta_min=self.cfg["min_lr"])
        self.loss_func = nn.CrossEntropyLoss()

    def train(self, train_loader, val_loader):
        self.model, log = train_model(self.cfg, self.model, self.loss_func, self.optimizer, self.scheduler,
                                      train_loader, val_loader)

        return self.model, log

    def inference(self):
        tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_name"])
        self.model.load_state_dict((torch.load(self.cfg["best_checkpoint"])))
        self.model.eval()
        submission = get_submission_csv(self.cfg, self.model, tokenizer)
        return submission
