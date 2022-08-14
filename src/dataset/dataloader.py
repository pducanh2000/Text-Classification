import torch
from torch.utils.data import Dataset
from src.utils.modify_dataframe import general_modify_dataframe, create_numerical_label


class LabeledFeedBackDataset(Dataset):
    def __init__(self, cfg, df, tokenizer):
        super(LabeledFeedBackDataset, self).__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.df = general_modify_dataframe(df, cfg["train_essay_folder"], self.tokenizer)
        self.df = create_numerical_label(self.df, self.cfg)
        self.inputs = self.df["inputs"].values
        self.essays = self.df["essays"].values
        self.labels = self.df["labels"].values

    def __getitem__(self, item):
        input_text = self.inputs[item]
        essay = self.essays[item]
        label = self.labels[item]
        encoded_dict = self.tokenizer.encode_plus(
            input_text,
            essay,  # The essay string is passed as text_pair argument to the tokenisation function
            truncation=True,
            add_special_tokens=True,
            max_length=self.cfg["max_len"],
            padding="max_length",
        )
        return {
            'input_ids': encoded_dict['input_ids'],
            'attention_mask': encoded_dict["attention_mask"],
            'target': torch.tensor(label)
        }

    def __len__(self):
        return len(self.df)
