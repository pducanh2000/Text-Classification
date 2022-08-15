import os

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F


def infer_sample(essay_id, discourse_type, discourse_text, data_path, model, tokenizer, max_len, device):
    essay_path = os.path.join(data_path, f"{essay_id}.txt")
    essay_text = open(essay_path, "r").read().strip().lower()
    input_text = discourse_type.lower() + ' ' + tokenizer.sep_token + ' ' + discourse_text.lower()
    inputs = tokenizer.encode_plus(
        input_text,
        essay_text,
        truncation=True,
        add_special_tokens=True,
        max_length=max_len
    )
    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
    model.eval()
    output = model(ids, mask)
    return F.softmax(output, dim=1).detach().cpu().numpy()[0]


def get_submission_csv(cfg, model, tokenizer):
    test_df = pd.read_csv(cfg["test_csv_path"])
    preds = []

    for idx in range(len(test_df)):
        essay_id = test_df.essay_id[idx]
        discourse_text = test_df.discourse_text[idx]
        discourse_type = test_df.discourse_type[idx]
        discourse_pred = infer_sample(essay_id, discourse_type, discourse_text, cfg["test_essay_folder"], model,
                                      tokenizer, cfg["max_len"])
        preds.append(discourse_pred)

    preds = np.array(preds)
    submission = pd.read_csv(cfg["sample_submission_path"])
    submission["Adequate"] = preds[:, cfg["label_map"]["Adequate"]]
    submission["Ineffective"] = preds[:, cfg["label_map"]["Ineffective"]]
    submission["Effective"] = preds[:, cfg["label_map"]["Effective"]]
    submission.to_csv("submission.csv", index=False)
    return submission
