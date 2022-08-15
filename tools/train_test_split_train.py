import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from src.dataset.dataloader import LabeledFeedBackDataset
from src.configs.config import Cfg
from src.engine.factory import Engine
from src.utils.visualize import plot_log


def get_args_parser():
    parser = argparse.ArgumentParser("VN_team_3_Danny", add_help=False)
    parser.add_argument('--cfg', default='./src/configs/config.yaml', type=str, help='Path to config yaml file')
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VN_team_3_Danny', parents=[get_args_parser()])
    args = parser.parse_args()

    cfg = Cfg().load_config_from_file(args.cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    df = pd.read_csv(cfg["train_csv_path"])
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=cfg["seed"], shuffle=True)
    train_dataset = LabeledFeedBackDataset(cfg, train_df, tokenizer=tokenizer)
    val_dataset = LabeledFeedBackDataset(cfg, val_df, tokenizer=tokenizer)
    collate_fn = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train_batch_size"],
        collate_fn=collate_fn,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["val_batch_size"],
        collate_fn=collate_fn,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )

    engine = Engine(cfg)
    model, log = engine.train(train_loader, val_loader)
    plot_log(log)
