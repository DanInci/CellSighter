import argparse
import sys

sys.path.append(".")
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import numpy as np
import pandas as pd
from model import Model
from data.data import CellCropsDataset
from data.utils import load_crops, is_nested
from data.transform import val_transform
from torch.utils.data import DataLoader
from metrics.metrics import Metrics
import json


def val_epoch(model, dataloader, device=None):
    with torch.no_grad():
        model.eval()
        results = []
        cells = []
        for i, batch in enumerate(dataloader):
            x = batch['image']
            m = batch.get('mask', None)
            if m is not None:
                x = torch.cat([x, m], dim=1)
            x = x.to(device=device)
            m = m.to(device=device)
            y_pred = model(x)
            results += y_pred.detach().cpu().numpy().tolist()

            del batch["image"]
            cells += [batch]
            if i % 500 == 0:
                print(f"Eval {i} / {len(dataloader)}")
        return cells, np.array(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--base_path', type=str,
                        help='configuration_path')

    args = parser.parse_args()
    writer = SummaryWriter(log_dir=args.base_path)

    config_path = os.path.join(args.base_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    train_sets = config["train_set"]
    val_sets = config["val_set"]

    # if nested list, then do cross validation on multiple splits
    if is_nested(train_sets):
        assert is_nested(val_sets)
        assert len(train_sets) == len(val_sets)

        print(f"Detected multiple data splits. Evaluating CellSighter on {len(train_sets)} folds.")
        splits = list(zip(train_sets, val_sets))

    # else just one split
    else:
        splits = [[train_sets, val_sets]]

    agg_eval_results_df = pd.DataFrame()
    for split_i, split in enumerate(splits):
        train_set, val_set = split

        if len(splits) > 1:
            print(f"Evaluating CellSighter on fold {split_i} ...")
            split_dir = os.path.join(args.base_path, f"split_{split_i}")
        else:
            split_dir = args.base_path

        assert os.path.exists(split_dir)

        _, val_crops = load_crops(config["root_dir"],
                                config["channels_path"],
                                config["crop_size"],
                                train_set,
                                val_set,
                                to_pad=True,
                                blacklist_channels=config["blacklist"])
        crop_input_size = config["crop_input_size"] if "crop_input_size" in config else 100
        val_dataset = CellCropsDataset(val_crops, transform=val_transform(crop_input_size), mask=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_channels = sum(1 for line in open(config["channels_path"])) + 1 - len(config["blacklist"])
        class_num = config["num_classes"]

        model = Model(num_channels+1, class_num)
        eval_weights = config["weight_to_eval"]
        eval_weights_split = os.path.join(split_dir, eval_weights)
        model.load_state_dict(torch.load(eval_weights_split))
        model = model.to(device=device)

        print(f'Evaluating on {device}')
        val_loader = DataLoader(val_dataset, batch_size=12,
                                num_workers=1, shuffle=False, pin_memory=True)
        cells, results = val_epoch(model, val_loader, device=device)

        metrics = Metrics([], writer, prefix="val")
        metrics(cells, results, 0)
        results_df = metrics.get_results(cells, results)
        agg_eval_results_df = pd.concat([agg_eval_results_df, results_df])

    agg_eval_results_df.to_csv(os.path.join(args.base_path, f"cellsighter_results_eval.csv"))
