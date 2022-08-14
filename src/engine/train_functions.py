import time
from tqdm import tqdm
import copy
import numpy as np
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
import torch


def train_on_step(cfg, model, criterion, optimizer, scheduler, dataloader):
    model.train()

    running_losses = []
    train_acc_scores = []
    train_precisions = []
    train_f1_scores = []
    train_recalls = []
    lr = []

    tqdm_bar = tqdm(dataloader, total=len(dataloader))

    for step, data in enumerate(tqdm_bar):
        ids = data["input_ids"].to(cfg["device"], dtype=torch.long)
        mask = data["attention_mask"].to(cfg["device"], dtype=torch.long)
        labels = data["target"].to(cfg["device"], dtype=torch.long)

        pred_logits = model(ids, mask)
        loss = criterion(pred_logits, labels)
        loss = loss / cfg["n_accumulate"]
        loss.backward()

        if (step + 1) % cfg["n_accumulate"] == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        running_losses.append(loss.item())

        # Calculate metrics=
        _, effectiveness_pred = torch.max(pred_logits, dim=1)
        train_acc_scores.append(accuracy_score(labels.cpu(), effectiveness_pred.cpu()))
        train_precisions.append(
            precision_score(labels.cpu(), effectiveness_pred.cpu(), average="macro", zero_division=0))
        train_f1_scores.append(f1_score(labels.cpu(), effectiveness_pred.cpu(), average="macro", zero_division=0))
        train_recalls.append(recall_score(labels.cpu(), effectiveness_pred.cpu(), average="macro", zero_division=0))

        epoch_loss = sum(running_losses) / len(running_losses)
        epoch_acc = sum(train_acc_scores) / len(train_acc_scores)
        epoch_precision = sum(train_precisions) / len(train_precisions)
        epoch_f1 = sum(train_f1_scores) / len(train_f1_scores)
        epoch_recall = sum(train_recalls) / len(train_recalls)

        tqdm_bar.set_postfix(
            Train_loss=epoch_loss,
            Train_acc=epoch_acc,
            Train_precision=epoch_precision,
            Train_f1=epoch_f1,
            Train_recall=epoch_recall,
            LR=optimizer.param_groups[0]["lr"],
        )
        lr.append(optimizer.param_groups[0]['lr'])
    return epoch_loss, epoch_acc, lr


def eval_on_step(cfg, model, criterion, dataloader):
    with torch.no_grad():
        model.eval()

        print("Evaluating ...")
        running_losses = []
        val_acc_scores = []
        val_precisions = []
        val_f1_scores = []
        val_recalls = []

        for data in tqdm(dataloader):
            ids = data["input_ids"].to(cfg["device"], dtype=torch.long)
            mask = data["attention_mask"].to(cfg["device"], dtype=torch.long)
            labels = data["target"].to(cfg["device"], dtype=torch.long)

            pred_logits = model(ids, mask)
            loss = criterion(pred_logits, labels)
            running_losses.append(loss.item())

            # Calculate metrics
            _, effectiveness_pred = torch.max(pred_logits, dim=1)
            val_acc_scores.append(accuracy_score(labels.cpu(), effectiveness_pred.cpu()))
            val_precisions.append(
                precision_score(labels.cpu(), effectiveness_pred.cpu(), average="macro", zero_division=0))
            val_f1_scores.append(f1_score(labels.cpu(), effectiveness_pred.cpu(), average="macro", zero_division=0))
            val_recalls.append(recall_score(labels.cpu(), effectiveness_pred.cpu(), average="macro", zero_division=0))

        epoch_loss = sum(running_losses) / len(running_losses)
        epoch_acc = sum(val_acc_scores) / len(val_acc_scores)
        epoch_precision = sum(val_precisions) / len(val_precisions)
        epoch_f1 = sum(val_f1_scores) / len(val_f1_scores)
        epoch_recall = sum(val_recalls) / len(val_recalls)

        print("Val loss: ", epoch_loss)
        print("Val scores: Precision = %2.5f, Recall = %2.5f, F1_score = %2.5f, Acc_score = %2.5f" % (
            epoch_precision,
            epoch_recall,
            epoch_f1,
            epoch_acc))

        return epoch_loss, epoch_acc


def train_model(cfg, model, criterion, optimizer, scheduler, train_loader, val_loader):
    start = time.time()

    if cfg["resume"]:
        saved_state_dict = torch.load(cfg["best_checkpoint"]).to(cfg["device"])
        model.load_state_dict(saved_state_dict)

    best_epoch_loss = np.inf
    log = {
        "Train Loss": [],
        "Train Acc": [],
        "Valid Loss": [],
        "Valid Acc": [],
        "LR": []
    }
    for epoch in range(1, cfg["num_epoch"] + 1):
        print("EPOCH : {}".format(epoch))
        train_epoch_loss, train_epoch_acc, train_lr = train_on_step(cfg, model, criterion, optimizer, scheduler,
                                                                    train_loader)
        val_epoch_loss, val_epoch_acc = eval_on_step(cfg, model, criterion, val_loader)
        log["Train Loss"].append(train_epoch_loss)
        log["Valid Loss"].append(val_epoch_loss)
        log["Train Acc"].append(train_epoch_acc)
        log["Valid Acc"].append(val_epoch_acc)
        log["LR"].extend(train_lr)

        # Save best model
        if val_epoch_loss <= best_epoch_loss:
            print(f'Validation Loss improved ({best_epoch_loss}--->{val_epoch_loss}) \nSaving as best model...')
            best_epoch_loss = val_epoch_loss
            best_epoch_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), cfg["best_checkpoint"])

        # Save last model
        print("Saving checkpoint(last epoch) ...", end=">>>>")
        torch.save(model.state_dict(), cfg["last_checkpoint"])
        print("Done")

    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    print(
        "Best Loss: {:.4f} Best Accuracy: {:.2f}".format(
            best_epoch_loss, best_epoch_acc * 100
        )
    )

    # Load best model
    model.load_state_dict(best_model_wts)

    return model, log