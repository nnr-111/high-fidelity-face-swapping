import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FaceSwapPairs
from losses import baseline_preservation_loss, edge_preservation_loss, self_reconstruction_loss


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_trainable_parameters(model, keywords):
    for name, param in model.named_parameters():
        param.requires_grad = any(key.lower() in name.lower() for key in keywords)


def load_simswap_model(checkpoint_path=None):
    raise NotImplementedError(
        "Connect this function to the local SimSwap model loader, then keep only identity-related layers trainable."
    )


def train(args):
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FaceSwapPairs(args.pairs_csv, image_size=config["image_size"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], drop_last=True)

    model = load_simswap_model(args.checkpoint)
    set_trainable_parameters(model, config["trainable_keywords"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"])

    for epoch in range(config["epochs"]):
        running_loss = 0.0

        for batch in tqdm(loader, desc=f"epoch {epoch + 1}"):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            baseline = batch.get("baseline")
            if baseline is not None:
                baseline = baseline.to(device)

            output = model(source, target)

            loss = 0.0
            if baseline is not None:
                loss = loss + config["lambda_base"] * baseline_preservation_loss(output, baseline)
            loss = loss + config["lambda_edge"] * edge_preservation_loss(output, target)
            loss = loss + config["lambda_self"] * self_reconstruction_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        torch.save({"epoch": epoch + 1, "model": model.state_dict(), "config": config}, output_dir / f"finetune_epoch_{epoch + 1}.pt")
        print(f"epoch={epoch + 1} loss={running_loss / max(1, len(loader)):.6f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
