"""
    tests / link.py
    ---------------
    Test CLI script for various debugging methods for the link state predictor
    model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import seaborn as sns
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from src.models.chanmod import ChannelModel
from data.loader import DataLoader, shuffle_and_split
from tests.debug.parser import build_parser, mainrunner, CommandSpec


# ============================================================
#       Debugging Testing Methods 
# ============================================================

def test_build_link_model(args: argparse.Namespace):
    c = ChannelModel()
    c.link.build()
    c.link.model.summary()


def test_train_link_model(args: argparse.Namespace):
    df = DataLoader()
    dt, ds = shuffle_and_split(df.load(args.dataset))

    c = ChannelModel()
    c.link.build()
    c.link.fit(
        dt, ds, epochs=args.epochs, batch_size=args.batch,
        learning_rate=args.learning_rate
    )

    c.link.save()


def test_evaluate_link_model(args: argparse.Namespace):
    loader = DataLoader()
    _, dts = shuffle_and_split(
        loader.load(args.dataset), val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    model = ChannelModel()
    model.link.load()

    x_test, y_test = model.link._prepare_arrays(dts, fit=False)
    y_prob = model.link.model.predict(x_test, batch_size=args.batch)
    y_pred = np.argmax(y_prob, axis=1)

    print(classification_report(
        y_test, y_pred, target_names=["No-Link","NLOS","LOS"]
    ))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()

    # Per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_acc):
        print(f"Class {i} ({['No-Link', 'NLOS', 'LOS'][i]}) accuracy: {acc:.3f}")
    
    # Overall accuracy
    overall_acc = np.mean(y_pred == y_test)
    print(f"\nOverall Accuracy: {overall_acc:.3f}")



# ============================================================
#       Main Script
# ============================================================
SEED = [{"flags": ["--seed","-s"], "kwargs": {"type": int, "default": 42}}]
DATASET = [
    {"flags": ["dataset"], "kwargs": {"type": str}},
    {"flags": ["--val-ratio","-vr"], "kwargs": {"type": float, "default": 0.20}},
    {"flags": ["--test-ratio","-tr"], "kwargs": {"type": float, "default": 0.00}},
]
TRAIN = [
    {"flags": ["--batch","-b"], "kwargs": {"type": int, "default": 512}},
    {"flags": ["--epochs","-e"], "kwargs": {"type": int, "default": 25}},
    {"flags": ["--learning-rate", "-lr"], "kwargs": {"type": float, "default": 1e-3}}
]

@mainrunner
def main():
    p = build_parser([
        CommandSpec(
            name="build", help="Test build link state predictor",
            handler=test_build_link_model, args=[]
        ),
        CommandSpec(
            "train", help="Testing training the dataloader",
            handler=test_train_link_model, args=[*DATASET,*TRAIN,*SEED]
        ),
        CommandSpec(
            name="eval", help="Evaluate teh model",
            handler=test_evaluate_link_model, args=[*DATASET,*SEED,*TRAIN]
        )
    ])
    args = p.parse_args()
    args._handler(args)


if __name__ == "__main__":
    main()
