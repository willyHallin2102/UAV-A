"""
    debug/link.py
    -------------
    Extends a cli script for debugging and testing the link model, this model is 
    being reused for any other model, city or generative model.
"""
from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from typing import Any, Tuple

from data.get_data import get_shuffled_city_data
from src.config.data import LinkState, DataConfig
from src.models.chanmod import ChannelModel

from logs.logger import Logger, LogLevel




def _create_test_data(n_samples: int=1000, seed: int=42) -> Dict[str, np.ndarray]:
    """
    """
    np.random.seed(seed)

    dvec = np.random.uniform(-1000, 1000, (n_samples, 3))
    rx_type = np.random.choice(["0", "1"], n_samples)

    # Generate link states using a simple heuristic
    horizontal_distance = np.linalg.norm(dvec[:, :2], axis=1)
    altitude = dvec[:, 2]

    # Simple rule: if altitude > 50 and horizontal_dist < 200, likely LoS
    link_state = np.where(
        (altitude > 50) & (horizontal_distance < 200),
        LinkState.LOS, LinkState.NLOS
    )

    # Add some NO_LINK states for extreme conditions
    no_link_mask = (horizontal_distance > 800) | (altitude < -100)
    link_state[no_link_mask] = LinkState.NO_LINK
    
    return {
        'dvec': dvec.astype(np.float32),
        'rx_type': rx_type,
        'link_state': link_state
    }



def test_data_generation(args: argparse.Namespace, logger: Logger):
    """Test data generation and basic statistics."""
    logger.info("Testing data generation...")
    
    # Test synthetic data
    synthetic_data = _create_test_data(n_samples=1000)
    logger.info(f"Synthetic data shapes - dvec: {synthetic_data['dvec'].shape}, "
                f"rx_type: {synthetic_data['rx_type'].shape}, "
                f"link_state: {synthetic_data['link_state'].shape}")
    
    # Analyze class distribution
    unique, counts = np.unique(synthetic_data['link_state'], return_counts=True)
    logger.info("Class distribution in synthetic data:")
    for state, count in zip(unique, counts):
        logger.info(
            f"{LinkState(state).name}: {count} samples "
            f"({count/len(synthetic_data['link_state'])*100:.1f}%)"
        )
    
    # Test real data loading
    try:
        dtr, dts = get_shuffled_city_data(
            cities=args.cities, validation_ratio=args.val_ratio
        )
        logger.info(
            f"Real data loaded - Training: {len(dtr['dvec'])}, "
            f"Test: {len(dts['dvec'])}"
        )
    except Exception as e:
        logger.error(f"Failed to load real data: {e}")



def test_model_construction(args: argparse.Namespace, logger: Logger):
    architectures = [(64, 32), (128, 64, 32), (256, 128, 64, 32)]
    logger.info("Testing Link State Predictor architectures")
    
    for i, architecture in enumerate(architectures):
        logger.info(f"\nTesting architecture `{i+1}`:\t{architecture}")
        data_cfg = DataConfig(
            rx_types=["0", "1"], n_unit_links=architecture,
            dropout_rate=0.20, add_zero_los_frac=0.10
        )

        try:
            model = ChannelModel(
                directory=f"test_arch_{i}", config=data_cfg,
                seed=args.seed
            )
            model.link.build()

            # Test model summary
            logger.info("Model summary:")
            model.link.model.summary(print_fn=lambda x: logger.debug(x))
            
            # Test forward pass with dummy data
            test_data = _create_test_data(n_samples=10)
            x_test, _ = model.link._prepare_arrays(test_data, fit=True)
            output = model.link.model.predict(x_test, verbose=0)
            logger.info(f"Forward pass successful - Output shape: {output.shape}")
        
        except Exception as e:
            logger.error(f"Architecture `{architecture}` failed:\t{e}")



def train_link(args: argparse.Namespace, logger: Logger):
    """Train the link state predictor with comprehensive logging."""
    logger.info("Starting link model training...")
    
    try:
        # Load data
        dtr, dts = get_shuffled_city_data(
            cities=args.cities, validation_ratio=args.val_ratio
        )
        logger.info(f"Data loaded - Train: {len(dtr['dvec'])}, Validation: {len(dts['dvec'])}")
        
        # Initialize and train model
        model = ChannelModel(directory=args.cities, seed=args.seed)
        model.link.build()
        
        history = model.link.fit(
            dtr=dtr, dts=dts, epochs=args.epochs, 
            batch_size=args.batch_size, learning_rate=args.learning_rate
        )
        
        # Log training results
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        logger.info(f"Training completed - Final Train Acc: {final_train_acc:.4f}, "
                   f"Final Val Acc: {final_val_acc:.4f}")
        
        # Save model
        model.link.save()
        logger.info(f"Model saved to {model.link.directory}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise



def test_predictions(args: argparse.Namespace, logger: Logger):
    """Test model predictions and sampling."""
    logger.info("Testing model predictions...")
    
    try:
        # Load trained model
        model = ChannelModel(directory=args.cities)
        model.link.load()
        logger.info("Model loaded successfully")
        
        # Create test cases covering different scenarios
        test_cases = [
            # [x, y, z], rx_type, description
            ([50, 50, 100], "0", "Medium distance, high altitude"),
            ([10, 10, 20], "1", "Close distance, low altitude"), 
            ([200, 200, 10], "0", "Far distance, very low altitude"),
            ([0, 0, 150], "1", "Directly above, high altitude"),
            ([500, 500, 5], "0", "Very far, ground level")
        ]
        
        test_dvec = np.array([case[0] for case in test_cases], dtype=np.float32)
        test_rx_type = np.array([case[1] for case in test_cases])
        
        # Get predictions
        probabilities = model.link.predict(test_dvec, test_rx_type)
        
        logger.info("Prediction results:")
        for i, (_, _, desc) in enumerate(test_cases):
            probs = probabilities[i]
            predicted_state = np.argmax(probs)
            logger.info(f"  {desc}:")
            logger.info(f"    Probabilities - NLoS: {probs[0]:.3f}, LoS: {probs[1]:.3f}, No-Link: {probs[2]:.3f}")
            logger.info(f"    Predicted: {LinkState(predicted_state).name}")
        
        # Test batch prediction
        large_test_data = _create_test_data(n_samples=100)
        large_probs = model.link.predict(
            large_test_data['dvec'], 
            large_test_data['rx_type']
        )
        logger.info(f"Batch prediction successful - Shape: {large_probs.shape}")
        
    except Exception as e:
        logger.error(f"Prediction test failed: {e}")
        raise




# ---------------========== Building Parser ==========--------------- #

def build_parser() -> argparse.ArgumentParser:
    """
    """
    parser = argparse.ArgumentParser(description="CLI tester for link state predictor")
    subparser = parser.add_subparsers(dest="command", required=True, help="Command to run")

    def add_common(p: argparse.ArgumentParser):
        p.add_argument(
            "--loglevel", type=str, default="INFO", help="loglevel assignment",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )

        p.add_argument(
            "--cities", type=str, default="beijing", help="which city to train on",
            choices=["beijing", "boston", "moscow", "london", "tokyo"]
        )

        p.add_argument("--epochs", type=int, default=10)
        p.add_argument("--batch-size", type=int, default=512)
        p.add_argument("--learning-rate", type=float, default=1e-4)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--val-ratio", type=float, default=0.10)
    
    for cmd, help_text in [
        ("train", "Train the link state predictor"),
        ("test_data", "Test data generation and loading"),
        ("test_arch", "Test different model architectures"),
        ("test_pred", "Test prediction capabilities")
    ]: add_common(subparser.add_parser(cmd, help=help_text))
    
    return parser



def main():
    parser = build_parser()
    args = parser.parse_args()

    logger = Logger("link-cli", to_disk=False, level=LogLevel.INFO)
    

    commands = {
        "train": train_link,
        "test_data": test_data_generation, 
        "test_arch": test_model_construction,
        "test_pred": test_predictions,
    }

    try: commands[args.command](args, logger)
    finally: Logger.shutdown_all()



if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        Logger.shutdown_all()
        sys.exit(0)
