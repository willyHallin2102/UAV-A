"""
    debug/path.py
    .............
    Debugging CLI script for the path model and various generators, this 
    includes training the path model with a attached generative AI model. 
    In addition this script also including architecture testing as well 
    as an generating path test.
"""
from __future__ import annotations

import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import traceback
import time

import numpy as np
import tensorflow as tf
from typing import Any, Dict

from data.get_data import get_shuffled_city_data
from src.config.data import DataConfig, LinkState
from src.models.chanmod import ChannelModel

from logs.logger import Logger, LogLevel



def _create_test_path_data(
    n_samples: int = 1000, n_max_paths: int = 10, seed: int = 42
) -> Dict[str, np.ndarray]:
    """Create synthetic test data for path model."""
    np.random.seed(seed)

    # Generate basic link data
    dvec = np.random.uniform(-1000, 1000, (n_samples, 3))
    rx_type = np.random.choice(["Rx0", "Rx1"], n_samples)

    # Generate link states
    horizontal_distance = np.linalg.norm(dvec[:, :2], axis=1)
    altitude = dvec[:, 2]
    link_state = np.where(
        (altitude > 50) & (horizontal_distance < 200),
        LinkState.LOS, LinkState.NLOS
    )
    no_link_mask = (horizontal_distance > 800) | (altitude < -100)
    link_state[no_link_mask] = LinkState.NO_LINK

    # Generate path data (NLOS paths)
    nlos_pl = np.random.uniform(80, 120, (n_samples, n_max_paths))

    # AOA_PHI, AOA_THETA, AOD_PHI, AOD_THETA
    nlos_ang = np.random.uniform(0, 360, (n_samples, n_max_paths, 4))
    nlos_dly = np.random.uniform(1e-6, 1e-4, (n_samples, n_max_paths))

    # Sort paths by path loss (descending)
    sorted_indices = np.argsort(-nlos_pl, axis=1)
    nlos_pl = np.take_along_axis(nlos_pl, sorted_indices, axis=1)
    nlos_ang = np.take_along_axis(nlos_ang, sorted_indices[:, :, None], axis=1)
    nlos_dly = np.take_along_axis(nlos_dly, sorted_indices, axis=1)

    return {
        'dvec': dvec.astype(np.float32),
        'rx_type': rx_type,
        'link_state': link_state,
        'nlos_pl': nlos_pl.astype(np.float32),
        'nlos_ang': nlos_ang.astype(np.float32),
        'nlos_dly': nlos_dly.astype(np.float32)
    }



def test_data_generation(args: argparse.Namespace, logger: Logger):
    """Test path data generation and statistics."""
    logger.info("Testing path data generation...")

    # Test synthetic data
    synthetic_data = _create_test_path_data(
        n_samples=1000, n_max_paths=args.n_max_paths
    )

    logger.info("Synthetic path data shapes:")
    for key, value in synthetic_data.items():
        logger.info(f"  {key}: {value.shape}")

    # Analyze link state distribution
    unique, counts = np.unique(synthetic_data['link_state'], return_counts=True)
    logger.info("Link state distribution:")
    for state, count in zip(unique, counts):
        logger.info(
            f"  {LinkState(state).name}: {count} samples "
            f"({count/len(synthetic_data['link_state'])*100:.1f}%)"
        )
    
    # Analyze path statistics
    valid_mask = synthetic_data['link_state'] != LinkState.NO_LINK
    n_valid = np.sum(valid_mask)
    logger.info(
        f"Valid links (non NO_LINK): {n_valid}/{len(valid_mask)} "
        f"({n_valid/len(valid_mask)*100:.1f}%)"
    )
    
    if n_valid > 0:
        logger.info("Path statistics for valid links:")
        logger.info(
            f"\tPath loss range: [{synthetic_data['nlos_pl'][valid_mask].min():.1f}, "
            f"{synthetic_data['nlos_pl'][valid_mask].max():.1f}] dB"
        )
        logger.info(
            f"\tDelay range: [{synthetic_data['nlos_dly'][valid_mask].min():.2e}, "
            f"{synthetic_data['nlos_dly'][valid_mask].max():.2e}] s"
        )
        logger.info(
            f"\tAngle range: [{synthetic_data['nlos_ang'][valid_mask].min():.1f}, "
            f"{synthetic_data['nlos_ang'][valid_mask].max():.1f}]°"
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
        
        # Check if path data exists in real data
        required_keys = ['nlos_pl', 'nlos_ang', 'nlos_dly']
        missing_keys = [key for key in required_keys if key not in dtr]
        if missing_keys:
            logger.warning(f"Missing path data keys in real data: {missing_keys}")
        else:
            logger.info("All required path data keys present in real data")
            
    except Exception as e:
        logger.error(f"Failed to load real data: {e}")



def test_model_construction(args: argparse.Namespace, logger: Logger):
    """Test different path model configurations."""
    model_types = ["vae"]  # Add other types like "gan", "flow" if implemented
    architectures = [
        {"encoder_layers": (128, 64), "decoder_layers": (64, 128)},
        {"encoder_layers": (256, 128, 64), "decoder_layers": (64, 128, 256)},
        {"encoder_layers": (512, 256, 128, 64), "decoder_layers": (64, 128, 256, 512)}
    ]
    
    logger.info("Testing Path Model architectures and types")
    
    for model_type in model_types:
        for i, arch in enumerate(architectures):
            logger.info(f"\nTesting {model_type.upper()} with architecture {i+1}: {arch}")
            
            data_cfg = DataConfig(
                rx_types=["Rx0", "Rx1"], n_max_paths=args.n_max_paths,
                max_path_loss=args.max_path_loss
            )
            
            try:
                model = ChannelModel(
                    directory=f"test_path_{model_type}_{i}",
                    config=data_cfg, model_type=model_type, seed=args.seed
                )
                model.path.build()
                
                # Test model summary
                logger.info("Model summary:")
                model.path.model.model.summary(print_fn=lambda x: logger.debug(x))
                
                # Test forward pass with dummy data
                test_data = _create_test_path_data(n_samples=10, n_max_paths=args.n_max_paths)
                dataset = model.path._prepare_dataset(test_data, batch_size=5, fit=True)
                
                # Get one batch to test forward pass
                for x_batch, cond_batch in dataset.take(1):
                    output = model.path.model([x_batch, cond_batch], training=False)
                    logger.info(f"Forward pass successful - Output shapes:")
                    
                    for j, out in enumerate(output):
                        logger.info(f"  Output {j}: {out.shape}")
                    
                    break
                    
            except Exception as e:
                logger.error(f"Model {model_type} with architecture {arch} failed: {e}")
                logger.debug(traceback.format_exc())



def test_preprocessing(args: argparse.Namespace, logger: Logger):
    """Test data preprocessing pipeline."""
    logger.info("Testing data preprocessing pipeline...")
    
    try:
        model = ChannelModel(
            directory=args.cities,
            model_type=args.model_type,
            seed=args.seed
        )
        model.path.build()
        
        # Generate test data
        test_data = _create_test_path_data(n_samples=100, n_max_paths=args.n_max_paths)
        
        # Test preprocessing in fit mode
        logger.info("Testing preprocessing (fit mode)...")
        dataset_fit = model.path._prepare_dataset(test_data, batch_size=32, fit=True)
        
        # Test preprocessing in transform mode
        logger.info("Testing preprocessing (transform mode)...")
        dataset_transform = model.path._prepare_dataset(test_data, batch_size=32, fit=False)
        
        # Check dataset shapes
        for i, (x_batch, cond_batch) in enumerate(dataset_fit.take(1)):
            logger.info(f"Batch {i} - X shape: {x_batch.shape}, Conditions shape: {cond_batch.shape}")
            break
            
        # Test inverse transformations
        logger.info("Testing inverse transformations...")
        for x_batch, cond_batch in dataset_fit.take(1):
            # Get original dvec for this batch (first 5 samples)
            batch_dvec = test_data['dvec'][:5]
            batch_x = x_batch[:5]
            
            try:
                path_loss, angles, delays = model.path._inverse_transform_data(batch_dvec, batch_x)
                logger.info("Inverse transformation successful:")
                logger.info(f"  Reconstructed path loss shape: {path_loss.shape}")
                logger.info(f"  Reconstructed angles shape: {angles.shape}")
                logger.info(f"  Reconstructed delays shape: {delays.shape}")
            except Exception as e:
                logger.error(f"Inverse transformation failed: {e}")
            break
            
    except Exception as e:
        logger.error(f"Preprocessing test failed: {e}")
        logger.debug(traceback.format_exc())



def test_train(args: argparse.Namespace, logger: Logger):
    """Train the path model with comprehensive logging."""
    logger.info("Starting path model training...")
    
    try:
        # Load data
        dtr, dts = get_shuffled_city_data(
            cities=args.cities, validation_ratio=args.val_ratio
        )
        logger.info(
            f"Data loaded - Train: {len(dtr['dvec'])}, "
            f"Validation: {len(dts['dvec'])}"
        )
        
        # Check for required path data
        required_keys = ['nlos_pl', 'nlos_ang', 'nlos_dly']
        for key in required_keys:
            if key not in dtr: logger.warning(f"Missing key in training data: {key}")
            else:logger.info(f"Found {key} with shape {dtr[key].shape}")
        
        # Initialize model
        model = ChannelModel(
            directory=args.cities, model_type=args.model_type, seed=args.seed
        )
        
        # Build or load model
        if args.load_existing:
            logger.info("Loading existing model...")
            model.path.load()
        else:
            logger.info("Building new model...")
            model.path.build()
        
        # Train model
        history = model.path.fit(
            dtr=dtr, dts=dts, epochs=args.epochs, 
            batch_size=args.batch_size, learning_rate=args.learning_rate
        )
        
        # Log training results
        if hasattr(history, 'history'):
            final_loss = history.history.get('loss', [-1])[-1]
            final_val_loss = history.history.get('val_loss', [-1])[-1]
            logger.info(
                f"Training completed - Final Loss: {final_loss:.4f}, "
                f"Final Val Loss: {final_val_loss:.4f}"
            )
        
        # Save model
        model.path.save()
        logger.info(f"Model saved to {model.path.directory}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.debug(traceback.format_exc())
        raise



def test_sampling(args: argparse.Namespace, logger: Logger):
    """Test sampling/generation from trained path model."""
    logger.info("Testing path sampling/generation...")
    
    try:
        # Load trained model
        model = ChannelModel(
            directory=args.cities,
            model_type=args.model_type
        )
        model.path.load()
        logger.info("Model loaded successfully")

        # logger.debug(f"Model expects rx_types: {model.path.rx_types}")
        # logger.debug(f"OneHotEncoder categories: {model.path.rx_encoder.categories_}")
        
        # Create test conditions
        n_samples = 10
        test_conditions = {
            'dvec': np.random.uniform(-500, 500, (n_samples, 3)).astype(np.float32),
            # 'rx_type': np.random.choice(["Rx0", "Rx1"], n_samples),
            'rx_type': np.random.choice(model.path.rx_types, n_samples),
            'link_state': np.full(n_samples, LinkState.LOS)  # Assume LOS for testing
        }
        
        # Prepare conditions for sampling
        valid_mask = test_conditions['link_state'] != LinkState.NO_LINK
        idx = np.flatnonzero(valid_mask)
        
        if len(idx) == 0:
            logger.warning("No valid links for sampling")
            return
            
        dvec = test_conditions['dvec'][idx]
        rx = test_conditions['rx_type'][idx]
        los = (test_conditions['link_state'][idx] == LinkState.LOS).astype(np.float32)
        
        conditions = model.path._transform_conditions(dvec, rx, los, fit=False)
        
        # Test sampling (if implemented in the model)
        if hasattr(model.path.model, 'sample'):
            logger.info("Testing direct sampling from model...")
            try:
                # Generate random latent codes
                n_latent = model.path.model.n_latent
                z = tf.random.normal((len(conditions), n_latent))
                
                # Sample from decoder
                samples = model.path.model.decoder([z, conditions], training=False)
                logger.info(f"Sampling successful - Generated {len(samples[0])} samples")
                
                # Inverse transform to get physical parameters
                path_loss, angles, delays = model.path._inverse_transform_data(dvec, samples[0])
                logger.info("Generated path statistics:")
                logger.info(f"  Path loss: [{path_loss.min():.1f}, {path_loss.max():.1f}] dB")
                logger.info(f"  Delays: [{delays.min():.2e}, {delays.max():.2e}] s")
                
            except Exception as e:
                logger.warning(f"Direct sampling not fully implemented: {e}")
        
        # Test reconstruction
        logger.info("Testing reconstruction...")
        test_data = _create_test_path_data(n_samples=5, n_max_paths=args.n_max_paths)
        dataset = model.path._prepare_dataset(test_data, batch_size=5, fit=False)
        
        for x_batch, cond_batch in dataset.take(1):
            reconstructions = model.path.model([x_batch, cond_batch], training=False)
            logger.info(f"Reconstruction successful - Input: {x_batch.shape}, Output: {reconstructions[0].shape}")
            break
            
    except Exception as e:
        logger.error(f"Sampling test failed: {e}")
        logger.debug(traceback.format_exc())



def test_model_persistence(args: argparse.Namespace, logger: Logger):
    """Test saving and loading path model."""
    logger.info("Testing path model persistence...")
    
    try:
        test_dir = "persistence_test_path"
        
        # Create and train a small model briefly
        data_cfg = DataConfig(
            rx_types=["0", "1"],
            n_max_paths=args.n_max_paths,
            max_path_loss=args.max_path_loss
        )
        
        model = ChannelModel(
            directory=test_dir,
            config=data_cfg,
            model_type=args.model_type,
            seed=args.seed
        )
        model.path.build()
        
        # Train briefly with synthetic data
        test_data = _create_test_path_data(n_samples=50, n_max_paths=args.n_max_paths)
        model.path.fit(dtr=test_data, dts=test_data, epochs=2, batch_size=16)
        
        # Save model
        model.path.save()
        logger.info("Model saved")
        
        # Create new instance and load
        model_loaded = ChannelModel(
            directory=test_dir,
            config=data_cfg,
            model_type=args.model_type
        )
        model_loaded.path.load()
        logger.info("Model loaded successfully")
        
        # Verify predictions match
        test_batch = _create_test_path_data(n_samples=5, n_max_paths=args.n_max_paths)
        dataset = model.path._prepare_dataset(test_batch, batch_size=5, fit=False)
        
        for x_batch, cond_batch in dataset:
            original_output = model.path.model([x_batch, cond_batch], training=False)
            loaded_output = model_loaded.path.model([x_batch, cond_batch], training=False)
            
            # Compare outputs
            max_diff = max(
                tf.reduce_max(tf.abs(orig - loaded)).numpy()
                for orig, loaded in zip(original_output, loaded_output)
            )
            
            logger.info(f"Model persistence test - Max output difference: {max_diff:.6f}")
            
            if max_diff < 1e-6:
                logger.info("✓ Model persistence test PASSED")
            else:
                logger.warning("Model persistence test shows differences")
            break
            
    except Exception as e:
        logger.error(f"Model persistence test failed: {e}")
        logger.debug(traceback.format_exc())



def performance_benchmark(args: argparse.Namespace, logger: Logger):
    """Benchmark path model performance."""
    logger.info("Running path model performance benchmark...")
    
    try:
        model = ChannelModel(directory=args.cities, model_type=args.model_type)
        model.path.load()
        
        # Create large dataset for benchmarking
        large_data = _create_test_path_data(n_samples=1000, n_max_paths=args.n_max_paths)
        dataset = model.path._prepare_dataset(large_data, batch_size=args.batch_size, fit=False)
        
        start_time = time.time()
        
        # Benchmark forward pass speed
        n_batches = 0
        for x_batch, cond_batch in dataset:
            _ = model.path.model([x_batch, cond_batch], training=False)
            n_batches += 1
            if n_batches >= 10:  # Test with 10 batches
                break
                
        end_time = time.time()
        duration = end_time - start_time
        samples_per_second = (n_batches * args.batch_size) / duration
        
        logger.info("Performance benchmark:")
        logger.info(f"  Processed {n_batches * args.batch_size} samples in {duration:.2f} seconds")
        logger.info(f"  Throughput: {samples_per_second:.2f} samples/second")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Model type: {args.model_type}")
        
    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")



def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for path model debugging."""
    parser = argparse.ArgumentParser(description="Extended CLI tester for path model")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    def add_common_args(subparser: argparse.ArgumentParser):
        """Add common arguments to subparsers."""
        subparser.add_argument(
            "--loglevel", type=str, default="INFO", 
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Log level assignment"
        )
        subparser.add_argument(
            "--cities", type=str, default="beijing",
            choices=["beijing", "boston", "moscow", "london", "tokyo"],
            help="Which city to train on"
        )
        subparser.add_argument(
            "--model-type", type=str, default="vae",
            choices=["vae"],
            help="Type of generative model to use"
        )
        subparser.add_argument("--epochs", type=int, default=10)
        subparser.add_argument("--batch-size", type=int, default=512)
        subparser.add_argument("--learning-rate", type=float, default=1e-4)
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument("--val-ratio", type=float, default=0.10)
        subparser.add_argument("--n-max-paths", type=int, default=10,
                             help="Maximum number of paths to model")
        subparser.add_argument("--max-path-loss", type=float, default=150.0,
                             help="Maximum path loss value for normalization")
        subparser.add_argument("--load-existing", action="store_true",
                             help="Load existing model instead of building new")

    # Define commands
    commands = [
        ("train", "Train the path model"),
        ("sampling", "Test sampling/generation capabilities"),
        #
        ("test_data", "Test path data generation and loading"),
        ("test_arch", "Test different model architectures"),
        ("test_preproc", "Test data preprocessing pipeline"),
        ("test_persistence", "Test model saving and loading"),
        ("benchmark", "Run performance benchmarks")
    ]
    
    for cmd, help_text in commands:
        add_common_args(subparsers.add_parser(cmd, help=help_text))
    
    return parser



def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Set up logger
    loglevel = LogLevel[args.loglevel]
    logger = Logger("path-cli", to_disk=False, level=loglevel)
    
    # Command mapping
    commands = {
        "train": test_train,
        "sampling": test_sampling,
        #
        "test_data": test_data_generation,
        "test_arch": test_model_construction,
        "test_preproc": test_preprocessing,
        "test_persistence": test_model_persistence,
        "benchmark": performance_benchmark
    }

    try:
        logger.info(f"Executing command: {args.command}")
        commands[args.command](args, logger)
        logger.info("Command completed successfully")

    except Exception as e:
        logger.error(f"Command failed: {e}")
        logger.debug(traceback.format_exc())
        raise
    
    finally: Logger.shutdown_all()




if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        Logger.shutdown_all()
        sys.exit(0)

