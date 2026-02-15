#!/usr/bin/env python3
"""
Train All — Run complete DreamerV3 training pipeline
=====================================================
Trains world model demo + all environments + generates visualizations.
"""

import subprocess
import sys
import time


def run_script(name, path):
    """Run a training script and report results."""
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run([sys.executable, path],
                          capture_output=False, text=True)
    elapsed = time.time() - start

    status = "✓" if result.returncode == 0 else "✗"
    print(f"\n{status} {name} — {elapsed:.1f}s")
    return result.returncode == 0


def main():
    print("=" * 60)
    print("  DreamerV3 From Scratch — Full Training Pipeline")
    print("=" * 60)

    scripts = [
        ("World Model Demo", "01_world_model_demo.py"),
        ("CartPole-v1 (Discrete)", "02_dreamer_cartpole.py"),
        ("Pendulum-v1 (Continuous)", "03_dreamer_pendulum.py"),
        ("Acrobot-v1 (Sparse Reward)", "05_dreamer_acrobot.py"),
        ("LunarLander-v3 (Complex)", "06_dreamer_lunar_lander.py"),
        ("Visualizations", "04_visualize.py"),
    ]

    results = {}
    for name, path in scripts:
        results[name] = run_script(name, path)

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}  {name}")

    total = sum(results.values())
    print(f"\n  {total}/{len(results)} completed successfully")


if __name__ == '__main__':
    main()
