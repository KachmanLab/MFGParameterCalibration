#!/usr/bin/env python3
"""
Master runner for all bike-sharing experiments.

Usage:
  # Run all tests sequentially:
  python run_all.py

  # Run a specific test:
  python run_all.py --test 1
  python run_all.py --test 3
  python run_all.py --test 4
  python run_all.py --test 5

  # Run tests 3 through 5:
  python run_all.py --test 3 4 5

Test descriptions:
  test1  Constant gamma (baseline vs MFG with scalar cost parameters)
  test2  Linear gamma (baseline vs MFG with scalar cost parameters)
  test3  Linear gamma + neural-network cost (baseline vs MFG)
  test4  Intervention scenario 1: Station 1 (Broadway & E 14 St) mid-day closure
  test5  Intervention scenario 2: Station 2 (8 Ave & W 31 St) morning rush closure

Tests 4 and 5 require test3 to have been run first (they load saved parameters).
If no test3 results are found, they will train from scratch (slower).
"""

import os
import sys
import subprocess
import argparse
import glob
from datetime import datetime


BIKESHARE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BIKESHARE_DIR, "results")
VENV_ACTIVATE = os.path.join(BIKESHARE_DIR, "..", "venv_jax", "bin", "activate")


def find_latest_test3_dir():
    """Find the most recent test3 results directory that has saved parameters."""
    # Match both old naming (run_test3_*) and new naming (run_test3_nncost_*)
    patterns = [
        os.path.join(RESULTS_DIR, "run_test3_nncost_*"),
        os.path.join(RESULTS_DIR, "run_test3_*"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    candidates = sorted(set(candidates), reverse=True)
    for d in candidates:
        if os.path.exists(os.path.join(d, "mfg", "params_seed0.pkl")):
            return d
    return None


def run_python(script_args, label):
    """Run a Python script in the bikeshare directory."""
    print("\n" + "=" * 70)
    print(f"  RUNNING: {label}")
    print("=" * 70 + "\n")

    cmd = ["python3"] + script_args
    result = subprocess.run(cmd, cwd=BIKESHARE_DIR)

    if result.returncode != 0:
        print(f"\n*** {label} FAILED with exit code {result.returncode} ***")
        return False
    else:
        print(f"\n*** {label} COMPLETED SUCCESSFULLY ***")
        return True


def run_test1():
    return run_python(
        ["train_bikeshare.py", "--complexity", "constant"],
        "Test 1: Constant Gamma (Baseline vs MFG)"
    )


def run_test2():
    return run_python(
        ["train_bikeshare.py", "--complexity", "linear"],
        "Test 2: Linear Gamma (Baseline vs MFG)"
    )


def run_test3():
    return run_python(
        ["train_test3.py"],
        "Test 3: Linear Gamma + Neural-Net Cost (Baseline vs MFG)"
    )


def run_test4():
    test3_dir = find_latest_test3_dir()
    args = ["run_intervention.py", "--scenario", "1"]
    if test3_dir is not None:
        print(f"  Using test3 params from: {test3_dir}")
        args += ["--test3-dir", test3_dir]
    else:
        print("  WARNING: No test3 params found. Will train from scratch.")
    return run_python(args, "Test 4: Intervention 1 (Station 1, mid-day closure)")


def run_test5():
    test3_dir = find_latest_test3_dir()
    args = ["run_intervention.py", "--scenario", "2"]
    if test3_dir is not None:
        print(f"  Using test3 params from: {test3_dir}")
        args += ["--test3-dir", test3_dir]
    else:
        print("  WARNING: No test3 params found. Will train from scratch.")
    return run_python(args, "Test 5: Intervention 2 (Station 2, morning rush closure)")


TESTS = {
    1: run_test1,
    2: run_test2,
    3: run_test3,
    4: run_test4,
    5: run_test5,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run bike-sharing experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test", type=int, nargs="*", default=None,
        help="Test numbers to run (1-5). Default: run all."
    )
    args = parser.parse_args()

    if args.test is None:
        tests_to_run = [1, 2, 3, 4, 5]
    else:
        tests_to_run = args.test

    print("=" * 70)
    print(f"  Bike-Share Experiment Suite")
    print(f"  Tests to run: {tests_to_run}")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}
    for t in tests_to_run:
        if t not in TESTS:
            print(f"\n*** Unknown test number: {t}. Skipping. ***")
            continue
        ok = TESTS[t]()
        results[t] = ok

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for t in tests_to_run:
        status = "OK" if results.get(t, False) else "FAILED"
        print(f"  Test {t}: {status}")
    print("=" * 70)
    print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
