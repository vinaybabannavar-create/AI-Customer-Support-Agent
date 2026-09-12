"""
One-command reproduction of the headline pipeline + results.
See README.md for the full step-by-step; this just chains the same steps.
"""
import subprocess
import sys


def run(cmd):
    print(f"\n$ {cmd}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    run("python3 -m src.data_prep")
    run("python3 -m eval.build_golden_set --n 200")
    run("python3 -m eval.make_calibration_drafts")
    run("python3 -m eval.human_agreement")
    run("python3 -m eval.run_eval")
    print("\nDone. See outputs/eval_summary.md and outputs/judge_human_agreement.json")


if __name__ == "__main__":
    main()
