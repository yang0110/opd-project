"""
Generate the full comparison report: tables + figures.

Run this after all experiments complete to produce the final output.
Calls collect_results.py and plot_curves.py, then generates a summary report.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.analysis.collect_results import main as collect_main
from experiments.analysis.plot_curves import main as plot_main


def main():
    print("=" * 60)
    print("OPD METHOD COMPARISON: FINAL REPORT GENERATION")
    print("=" * 60)

    print("\n[1/2] Collecting results and generating tables...")
    collect_main()

    print("\n[2/2] Generating figures...")
    plot_main()

    summary_dir = os.path.join(PROJECT_ROOT, "experiments", "results", "summary")
    print("\n" + "=" * 60)
    print("DONE. Outputs:")
    print(f"  Table: {summary_dir}/comparison_table.md")
    print(f"  Data:  {summary_dir}/comparison_table.json")
    print(f"  Figs:  {summary_dir}/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
