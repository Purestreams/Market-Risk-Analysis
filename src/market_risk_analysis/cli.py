from __future__ import annotations

import argparse

from .pipeline import run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bitcoin market risk analysis pipeline")
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download fresh hourly BTC/USD proxy data from Binance instead of reusing cached CSV files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_analysis(refresh_data=args.refresh_data)
    print(f"Report written to {summary['report_markdown_path']}")
    print(f"TeX paper written to {summary['report_tex_path']}")
    print(f"TODO written to {summary['report_todo_path']}")
    print(f"Tables written to {summary['tables_dir']}")
    print(f"Figures written to {summary['figures_dir']}")
    print(f"Results written to {summary['results_path']}")