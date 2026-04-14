#!/usr/bin/env python3
"""Tidus AI Model Latest Pricing Report — CLI generator.

Usage:
    uv run python scripts/generate_pricing_report.py [--output-dir reports/] [--gh-release]

Generates a markdown pricing report comparing the current active revision
to the previous superseded revision. Optionally creates a GitHub Release.

Arguments:
    --output-dir   Directory to write the report file (default: reports/)
    --revision     Specific revision ID to report on (default: current ACTIVE)
    --base         Specific base revision to compare against (default: last SUPERSEDED)
    --gh-release   Create a GitHub Release after generating the report
    --tag          Git tag for the GitHub Release (default: pricing-YYYY-MM-DD)
    --dry-run      Print report to stdout without writing files
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main(args: argparse.Namespace) -> int:
    from tidus.db.engine import create_tables, get_session_factory
    from tidus.reporting.pricing_report import PricingReportGenerator

    await create_tables()
    sf = get_session_factory()

    print("Generating pricing report…")
    generator = PricingReportGenerator(sf)
    report = await generator.generate(
        revision_id=args.revision or None,
        base_revision_id=args.base or None,
    )

    # Determine output path
    output_dir = Path(args.output_dir)
    report_filename = f"pricing-{report.report_date}.md"
    report_path = output_dir / report_filename

    if args.dry_run:
        print(report.markdown)
        return 0

    # Write report file
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.markdown, encoding="utf-8")
    print(f"Report written: {report_path}")

    # Optionally save HTML for browser preview
    if args.save_html:
        html_path = output_dir / f"pricing-{report.report_date}.html"
        html_path.write_text(report.html, encoding="utf-8")
        print(f"HTML written:   {html_path}")

    # Print summary
    n_changes = len({c.model_id for c in report.price_changes})
    n_new = len(report.new_models)
    print(f"  New models:     {n_new}")
    print(f"  Price changes:  {n_changes} models")
    print(f"  Stale models:   {len(report.stale_models)}")

    if report.price_changes:
        print("\nTop price moves:")
        seen: set[str] = set()
        for change in report.price_changes[:6]:
            if change.model_id in seen:
                continue
            seen.add(change.model_id)
            sign = "+" if change.delta_pct > 0 else ""
            print(f"  {change.model_id:<30} {change.emoji} {sign}{change.delta_pct:.1f}%")

    # Optionally create GitHub Release
    if args.gh_release:
        tag = args.tag or f"pricing-{report.report_date}"
        title = f"Tidus Pricing Update — {report.report_date}"
        print(f"\nCreating GitHub Release: {tag}")
        _create_github_release(tag, title, report.github_release_body, report_path)

    return 0


def _create_github_release(
    tag: str,
    title: str,
    body: str,
    report_path: Path,
) -> None:
    """Create a GitHub Release using gh CLI."""
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  Warning: gh CLI not found — skipping GitHub Release creation.")
        print("  Install: https://cli.github.com")
        return

    # Write release body to temp file to avoid shell escaping issues
    body_file = Path("/tmp/release_body.md")
    body_file.write_text(body, encoding="utf-8")

    cmd = [
        "gh", "release", "create", tag,
        "--title", title,
        "--notes-file", str(body_file),
        "--latest=false",  # don't make pricing updates the "latest" code release
        str(report_path),  # attach the report as a release asset
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  GitHub Release created: {result.stdout.strip()}")
    else:
        print(f"  GitHub Release failed: {result.stderr.strip()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Tidus AI pricing report")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--revision", default="", help="Revision ID (default: active)")
    parser.add_argument("--base", default="", help="Base revision ID for comparison")
    parser.add_argument("--gh-release", action="store_true", help="Create GitHub Release")
    parser.add_argument("--tag", default="", help="Git tag for release")
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout only")
    parser.add_argument("--save-html", action="store_true", help="Also save HTML version")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))
