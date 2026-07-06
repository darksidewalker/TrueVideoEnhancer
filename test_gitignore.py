#!/usr/bin/env python3
"""Test script to validate .gitignore rules work correctly.

Creates temporary files inside the repository to verify they're properly ignored,
then cleans them up. Run this during development to catch accidental commits
of unwanted files.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_git_check_ignore(filepath: str) -> bool:
    """Check whether git would ignore a path."""
    result = subprocess.run(
        ["git", "check-ignore", "-v", filepath],
        capture_output=True, text=True
    )
    return result.returncode == 0


def main():
    repo_root = Path(__file__).parent.resolve()
    os.chdir(repo_root)

    # Files to create temporarily for testing
    test_files = [
        "test_preview.png",
        "test_photo.jpg",
        "test_image.jpeg",
        "test_artifact.webp",
        "scratch.tmp",
        "old_file.bak",
        "config.lock",
        "notebook.ipynb",
        "settings.env",
        ".DS_Store",
    ]

    print("=" * 60)
    print("Testing .gitignore rules")
    print("=" * 60)
    print()

    failures = []
    for fname in test_files:
        fpath = repo_root / fname
        # Create the test file
        fpath.write_text(f"ignored by gitignore\n")

        ignored = run_git_check_ignore(str(fpath))
        status = "PASS" if ignored else "FAIL"
        print(f"  [{status}] {fname}")
        if not ignored:
            failures.append(fname)

        # Always clean up
        fpath.unlink(missing_ok=True)

    print()
    print("-" * 60)
    if failures:
        print(f"FAILED: {len(failures)} pattern(s) not ignored:")
        for f in failures:
            print(f"  - {f}")
        print()
        print("Review .gitignore and ensure these patterns are covered.")
        return 1
    else:
        print("All patterns correctly ignored! .gitignore looks good.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
