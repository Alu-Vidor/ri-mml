from __future__ import annotations

from .reporting import write_report_files
from .utils import ensure_directories


def main() -> None:
    ensure_directories()
    write_report_files()


if __name__ == "__main__":
    main()
