"""Compile the report from its Typst sources.

    python scripts/build_report.py --video https://...   # submission build
    python scripts/build_report.py                       # proofreading build

`reports/report/report.typ` includes one file per section from
`reports/report/sections/`, and resolves figure paths against the repository root,
so the root is passed to the compiler rather than the report directory.

The video URL reaches the document through Typst's `sys.inputs` instead of being
typed into the bibliography. Reference [28] is emitted only when a URL is supplied,
which is why the reference list cannot go out carrying a note to self in place of a
link. `make report` refuses to run without one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "reports" / "report" / "report.typ"
OUTPUT = REPO / "reports" / "report" / "report.pdf"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--video",
        default="",
        help="URL of the uploaded video presentation; omit for a proofreading build",
    )
    p.add_argument("--out", type=Path, default=OUTPUT)
    args = p.parse_args()

    try:
        import typst
    except ImportError as exc:  # pragma: no cover - environment, not code
        raise SystemExit(
            "typst is not installed in this environment. It is pinned in "
            "requirements-lock.txt; install it with `pip install typst`."
        ) from exc

    video = args.video.strip()
    if video and not video.startswith(("http://", "https://")):
        raise SystemExit(f"--video must be an http(s) URL, got {video!r}")

    typst.compile(
        str(SOURCE),
        output=str(args.out),
        root=str(REPO),
        sys_inputs={"video": video},
    )

    if video:
        print(f"wrote {args.out} citing the video at {video}")
        return

    print(f"wrote {args.out}")
    print(
        "\nPROOFREADING BUILD. Reference [28] is absent, so this copy does not carry "
        "the\nvideo link the brief requires in the reference list. Before submitting:"
        "\n    make report VIDEO_URL=https://...",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
