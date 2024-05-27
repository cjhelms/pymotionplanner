"""Simple driver to run all search algorithms for quick sanity check"""

import subprocess


def main() -> None:
    for a in ("breadth-first", "depth-first", "dijkstra", "astar"):
        subprocess.run(
            [
                "python3",
                "./example_discrete.py",
                "-a",
                f"{a}",
                "-r",
                "-o",
                "run_all_results",
            ]
        )


if __name__ == "__main__":
    main()
