import argparse

from benchmark_vocabulous import bench_sentence_expansion


def main():
    parser = argparse.ArgumentParser(
        description="Run sentence expansion benchmark with configurable parameters."
    )
    parser.add_argument("--rows", type=int, default=200000, help="Number of rows")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Sentence chunk size for expansion",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of sentence expansion workers to test",
    )
    args = parser.parse_args()

    bench_sentence_expansion(
        n_rows=args.rows,
        worker_options=tuple(args.workers),
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
