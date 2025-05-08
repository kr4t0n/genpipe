import os
import argparse
import dask.dataframe as dd

from genpipe.schema import schema

args_parser = argparse.ArgumentParser()
args_parser.add_argument("--src", type=str, nargs="+")
args_parser.add_argument("--dst", type=str)
args = args_parser.parse_args()


def main():
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    name = args.dst.split("/")[-1]
    log_file = os.path.join(log_dir, f"{name}.txt")

    with open(log_file, "w") as f:
        data = []
        for fname in args.src:
            temp = dd.read_parquet(fname)
            data.append(temp)
            f.write(f"mix {len(temp)} data from {fname}\n")

        df = dd.concat(data).repartition(npartitions=1).sample(frac=1).repartition(npartitions=10)
        f.write(f"total {len(df)} rows to {args.dst}\n")

    df.to_parquet(args.dst, write_index=False, schema=schema)


if __name__ == "__main__":
    main()
