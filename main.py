import os
import argparse
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

from genpipe.distill import distill
from genpipe.merge import merge
from genpipe.transform import transform

load_dotenv()

args_parser = argparse.ArgumentParser()
args_parser.add_argument("--dir", type=str)
args_parser.add_argument("--model", type=str)
args_parser.add_argument("--thinking", action="store_true")
args_parser.add_argument("--budget_tokens", type=int, default=4 * 1000)
args_parser.add_argument("--max_tokens", type=int, default=32 * 1000)
args_parser.add_argument("--num_workers", type=int, default=32)
args = args_parser.parse_args()


def main():
    # check working directory
    assert "meta.parquet" in os.listdir(args.dir)
    assert "template.txt" in os.listdir(args.dir)

    # read metadata
    df = pd.read_parquet(os.path.join(args.dir, "meta.parquet"))
    if os.path.exists(os.path.join(args.dir, "system.txt")):
        with open(os.path.join(args.dir, "system.txt")) as f:
            system = f.read()
    else:
        system = None
    with open(os.path.join(args.dir, "template.txt")) as f:
        template = f.read()

    # distill data
    distill(
        llm=OpenAI(),
        model=args.model,
        system=system,
        template=template,
        df=df,
        dst=os.path.join(args.dir, "output"),
        thinking=args.thinking,
        budget_tokens=args.budget_tokens,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
    )
    # merge data
    merge(
        os.path.join(args.dir, "output"),
        os.path.join(args.dir, "merged"),
    )


if __name__ == "__main__":
    main()
