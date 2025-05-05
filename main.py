import os
import argparse
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

from genpipe.distill import distill
from genpipe.merge import merge

load_dotenv()

args_parser = argparse.ArgumentParser()
args_parser.add_argument("--dir", type=str)
args_parser.add_argument("--model", type=str)
args_parser.add_argument("--thinking", action="store_true")
args_parser.add_argument("--budget_tokens", type=int, default=4 * 1000)
args_parser.add_argument("--max_tokens", type=int, default=32 * 1000)
args = args_parser.parse_args()


def main():
    # check working directory
    assert "meta.csv" in os.listdir(args.dir)
    assert "template.txt" in os.listdir(args.dir)

    # read metadata
    df = pd.read_csv(os.path.join(args.dir, "meta.csv"))
    with open(os.path.join(args.dir, "template.txt")) as f:
        template = f.read()

    # distill data
    distill(
        llm=OpenAI(),
        model=args.model,
        template=template,
        df=df,
        dst=os.path.join(args.dir, "output"),
        thinking=args.thinking,
        budget_tokens=args.budget_tokens,
        max_tokens=args.max_tokens,
        num_workers=32,
    )
    # merge data
    merge(os.path.join(args.dir, "output"), os.path.join(args.dir, "merged"))


if __name__ == "__main__":
    main()
