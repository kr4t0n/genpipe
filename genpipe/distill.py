import os
import json
import string

from pandas.core.frame import DataFrame

from uuid import uuid4
from functools import partial
from openai import OpenAI
from tqdm.contrib.concurrent import thread_map

from .utils import call


def _print_delimiter():
    print("=" * 80)


def _distill(
    llm: OpenAI,
    model: str,
    system: str | None,
    template: str,
    dst: str,
    thinking: bool,
    budget_tokens: int,
    max_tokens: int,
    df_row: tuple,
) -> None:
    _, row = df_row
    prompt = template.format(**row)

    if system is not None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [
            {"role": "user", "content": prompt},
        ]

    try:
        answer, reasoning = call(
            llm,
            model,
            messages,
            thinking=thinking,
            budget_tokens=budget_tokens,
            max_tokens=max_tokens,
        )

        data = {
            **row,
            "system": system,
            "prompt": prompt,
            "answer": answer,
            "reasoning": reasoning,
        }

        with open(f"{dst}/{uuid4().hex}.json", "w") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"error encoutered during distillation: {e}")


def distill(
    llm: OpenAI,
    model: str,
    df: DataFrame,
    system: str | None,
    template: str,
    dst: str = "output",
    thinking: bool = False,
    budget_tokens: int = 4 * 1000,
    max_tokens: int = 64 * 1000,
    num_workers=32,
) -> None:
    print(f"data size: {len(df)}")
    _print_delimiter()

    print(f"system is:\n{system}")
    print(f"template is:\n{template}")
    _print_delimiter()

    keys = [p[1] for p in string.Formatter().parse(template) if p[1] is not None]
    print(f"key needed to fill: {keys}")
    check_keys = [key in df.columns for key in keys]
    if check_keys:
        print("all keys are in the data! pass checking.")
        _print_delimiter()
    else:
        raise ValueError("some keys are not in the data!")

    os.makedirs(dst, exist_ok=True)
    print(f"distilling data to {dst}")
    thread_map(
        partial(
            _distill,
            llm,
            model,
            system,
            template,
            dst,
            thinking,
            budget_tokens,
            max_tokens,
        ),
        df.iterrows(),
        total=len(df),
        max_workers=num_workers,
    )
