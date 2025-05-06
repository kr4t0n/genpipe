import numpy as np
import dask.dataframe as dd

from .schema import schema


def transform(
    merge_dir: str,
    transform_dir: str,
    domain: str,
    cat: str,
    source: str,
    version: str,
    budget_tokens: int,
) -> None:
    ddf = dd.read_parquet(merge_dir)
    print(f"found {len(ddf)} rows of data")

    ddf["messages"] = ddf.apply(
        lambda x: np.array(
            [
                {
                    "content": np.array([x["prompt"].strip()], dtype=object),
                    "loss_mask": np.array([0.0]),
                    "name": "",
                    "role": "user",
                },
                {
                    "content": np.array([f"<think>{x['reasoning']}</think>\n{x['answer']}"], dtype=object),
                    "loss_mask": np.array([1.0]),
                    "name": "",
                    "role": "assistant",
                },
            ]
        ),
        axis=1,
        meta=("messages", "object"),
    )
    ddf["domain"] = domain
    ddf["cat"] = cat
    ddf["source"] = source
    ddf["version"] = version
    ddf["budget_tokens"] = budget_tokens

    ddf.to_parquet(transform_dir, write_index=False, schema=schema)
