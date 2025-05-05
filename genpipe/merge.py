import os
import json
import shutil
import pandas as pd
import dask.dataframe as dd


def merge(output_dir: str, merge_dir: str) -> None:
    fs = os.listdir(output_dir)
    print(f"found {len(fs)} files under {output_dir}")

    data = [json.load(open(os.path.join(output_dir, f))) for f in fs]
    df = pd.DataFrame.from_dict(data)

    n_partitions = 1 if len(df) < 10 * 10000 else 10
    ddf = dd.from_pandas(df, npartitions=n_partitions)

    ddf.to_parquet(merge_dir, write_index=False)
    print(f"write merged data to {merge_dir}")

    shutil.rmtree(output_dir)
    print(f"clean up {output_dir}")
