import json
import ctypes
import dask.dataframe as dd
from dask.distributed import Client

def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

def PA0(path_to_user_reviews_csv):
    client = Client(
        n_workers=4,
        threads_per_worker=1,
        memory_limit="3.5GB"
    )
    # Helps fix any memory leaks.
    client.run(trim_memory)
    client = client.restart()

    cols = ["reviewerID", "overall", "unixReviewTime", "helpful"]

    reviews = dd.read_csv(
        path_to_user_reviews_csv,
        usecols=cols,
        dtype={
            "reviewerID": "object",
            "overall": "float32",
            "unixReviewTime": "float64",
            "helpful": "object",
        },
        blocksize="256MB",
        assume_missing=False,
        on_bad_lines="skip"
    )

    reviews = reviews.dropna(
        subset=["reviewerID", "overall", "unixReviewTime"]
    )

    # helpful looks like: "[3, 5]"
    helpful_parts = reviews["helpful"].str.extract(
        r"\[(\d+),\s*(\d+)\]",
        expand=True
    )

    reviews["helpful_votes"] = helpful_parts[0].fillna(0).astype("uint32")
    reviews["total_votes"] = helpful_parts[1].fillna(0).astype("uint32")

    users = reviews.groupby("reviewerID").agg({
        "overall": ["count", "mean"],
        "unixReviewTime": "min",
        "helpful_votes": "sum",
        "total_votes": "sum"
    }, split_out=64)

    users.columns = [
        "number_products_rated",
        "avg_ratings",
        "first_review_time",
        "helpful_votes",
        "total_votes"
    ]

    users["reviewing_since"] = dd.to_datetime(
        users["first_review_time"],
        unit="s"
    ).dt.year.astype("int16")

    users = users[
        [
            "number_products_rated",
            "avg_ratings",
            "reviewing_since",
            "helpful_votes",
            "total_votes"
        ]
    ]
    
    submit = users.describe().compute().round(2)    
    with open('results_PA0.json', 'w') as outfile: 
        json.dump(json.loads(submit.to_json()), outfile)

if __name__ == "__main__":
    PA0("user_reviews.csv")