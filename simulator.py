
import argparse
import json
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "simulated_data"
STATE_DIR = BASE_DIR / "state"
CHECKPOINT_FILE = STATE_DIR / "checkpoint.json"

DEFAULT_BATCH_SIZE = 1000

# Each CSV is loaded as strings to preserve IDs and
# ZIP prefixes, including values with leading zeros.
FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": (
        "product_category_name_translation.csv"
    ),
}


# --------------------------------------------------
# 2. Load the original source data
# --------------------------------------------------

def load_source_data():
    data = {}

    for name, filename in FILES.items():
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing source file: {path}"
            )

        data[name] = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )

        print(
            f"Loaded {filename}: "
            f"{len(data[name]):,} rows"
        )

    return data


# --------------------------------------------------
# 3. Read the previous checkpoint
# --------------------------------------------------

def load_checkpoint():
    if not CHECKPOINT_FILE.exists():
        return {
            "next_offset": 0,
            "next_batch_number": 1,
        }

    with CHECKPOINT_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)


# --------------------------------------------------
# 4. Save checkpoint after successful batch creation
# --------------------------------------------------

def save_checkpoint(next_offset, next_batch_number):
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "next_offset": next_offset,
        "next_batch_number": next_batch_number,
    }

    temp_file = CHECKPOINT_FILE.with_suffix(".tmp")

    with temp_file.open("w", encoding="utf-8") as file:
        json.dump(checkpoint, file, indent=4)

    temp_file.replace(CHECKPOINT_FILE)


# --------------------------------------------------
# 5. Select the next batch of orders
# --------------------------------------------------

def select_orders(orders, offset, batch_size):
    # Stable ordering makes the simulation repeatable.
    sorted_orders = orders.sort_values(
        by=[
            "order_purchase_timestamp",
            "order_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    batch = sorted_orders.iloc[
        offset: offset + batch_size
    ].copy()

    return sorted_orders, batch


# --------------------------------------------------
# 6. Select records related to the chosen orders
# --------------------------------------------------

def create_batch(data, selected_orders):
    order_ids = set(selected_orders["order_id"])

    # Customers who placed the selected orders
    customer_ids = set(
        selected_orders["customer_id"]
    )

    customers = data["customers"][
        data["customers"]["customer_id"].isin(
            customer_ids
        )
    ].copy()

    # All items belonging to the selected orders
    order_items = data["order_items"][
        data["order_items"]["order_id"].isin(
            order_ids
        )
    ].copy()

    # Related products and sellers
    product_ids = set(order_items["product_id"])
    seller_ids = set(order_items["seller_id"])

    products = data["products"][
        data["products"]["product_id"].isin(
            product_ids
        )
    ].copy()

    sellers = data["sellers"][
        data["sellers"]["seller_id"].isin(
            seller_ids
        )
    ].copy()

    # Payments and reviews associated with these orders
    payments = data["payments"][
        data["payments"]["order_id"].isin(
            order_ids
        )
    ].copy()

    reviews = data["reviews"][
        data["reviews"]["order_id"].isin(
            order_ids
        )
    ].copy()

    # Relevant product category translations
    categories = set(
        products["product_category_name"]
    )

    translations = data["category_translation"][
        data["category_translation"][
            "product_category_name"
        ].isin(categories)
    ].copy()

    # Geographic records for the selected customers
    # and sellers. ZIP prefixes can have multiple rows.
    customer_zips = set(
        customers["customer_zip_code_prefix"]
    )
    seller_zips = set(
        sellers["seller_zip_code_prefix"]
    )

    relevant_zips = customer_zips | seller_zips

    geolocation = data["geolocation"][
        data["geolocation"][
            "geolocation_zip_code_prefix"
        ].isin(relevant_zips)
    ].copy()

    return {
        "olist_orders_dataset": selected_orders,
        "olist_customers_dataset": customers,
        "olist_order_items_dataset": order_items,
        "olist_order_payments_dataset": payments,
        "olist_order_reviews_dataset": reviews,
        "olist_products_dataset": products,
        "olist_sellers_dataset": sellers,
        "olist_geolocation_dataset": geolocation,
        "product_category_name_translation": translations,
    }


# --------------------------------------------------
# 7. Write batch CSV files
# --------------------------------------------------

def write_batch(batch_data, batch_number):
    batch_name = f"batch_{batch_number:04d}"
    batch_dir = OUTPUT_DIR / batch_name

    # Never silently overwrite an existing batch.
    if batch_dir.exists():
        raise FileExistsError(
            f"{batch_dir} already exists. "
            "Check the checkpoint before continuing."
        )

    batch_dir.mkdir(parents=True)

    try:
        for filename, dataframe in batch_data.items():
            path = batch_dir / f"{filename}.csv"

            dataframe.to_csv(
                path,
                index=False,
            )

            print(
                f"  {filename}: "
                f"{len(dataframe):,} rows"
            )

    except Exception:
        # Remove a partially written batch so that a
        # retry cannot mistake it for a completed batch.
        import shutil
        shutil.rmtree(batch_dir)
        raise

    return batch_dir


# --------------------------------------------------
# 8. Run one incremental ingestion simulation
# --------------------------------------------------

def run_simulation(batch_size):
    data = load_source_data()
    checkpoint = load_checkpoint()

    offset = checkpoint["next_offset"]
    batch_number = checkpoint["next_batch_number"]

    sorted_orders, selected_orders = select_orders(
        data["orders"],
        offset,
        batch_size,
    )

    if selected_orders.empty:
        print("No new orders available. Simulation complete.")
        return

    print(f"\nCreating batch {batch_number:04d}")
    print(f"Orders starting at offset: {offset:,}")
    print(f"Orders in this batch: {len(selected_orders):,}")

    batch_data = create_batch(
        data,
        selected_orders,
    )

    batch_dir = write_batch(
        batch_data,
        batch_number,
    )

    # Advance the checkpoint only after every CSV has
    # been written successfully.
    next_offset = offset + len(selected_orders)

    save_checkpoint(
        next_offset=next_offset,
        next_batch_number=batch_number + 1,
    )

    print(f"\nBatch saved to: {batch_dir}")
    print(f"Total orders released: {next_offset:,}")
    print(f"Remaining orders: {len(sorted_orders) - next_offset:,}")


# --------------------------------------------------
# 9. Command-line entry point
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate incremental Olist ingestion"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of orders released per run",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")

    run_simulation(args.batch_size)
