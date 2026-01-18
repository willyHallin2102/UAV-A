import numpy as np
import json
import csv


def generate_row():
    row = {}

    row["dvec"] = np.random.uniform(-1000, 1000, size=3).astype(np.float32)
    row["rx_type"] = np.random.randint(0, 3, dtype=np.uint8)
    row["link_state"] = np.random.randint(0, 3, dtype=np.uint8)
    row["los_pl"] = np.float32(np.random.uniform(80, 200))
    row["los_ang"] = np.zeros(4, dtype=np.float32)  # often zeros in your data
    row["los_dly"] = np.float32(np.random.uniform(0, 5e-6))
    row["nlos_pl"] = np.random.uniform(140, 180, size=20).astype(np.float32)
    row["nlos_ang"] = np.random.uniform(
        low=[-90, 90, -180, 80], high=[-50, 100, 100, 90], size=(20, 4)
    ).astype(np.float32)
    row["nlos_dly"] = np.random.uniform(2.4e-6, 3.2e-6, size=20).astype(np.float32)

    return row


def serialize_for_csv(row):
    out = {}
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            out[k] = json.dumps(v.tolist())
        else:
            out[k] = v
    return out


rows = [serialize_for_csv(generate_row()) for _ in range(1000)]
with open("synthetic_data.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
