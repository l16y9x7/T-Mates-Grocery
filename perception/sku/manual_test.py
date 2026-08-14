import sys
from pathlib import Path

import requests


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))
from config import SKU_API_URL  # noqa: E402

response = requests.get(
    f"{SKU_API_URL}/sku/get_candidate_SKU",
    json={
        "location_id": "H2_F_L4_C05",
        "pose_type": "SHELF_VIEW_UPPER",
    },
)

print(response.content.decode("utf-8"))
