import requests

response = requests.get(
    "http://192.168.1.226:25540/sku/get_candidate_SKU",
    json={
        "location_id": "H2_F_L4_C05",
        "pose_type": "SHELF_VIEW_UPPER",
    },
)

print(response.content.decode("utf-8"))
