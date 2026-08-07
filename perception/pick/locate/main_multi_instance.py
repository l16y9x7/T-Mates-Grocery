from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from main import LocateRequest, locate_product_debug, normalize_bbox_to_1_1000


app = FastAPI(title="Sorting Pick Locate Multi Instance", version="2.0.0")


class LocateResponse(BaseModel):
    product_name: str
    bbox: list[list[int]]
    mask: list[str]
    image_path: str


def make_locate_response(request: LocateRequest) -> LocateResponse:
    result = locate_product_debug(request)
    if not result.instances:
        raise HTTPException(status_code=404, detail="SAM3 没有找到目标商品实例")

    # result.instances 已经过 SAM3 重叠链的 frontmost 过滤；这里不再进行全局 frontmost 筛选。
    return LocateResponse(
        product_name=result.product_name,
        bbox=[
            normalize_bbox_to_1_1000(instance.bbox, result.image_size)
            for instance in result.instances
        ],
        mask=[instance.mask for instance in result.instances],
        image_path=result.image_path,
    )


@app.post("/perception/pick/locate", response_model=LocateResponse)
def locate_product(request: LocateRequest) -> LocateResponse:
    return make_locate_response(request)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("MULTI_LOCATE_PORT", "8081")),
    )
