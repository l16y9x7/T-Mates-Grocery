from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

import main
import test_inference


def png_base64(size: tuple[int, int], value: int = 255) -> str:
    image = Image.new("L", size, value)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def mask_base64_with_pixel_count(size: tuple[int, int], count: int) -> str:
    image = Image.new("L", size, 0)
    pixels = [255] * count + [0] * (size[0] * size[1] - count)
    image.putdata(pixels)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


class LocateLogicTest(unittest.TestCase):
    def test_inference_client_does_not_import_main(self) -> None:
        self.assertNotIn("main", test_inference.__dict__)

    def test_inference_sku_lookup_uses_remote_http_api(self) -> None:
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {"sku_id": "SKU_001", "name": "NFC橙汁"}
        with patch.object(
            test_inference.requests,
            "get",
            return_value=response,
        ) as request_mock:
            product = test_inference.lookup_sku_by_name("NFC橙汁")

        self.assertEqual(product["sku_id"], "SKU_001")
        request_mock.assert_called_once_with(
            "http://192.168.130.59:25540/sku/search_by_name",
            params={"name": "NFC橙汁"},
            timeout=test_inference.SKU_REQUEST_TIMEOUT_SECONDS,
        )

    def test_find_test_images_by_sku_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first_image = directory / "first_rgb.jpg"
            second_image = directory / "second_rgb.jpg"
            first_image.touch()
            second_image.touch()
            mapping_path = directory / "image_name_mapping.json"
            mapping_path.write_text(
                json.dumps(
                    {
                        first_image.name: ["SKU_001"],
                        second_image.name: ["SKU_001", "SKU_002"],
                    }
                ),
                encoding="utf-8",
            )

            paths = test_inference.find_test_images(
                "sku_001",
                mapping_path=mapping_path,
                image_directory=directory,
            )

            self.assertEqual(paths, [first_image, second_image])

    def test_save_result_visualization_draws_mask_and_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            image_path = directory / "source_rgb.jpg"
            Image.new("RGB", (20, 20), "white").save(image_path)
            result_path = test_inference.save_result_visualization(
                image_path,
                {
                    "instances": [
                        {
                            "bbox": [4, 4, 15, 15],
                            "mask": png_base64((20, 20)),
                            "score": 0.9,
                        }
                    ]
                },
                directory / "results",
                "SKU_001",
            )

            self.assertTrue(result_path.is_file())
            self.assertEqual(result_path.name, "source_rgb_SKU_001_locate.png")
            with Image.open(result_path) as result_image:
                self.assertEqual(result_image.size, (20, 20))
                self.assertNotEqual(result_image.getpixel((10, 10)), (255, 255, 255))

    def test_parse_qwen_json_from_code_fence(self) -> None:
        detections = main.parse_qwen_detections(
            '结果如下：```json\n[{"name":"商品","bbox":[10,20,30,40]}]\n```'
        )
        self.assertEqual(
            detections,
            [{"name": "商品", "bbox": [10.0, 20.0, 30.0, 40.0]}],
        )

    def test_consensus_keeps_cross_sample_match_and_drops_singleton(self) -> None:
        samples = [
            (
                1,
                [
                    {"name": "目标", "bbox": [100.0, 100.0, 500.0, 500.0]},
                    {"name": "目标", "bbox": [700.0, 100.0, 800.0, 200.0]},
                ],
            ),
            (2, [{"name": "目标", "bbox": [102.0, 102.0, 498.0, 498.0]}]),
            (3, [{"name": "误检", "bbox": [20.0, 20.0, 80.0, 80.0]}]),
        ]

        result = main.consensus_qwen_bboxes(samples)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [101.0, 101.0, 499.0, 499.0])

    def test_same_sample_duplicates_do_not_form_consensus(self) -> None:
        samples = [
            (
                1,
                [
                    {"name": "目标", "bbox": [100.0, 100.0, 500.0, 500.0]},
                    {"name": "目标", "bbox": [101.0, 101.0, 499.0, 499.0]},
                ],
            )
        ]
        self.assertEqual(main.consensus_qwen_bboxes(samples), [])

    def test_normalized_qwen_bbox_matches_web_crop(self) -> None:
        crop_box = main.qwen_bbox_to_crop(
            [645.0, 689.0, 928.0, 899.0],
            (1280, 720),
        )
        self.assertEqual(crop_box, (789, 480, 1225, 663))

    def test_decode_uploaded_image(self) -> None:
        encoded = base64.b64encode(b"image bytes").decode("ascii")
        self.assertEqual(main.decode_uploaded_image(encoded), b"image bytes")

    def test_sam_bbox_and_mask_are_mapped_to_original_image(self) -> None:
        instance = {
            "bbox_xyxy": [1, 2, 3, 4],
            "mask_png_base64": png_base64((4, 5)),
            "score": 0.9,
        }
        mapped = main.map_sam_instance_to_original(
            instance,
            crop_box=(10, 20, 14, 25),
            original_size=(30, 40),
        )

        self.assertEqual(mapped.bbox, [11.0, 22.0, 13.0, 24.0])
        self.assertEqual(mapped.score, 0.9)
        with Image.open(io.BytesIO(base64.b64decode(mapped.mask))) as mask:
            self.assertEqual(mask.size, (30, 40))
            self.assertEqual(mask.getpixel((10, 20)), 255)
            self.assertEqual(mask.getpixel((0, 0)), 0)

    def test_overlap_chain_keeps_highest_mask_density(self) -> None:
        instances = [
            main.LocatedInstance(
                bbox=[0, 0, 10, 10],
                mask=mask_base64_with_pixel_count((40, 10), 60),
                score=0.6,
            ),
            main.LocatedInstance(
                bbox=[5, 0, 15, 10],
                mask=mask_base64_with_pixel_count((40, 10), 70),
                score=0.7,
            ),
            main.LocatedInstance(
                bbox=[10, 0, 20, 10],
                mask=mask_base64_with_pixel_count((40, 10), 50),
                score=0.9,
            ),
            main.LocatedInstance(
                bbox=[30, 0, 40, 10],
                mask=mask_base64_with_pixel_count((40, 10), 80),
                score=0.8,
            ),
        ]

        filtered = main.keep_frontmost_in_overlap_chains(instances)

        self.assertEqual(filtered, [instances[1], instances[3]])

    def test_overlap_chain_uses_two_times_mask_area_shortcut(self) -> None:
        large_mask = main.LocatedInstance(
            bbox=[0, 0, 20, 20],
            mask=mask_base64_with_pixel_count((20, 20), 120),
            score=0.5,
        )
        denser_small_mask = main.LocatedInstance(
            bbox=[5, 5, 15, 15],
            mask=mask_base64_with_pixel_count((20, 20), 50),
            score=0.99,
        )

        filtered = main.keep_frontmost_in_overlap_chains(
            [large_mask, denser_small_mask]
        )

        self.assertEqual(filtered, [large_mask])

    def test_small_bbox_overlap_does_not_merge_neighbors(self) -> None:
        first = main.LocatedInstance(
            bbox=[0, 0, 10, 10],
            mask=mask_base64_with_pixel_count((20, 10), 80),
            score=0.8,
        )
        second = main.LocatedInstance(
            bbox=[9, 0, 19, 10],
            mask=mask_base64_with_pixel_count((20, 10), 80),
            score=0.8,
        )

        filtered = main.keep_frontmost_in_overlap_chains([first, second])

        self.assertEqual(filtered, [first, second])

    def test_locate_returns_multiple_original_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "frame_rgb.jpg"
            Image.new("RGB", (100, 80), "white").save(image_path)

            def fake_sam(_: str, crop_image: Image.Image) -> list[dict]:
                width, height = crop_image.size
                return [
                    {
                        "bbox_xyxy": [0, 0, width / 2, height],
                        "mask_png_base64": png_base64((width, height)),
                        "score": 0.95,
                    },
                    {
                        "bbox_xyxy": [width / 2, 0, width, height],
                        "mask_png_base64": png_base64((width, height)),
                        "score": 0.91,
                    },
                ]

            with (
                patch.object(
                    main,
                    "lookup_sku_by_name",
                    return_value={"sku_id": "SKU_001", "name": "NFC桔汁"},
                ),
                patch.object(
                    main,
                    "load_prompt_pair",
                    return_value=("qwen prompt", "sam prompt"),
                ),
                patch.object(main, "get_latest_rgb", return_value=image_path),
                patch.object(
                    main,
                    "get_stable_qwen_bboxes",
                    return_value=[[100.0, 100.0, 500.0, 500.0]],
                ),
                patch.object(main, "call_sam3", side_effect=fake_sam),
            ):
                result = main.locate_product(main.LocateRequest(name="NFC桔汁"))

            self.assertEqual(result.sku_id, "SKU_001")
            self.assertEqual(result.name, "NFC桔汁")
            self.assertEqual(result.image_name, "frame_rgb.jpg")
            self.assertEqual(len(result.instances), 2)
            for instance in result.instances:
                with Image.open(io.BytesIO(base64.b64decode(instance.mask))) as mask:
                    self.assertEqual(mask.size, (100, 80))

    def test_locate_accepts_uploaded_image(self) -> None:
        image_buffer = io.BytesIO()
        Image.new("RGB", (20, 10), "white").save(image_buffer, format="JPEG")
        request = main.LocateRequest(
            name="NFC桔汁",
            image_name="mapped_rgb.jpg",
            image_base64=base64.b64encode(image_buffer.getvalue()).decode("ascii"),
        )
        expected = main.LocateResponse(
            sku_id="SKU_001",
            name="NFC桔汁",
            image_name="mapped_rgb.jpg",
            instances=[],
        )
        with (
            patch.object(
                main,
                "lookup_sku_by_name",
                return_value={"sku_id": "SKU_001", "name": "NFC桔汁"},
            ),
            patch.object(
                main,
                "locate_product_in_image",
                return_value=expected,
            ) as locate_mock,
        ):
            result = main.locate_product(request)

        self.assertEqual(result, expected)
        uploaded_path = locate_mock.call_args.args[1]
        self.assertEqual(uploaded_path.name, "mapped_rgb.jpg")


if __name__ == "__main__":
    unittest.main()
