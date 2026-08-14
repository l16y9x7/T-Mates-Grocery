from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
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
    def test_get_latest_rgb_prefers_camera_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            image_buffer = io.BytesIO()
            Image.new("RGB", (16, 12), "blue").save(image_buffer, format="JPEG")
            response = Mock(content=image_buffer.getvalue())
            response.raise_for_status.return_value = None
            with (
                patch.object(main.requests, "get", return_value=response) as get_mock,
                patch.object(main, "CAMERA_SNAPSHOT_CACHE_DIR", directory / "camera"),
            ):
                image_path = main.get_latest_rgb()

            self.assertTrue(image_path.is_file())
            self.assertEqual(image_path.name, "latest_camera_rgb.jpg")
            with Image.open(image_path) as image:
                self.assertEqual(image.size, (16, 12))
            get_mock.assert_called_once_with(
                main.CAMERA_SNAPSHOT_URL,
                timeout=main.CAMERA_SNAPSHOT_TIMEOUT_SECONDS,
            )

    def test_get_latest_rgb_returns_400_without_local_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "002_rgb.jpg").touch()
            with (
                patch.object(
                    main.requests,
                    "get",
                    side_effect=main.requests.RequestException("camera offline"),
                ),
                patch.object(main, "CAMERA_SNAPSHOT_CACHE_DIR", directory / "camera"),
                self.assertRaises(main.HTTPException) as raised,
            ):
                main.get_latest_rgb()

            self.assertEqual(raised.exception.status_code, 400)

    def test_fetch_camera_depth_parses_raw_16uc1(self) -> None:
        values = [0, 625, 710, 845]
        response = Mock(
            content=b"".join(value.to_bytes(2, "little") for value in values),
            headers={
                "X-Image-Width": "2",
                "X-Image-Height": "2",
                "X-Image-Encoding": "16UC1",
                "X-Image-Step": "4",
                "X-Image-Is-Bigendian": "0",
            },
        )
        response.raise_for_status.return_value = None

        with patch.object(main.requests, "get", return_value=response) as get_mock:
            depth_image = main.fetch_camera_depth("left", (2, 2))

        self.assertIsNotNone(depth_image)
        self.assertEqual(list(depth_image.getdata()), values)
        get_mock.assert_called_once_with(
            main.LEFT_CAMERA_DEPTH_SNAPSHOT_URL,
            timeout=main.CAMERA_SNAPSHOT_TIMEOUT_SECONDS,
        )

    def test_fetch_camera_depth_rejects_unaligned_size(self) -> None:
        response = Mock(
            content=b"\x00" * 8,
            headers={
                "X-Image-Width": "2",
                "X-Image-Height": "2",
                "X-Image-Encoding": "16UC1",
                "X-Image-Step": "4",
                "X-Image-Is-Bigendian": "0",
            },
        )
        response.raise_for_status.return_value = None

        with patch.object(main.requests, "get", return_value=response):
            depth_image = main.fetch_camera_depth("left", (4, 2))

        self.assertIsNone(depth_image)

    def test_decode_uploaded_raw_depth_uses_rgb_dimensions(self) -> None:
        values = [0, 625, 710, 845]
        encoded = base64.b64encode(
            b"".join(value.to_bytes(2, "little") for value in values)
        ).decode("ascii")

        depth_image = main.decode_uploaded_depth_image(
            encoded,
            "frame_depth.raw",
            (2, 2),
        )

        self.assertEqual(depth_image.size, (2, 2))
        self.assertEqual(list(depth_image.getdata()), values)

    def test_decode_uploaded_raw_depth_rejects_wrong_size(self) -> None:
        with self.assertRaises(main.HTTPException) as raised:
            main.decode_uploaded_depth_image(
                base64.b64encode(b"\x00" * 6).decode("ascii"),
                "frame_depth.raw",
                (2, 2),
            )
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("尺寸不匹配", str(raised.exception.detail))

    def test_decode_uploaded_npy_depth(self) -> None:
        depth_buffer = io.BytesIO()
        np.save(
            depth_buffer,
            np.array([[0, 625], [710, 845]], dtype=np.uint16),
            allow_pickle=False,
        )

        depth_image = main.decode_uploaded_depth_image(
            base64.b64encode(depth_buffer.getvalue()).decode("ascii"),
            "frame_depth.npy",
            (2, 2),
        )

        self.assertEqual(depth_image.size, (2, 2))
        self.assertEqual(list(depth_image.getdata()), [0, 625, 710, 845])

    def test_decode_uploaded_npy_depth_rejects_wrong_shape(self) -> None:
        depth_buffer = io.BytesIO()
        np.save(
            depth_buffer,
            np.zeros((2, 2, 1), dtype=np.uint16),
            allow_pickle=False,
        )
        with self.assertRaises(main.HTTPException) as raised:
            main.decode_uploaded_depth_image(
                base64.b64encode(depth_buffer.getvalue()).decode("ascii"),
                "frame_depth.npy",
                (2, 2),
            )
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("二维数组", str(raised.exception.detail))

    def test_formal_request_without_image_returns_400_when_camera_fails(self) -> None:
        request = main.LocateRequest(
            task_type="SORTING",
            product_name="可口可乐",
            level="L1",
            hand="left",
        )
        with (
            patch.object(
                main,
                "lookup_sku_by_name",
                return_value={"sku_id": "SKU_001", "name": "可口可乐"},
            ),
            patch.object(main, "fetch_camera_snapshot", return_value=None),
            self.assertRaises(main.HTTPException) as raised,
        ):
            main.locate_product(request)

        self.assertEqual(raised.exception.status_code, 400)

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
            "http://127.0.0.1:25540/sku/search_by_name",
            params={"name": "NFC橙汁"},
            timeout=test_inference.SKU_REQUEST_TIMEOUT_SECONDS,
        )

    def test_debug_inference_uses_formal_request_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "mapped_rgb.jpg"
            image_path.write_bytes(b"image bytes")
            response = Mock(ok=False, status_code=400)
            response.json.return_value = {"detail": "test stop"}
            with (
                patch.object(
                    test_inference,
                    "lookup_sku_by_name",
                    return_value={"sku_id": "SKU_001", "name": "NFC桔汁"},
                ),
                patch.object(
                    test_inference,
                    "find_test_images",
                    return_value=[image_path],
                ),
                patch.object(
                    test_inference.requests,
                    "post",
                    return_value=response,
                ) as post_mock,
            ):
                test_inference.run_test_inference(
                    "SORTING",
                    "NFC桔汁",
                    "L1",
                    "right",
                    output_directory=Path(temporary_directory) / "results",
                )

            self.assertEqual(
                post_mock.call_args.args[0],
                "http://127.0.0.1:8083/perception/pick/locate/debug",
            )
            request_payload = post_mock.call_args.kwargs["json"]
            self.assertEqual(request_payload["task_type"], "SORTING")
            self.assertEqual(request_payload["product_name"], "NFC桔汁")
            self.assertEqual(request_payload["level"], "L1")
            self.assertEqual(request_payload["hand"], "right")
            self.assertEqual(request_payload["image_name"], image_path.name)
            self.assertEqual(
                base64.b64decode(request_payload["image_base64"]),
                b"image bytes",
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

    def test_save_qwen_visualization_draws_original_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            image_path = directory / "source_rgb.jpg"
            Image.new("RGB", (30, 20), "white").save(image_path)
            result_path = test_inference.save_qwen_visualization(
                image_path,
                {
                    "qwen_bboxes": [
                        {
                            "bbox_normalized": [100, 200, 500, 800],
                            "bbox_original": [4, 5, 24, 16],
                            "crop_box_original": [2, 3, 26, 18],
                        }
                    ]
                },
                directory / "results",
                "SKU_001",
            )

            self.assertTrue(result_path.is_file())
            self.assertEqual(result_path.name, "source_rgb_SKU_001_qwen.png")
            with Image.open(result_path) as result_image:
                self.assertEqual(result_image.size, (30, 20))
                self.assertNotEqual(result_image.getpixel((4, 10)), (255, 255, 255))

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
        self.assertEqual(
            main.qwen_bbox_to_original(
                [645.0, 689.0, 928.0, 899.0],
                (1280, 720),
            ),
            [825.6, 496.08, 1187.84, 647.28],
        )

    def test_decode_uploaded_image(self) -> None:
        encoded = base64.b64encode(b"image bytes").decode("ascii")
        self.assertEqual(main.decode_uploaded_image(encoded), b"image bytes")

    def test_monitor_image_path_remains_after_source_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source_path = directory / "uploaded_rgb.jpg"
            source_path.write_bytes(b"uploaded image bytes")
            with patch.object(main, "MONITOR_IMAGE_DIR", directory / "stored"):
                stored_path = Path(main.store_monitor_image(source_path))

            source_path.unlink()
            self.assertTrue(stored_path.is_absolute())
            self.assertTrue(stored_path.is_file())
            self.assertEqual(stored_path.read_bytes(), b"uploaded image bytes")

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

    def test_overlap_chain_prefers_front_depth_before_mask_density(self) -> None:
        denser_back = main.LocatedInstance(
            bbox=[0, 0, 10, 10],
            mask=mask_base64_with_pixel_count((15, 10), 90),
            score=0.99,
            depth_mm=600,
        )
        expected_front = main.LocatedInstance(
            bbox=[5, 0, 15, 10],
            mask=mask_base64_with_pixel_count((15, 10), 50),
            score=0.5,
            depth_mm=500,
        )

        filtered = main.keep_frontmost_in_overlap_chains(
            [denser_back, expected_front]
        )

        self.assertEqual(filtered, [expected_front])

    def test_overlap_chain_uses_mask_rules_within_same_front_depth_layer(self) -> None:
        expected_denser = main.LocatedInstance(
            bbox=[0, 0, 10, 10],
            mask=mask_base64_with_pixel_count((15, 10), 80),
            depth_mm=500,
        )
        slightly_nearer = main.LocatedInstance(
            bbox=[5, 0, 15, 10],
            mask=mask_base64_with_pixel_count((15, 10), 60),
            depth_mm=480,
        )

        filtered = main.keep_frontmost_in_overlap_chains(
            [expected_denser, slightly_nearer]
        )

        self.assertEqual(filtered, [expected_denser])

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

    def test_smallest_mask_under_half_of_second_smallest_is_removed(self) -> None:
        instances = [
            main.LocatedInstance(
                bbox=[0, 0, 10, 10],
                mask=mask_base64_with_pixel_count((30, 10), 100),
                score=0.8,
            ),
            main.LocatedInstance(
                bbox=[10, 0, 20, 10],
                mask=mask_base64_with_pixel_count((30, 10), 40),
                score=0.8,
            ),
            main.LocatedInstance(
                bbox=[20, 0, 30, 10],
                mask=mask_base64_with_pixel_count((30, 10), 10),
                score=0.8,
            ),
        ]

        filtered = main.drop_smallest_mask_area_outlier(instances)

        self.assertEqual(filtered, instances[:2])

    def test_smallest_mask_over_half_of_second_smallest_is_kept(self) -> None:
        instances = [
            main.LocatedInstance(
                bbox=[0, 0, 10, 10],
                mask=mask_base64_with_pixel_count((20, 10), 100),
                score=0.8,
            ),
            main.LocatedInstance(
                bbox=[10, 0, 20, 10],
                mask=mask_base64_with_pixel_count((20, 10), 60),
                score=0.8,
            ),
        ]

        filtered = main.drop_smallest_mask_area_outlier(instances)

        self.assertEqual(filtered, instances)

    def test_locate_returns_multiple_original_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            image_path = temporary_path / "frame_rgb.jpg"
            monitor_directory = temporary_path / "monitor_images"
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
                patch.object(main, "fetch_camera_depth", return_value=None) as depth_mock,
                patch.object(
                    main,
                    "get_stable_qwen_bboxes",
                    return_value=[[100.0, 100.0, 500.0, 500.0]],
                ),
                patch.object(main, "call_sam3", side_effect=fake_sam),
                patch.object(main, "MONITOR_IMAGE_DIR", monitor_directory),
            ):
                result = main.locate_product_debug(
                    main.LocateRequest(
                        task_type="SORTING",
                        product_name="NFC桔汁",
                        level="L1",
                        hand="left",
                    )
                )

            depth_mock.assert_called_once_with("left", (100, 80))
            self.assertEqual(result.sku_id, "SKU_001")
            self.assertEqual(result.product_name, "NFC桔汁")
            self.assertEqual(result.image_name, "frame_rgb.jpg")
            self.assertTrue(Path(result.image_path).is_file())
            self.assertEqual(Path(result.image_path).parent, monitor_directory.resolve())
            self.assertTrue(base64.b64decode(result.image_base64))
            self.assertEqual(result.image_media_type, "image/jpeg")
            self.assertEqual(len(result.qwen_bboxes), 1)
            self.assertEqual(
                result.qwen_bboxes[0].bbox_original,
                [10.0, 8.0, 50.0, 40.0],
            )
            self.assertEqual(len(result.instances), 2)
            self.assertIsNotNone(result.selected_instance)
            self.assertIn(result.selected_instance_index, {1, 2})
            self.assertEqual(
                result.selected_instance,
                result.instances[result.selected_instance_index - 1],
            )
            for instance in result.instances:
                with Image.open(io.BytesIO(base64.b64decode(instance.mask))) as mask:
                    self.assertEqual(mask.size, (100, 80))

    def test_locate_skips_depth_snapshot_for_single_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            image_path = temporary_path / "frame_rgb.jpg"
            monitor_directory = temporary_path / "monitor_images"
            Image.new("RGB", (100, 80), "white").save(image_path)

            with (
                patch.object(
                    main,
                    "lookup_sku_by_name",
                    return_value={"sku_id": "SKU_001", "name": "NFC桔汁"},
                ),
                patch.object(main, "get_latest_rgb", return_value=image_path),
                patch.object(main, "fetch_camera_depth") as depth_mock,
                patch.object(
                    main,
                    "get_stable_qwen_bboxes",
                    return_value=[[100.0, 100.0, 500.0, 500.0]],
                ),
                patch.object(
                    main,
                    "call_sam3",
                    return_value=[
                        {
                            "bbox_xyxy": [0, 0, 40, 32],
                            "mask_png_base64": png_base64((40, 32)),
                            "score": 0.95,
                        }
                    ],
                ),
                patch.object(main, "MONITOR_IMAGE_DIR", monitor_directory),
            ):
                result = main.locate_product_debug(
                    main.LocateRequest(
                        task_type="SORTING",
                        product_name="NFC桔汁",
                        level="L1",
                        hand="left",
                        qwen3_prompt="qwen prompt",
                        sam3_prompt="sam prompt",
                    ),
                    allow_prompt_overrides=True,
                )

            depth_mock.assert_not_called()
            self.assertEqual(len(result.instances), 1)
            self.assertIs(result.selected_instance, result.instances[0])

    def test_locate_skips_depth_snapshot_for_shortage_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            image_path = temporary_path / "frame_rgb.jpg"
            monitor_directory = temporary_path / "monitor_images"
            Image.new("RGB", (100, 80), "white").save(image_path)
            depth_provider = Mock(return_value=Image.new("I", (100, 80), 700))

            with (
                patch.object(
                    main,
                    "get_stable_qwen_bboxes",
                    return_value=[[100.0, 100.0, 500.0, 500.0]],
                ),
                patch.object(
                    main,
                    "call_sam3",
                    return_value=[
                        {
                            "bbox_xyxy": [0, 0, 20, 32],
                            "mask_png_base64": png_base64((40, 32)),
                            "score": 0.95,
                        },
                        {
                            "bbox_xyxy": [20, 0, 40, 32],
                            "mask_png_base64": png_base64((40, 32)),
                            "score": 0.91,
                        },
                    ],
                ),
                patch.object(main, "MONITOR_IMAGE_DIR", monitor_directory),
            ):
                result = main.locate_product_in_image(
                    {"sku_id": "SKU_001", "name": "NFC桔汁"},
                    image_path,
                    task_type="SHORTAGE",
                    level="L1",
                    qwen_prompt_override="qwen prompt",
                    sam_prompt_override="sam prompt",
                    depth_image_provider=depth_provider,
                )

            depth_provider.assert_not_called()
            self.assertEqual(len(result.instances), 2)
            self.assertTrue(
                all(instance.depth_mm is None for instance in result.instances)
            )

    def test_normal_case_filters_depth_row_before_overlap_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            image_path = temporary_path / "frame_rgb.jpg"
            Image.new("RGB", (140, 120), "white").save(image_path)
            mask = png_base64((140, 120))
            sam_instances = [
                {
                    "bbox_xyxy": [10, 20, 60, 100],
                    "mask_png_base64": mask,
                    "score": 0.9,
                },
                {
                    "bbox_xyxy": [40, 10, 90, 70],
                    "mask_png_base64": mask,
                    "score": 0.95,
                },
                {
                    "bbox_xyxy": [70, 20, 120, 100],
                    "mask_png_base64": mask,
                    "score": 0.9,
                },
            ]
            with (
                patch.object(main, "get_stable_qwen_bboxes", return_value=[[0, 0, 1000, 1000]]),
                patch.object(main, "call_sam3", return_value=sam_instances),
                patch.object(
                    main,
                    "estimate_instance_depth_mm",
                    side_effect=[700.0, 800.0, 710.0],
                ) as depth_mock,
                patch.object(main, "MONITOR_IMAGE_DIR", temporary_path / "monitor"),
            ):
                result = main.locate_product_in_image(
                    {"sku_id": "SKU_001", "name": "NFC桔汁"},
                    image_path,
                    task_type="SORTING",
                    level="L1",
                    hand="left",
                    qwen_prompt_override="qwen prompt",
                    sam_prompt_override="sam prompt",
                    depth_image=Image.new("I", (140, 120), 700),
                )

        self.assertEqual(depth_mock.call_count, 3)
        self.assertEqual(len(result.raw_sam_instances), 3)
        self.assertEqual(
            [instance.depth_mm for instance in result.raw_sam_instances],
            [700.0, 800.0, 710.0],
        )
        self.assertEqual(
            [instance.bbox for instance in result.instances],
            [[10.0, 20.0, 60.0, 100.0], [70.0, 20.0, 120.0, 100.0]],
        )

    def test_locate_accepts_uploaded_image(self) -> None:
        image_buffer = io.BytesIO()
        Image.new("RGB", (20, 10), "white").save(image_buffer, format="JPEG")
        request = main.LocateRequest(
            task_type="SORTING",
            product_name="NFC桔汁",
            level="L1",
            hand="left",
            image_name="mapped_rgb.jpg",
            image_base64=base64.b64encode(image_buffer.getvalue()).decode("ascii"),
            depth_image_name="mapped_depth.raw",
            depth_image_base64=base64.b64encode(
                b"".join((700).to_bytes(2, "little") for _ in range(20 * 10))
            ).decode("ascii"),
        )
        expected = main.LocateDebugResponse(
            sku_id="SKU_001",
            product_name="NFC桔汁",
            image_name="mapped_rgb.jpg",
            image_path="C:/monitor/mapped_rgb.jpg",
            image_base64=base64.b64encode(b"image").decode("ascii"),
            image_media_type="image/jpeg",
            image_size=[20, 10],
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
            result = main.locate_product_debug(request)

        self.assertEqual(result, expected)
        uploaded_path = locate_mock.call_args.args[1]
        self.assertEqual(uploaded_path.name, "mapped_rgb.jpg")
        uploaded_depth = locate_mock.call_args.kwargs["depth_image"]
        self.assertEqual(uploaded_depth.size, (20, 10))
        self.assertEqual(set(uploaded_depth.getdata()), {700})

    def test_debug_response_keeps_uploaded_image_when_inference_fails(self) -> None:
        image_buffer = io.BytesIO()
        Image.new("RGB", (20, 10), "white").save(image_buffer, format="JPEG")
        request = main.LocateRequest(
            task_type="SORTING",
            product_name="雪碧罐装",
            level="L1",
            hand="left",
            image_name="sprite.jpg",
            image_base64=base64.b64encode(image_buffer.getvalue()).decode("ascii"),
        )
        with (
            tempfile.TemporaryDirectory() as monitor_directory,
            patch.object(
                main,
                "lookup_sku_by_name",
                return_value={"sku_id": "SKU_001", "name": "雪碧罐装"},
            ),
            patch.object(
                main,
                "locate_product_in_image",
                side_effect=main.HTTPException(
                    status_code=502,
                    detail="Qwen3 无法形成跨采样共识",
                ),
            ),
            patch.object(main, "MONITOR_IMAGE_DIR", Path(monitor_directory)),
        ):
            result = main.locate_product_debug(
                request,
                capture_inference_errors=True,
            )

        self.assertEqual(result.error_status_code, 502)
        self.assertEqual(result.error, "Qwen3 无法形成跨采样共识")
        self.assertEqual(result.image_size, [20, 10])
        self.assertTrue(base64.b64decode(result.image_base64))
        self.assertEqual(result.qwen_bboxes, [])
        self.assertEqual(result.instances, [])

    def test_public_locate_response_has_normalized_bbox_and_mask(self) -> None:
        mask = png_base64((100, 100))
        debug_response = main.LocateDebugResponse(
            sku_id="SKU_001",
            product_name="可口可乐",
            image_name="frame_rgb.jpg",
            image_path="C:/monitor/frame_rgb.jpg",
            image_base64=base64.b64encode(b"image").decode("ascii"),
            image_media_type="image/jpeg",
            image_size=[100, 100],
            instances=[
                main.LocatedInstance(
                    bbox=[10, 20, 30, 40],
                    mask=mask,
                    score=0.9,
                )
            ],
        )

        response = main.make_locate_response(debug_response)

        self.assertEqual(response.product_name, "可口可乐")
        self.assertEqual(response.bbox, [101, 201, 301, 401])
        self.assertEqual(response.mask, mask)
        self.assertEqual(response.image_path, "C:/monitor/frame_rgb.jpg")
        self.assertEqual(
            set(response.model_dump()),
            {"product_name", "bbox", "mask", "image_path"},
        )
        self.assertEqual(
            main.normalize_bbox_to_1_1000([-10, 0, 100, 120], [100, 100]),
            [1, 1, 1000, 1000],
        )

    def test_public_locate_uses_selected_instance_from_debug_response(self) -> None:
        center = main.LocatedInstance(
            bbox=[40, 40, 60, 60],
            mask="center",
        )
        selected = main.LocatedInstance(
            bbox=[70, 40, 90, 60],
            mask="selected",
        )
        debug_response = main.LocateDebugResponse(
            sku_id="SKU_001",
            product_name="可口可乐",
            image_name="frame_rgb.jpg",
            image_path="C:/monitor/frame_rgb.jpg",
            image_base64=base64.b64encode(b"image").decode("ascii"),
            image_media_type="image/jpeg",
            image_size=[100, 100],
            instances=[center, selected],
            selected_instance=selected,
            selected_instance_index=2,
        )

        response = main.make_locate_response(debug_response)

        self.assertEqual(response.mask, "selected")
        self.assertEqual(response.bbox, [700, 401, 900, 600])

    def test_pick_selection_rejects_narrow_occluded_center_candidate(self) -> None:
        narrow_center = main.LocatedInstance(
            bbox=[325, 140, 379, 346],
            mask="narrow-center",
            score=0.95,
        )
        expected = main.LocatedInstance(
            bbox=[375, 133, 460, 349],
            mask="expected",
            score=0.95,
        )
        instances = [
            main.LocatedInstance(
                bbox=[158, 133, 256, 381],
                mask="left-complete",
                score=0.95,
            ),
            main.LocatedInstance(
                bbox=[277, 148, 329, 333],
                mask="left-narrow",
                score=0.95,
            ),
            narrow_center,
            expected,
            main.LocatedInstance(
                bbox=[506, 155, 587, 336],
                mask="right-complete",
                score=0.95,
            ),
        ]

        selected = main.select_pick_instance(instances, [640, 480])

        self.assertIs(selected, expected)
        self.assertNotIn(
            narrow_center,
            main.keep_visibly_complete_pick_candidates(instances),
        )

    def test_pick_selection_keeps_complete_center_candidate(self) -> None:
        expected = main.LocatedInstance(
            bbox=[280, 140, 360, 340],
            mask="center",
            score=0.8,
        )
        instances = [
            expected,
            main.LocatedInstance(
                bbox=[400, 140, 488, 340],
                mask="right",
                score=0.9,
            ),
        ]

        selected = main.select_pick_instance(instances, [640, 480])

        self.assertIs(selected, expected)

    def test_estimate_instance_depth_uses_mask_median(self) -> None:
        depth_image = Image.new("I", (4, 2))
        depth_image.putdata([700, 710, 720, 730, 5000, 0, 5000, 5000])
        instance = main.LocatedInstance(
            bbox=[0, 0, 4, 2],
            mask=mask_base64_with_pixel_count((4, 2), 4),
        )

        depth_mm = main.estimate_instance_depth_mm(
            instance,
            depth_image,
            min_valid_pixels=4,
        )

        self.assertEqual(depth_mm, 715.0)

    def test_depth_selection_rejects_candidate_behind_both_neighbors(self) -> None:
        left_front = main.LocatedInstance(
            bbox=[190, 140, 270, 340],
            mask="left-front",
            depth_mm=700,
        )
        occluded_center = main.LocatedInstance(
            bbox=[280, 140, 360, 340],
            mask="occluded-center",
            depth_mm=800,
        )
        expected = main.LocatedInstance(
            bbox=[365, 140, 445, 340],
            mask="right-front",
            depth_mm=710,
        )
        instances = [left_front, occluded_center, expected]

        selected = main.select_pick_instance(instances, [640, 480])

        self.assertIs(selected, expected)
        self.assertNotIn(
            occluded_center,
            main.keep_depth_unoccluded_pick_candidates(instances),
        )

    def test_depth_selection_prefers_front_row_before_image_center(self) -> None:
        center_back = main.LocatedInstance(
            bbox=[280, 140, 360, 340],
            mask="center-back",
            depth_mm=800,
        )
        expected = main.LocatedInstance(
            bbox=[400, 140, 480, 340],
            mask="right-front",
            depth_mm=700,
        )

        selected = main.select_pick_instance([center_back, expected], [640, 480])

        self.assertIs(selected, expected)

    def test_depth_selection_uses_center_within_same_front_row(self) -> None:
        left_front = main.LocatedInstance(
            bbox=[160, 140, 240, 340],
            mask="left-front",
            depth_mm=700,
        )
        expected = main.LocatedInstance(
            bbox=[280, 140, 360, 340],
            mask="center-front",
            depth_mm=725,
        )
        back = main.LocatedInstance(
            bbox=[400, 140, 480, 340],
            mask="right-back",
            depth_mm=800,
        )

        front_row = main.keep_front_row_pick_candidates(
            [left_front, expected, back]
        )
        selected = main.select_pick_instance(
            [left_front, expected, back],
            [640, 480],
        )

        self.assertEqual(front_row, [left_front, expected])
        self.assertIs(selected, expected)

    def test_depth_selection_falls_back_to_center_without_valid_depth(self) -> None:
        expected = main.LocatedInstance(
            bbox=[280, 140, 360, 340],
            mask="center",
        )
        right = main.LocatedInstance(
            bbox=[400, 140, 480, 340],
            mask="right",
            depth_mm=None,
        )

        selected = main.select_pick_instance([expected, right], [640, 480])

        self.assertIs(selected, expected)
    def test_hard_case_selects_requested_location_level(self) -> None:
        location, match = main.select_location_for_level(
            {
                "locations": ["H2_F_L5_C02", "H2_F_L4_C02"],
            },
            "L5",
        )
        self.assertEqual(location, "H2_F_L5_C02")
        self.assertEqual(match.group("level"), "5")

    def test_non_configured_hand_is_ordinary_case(self) -> None:
        instance = main.LocatedInstance(bbox=[1, 2, 3, 4], mask=png_base64((10, 10)))
        filtered, debug = main.apply_hard_case_ordering(
            [instance],
            product={
                "name": "脉动芒果口味",
                "locations": ["H2_F_L4_C02", "H2_F_L5_C02"],
            },
            task_type="SORTING",
            level="L4",
            hand="right",
        )
        self.assertEqual(filtered, [instance])
        self.assertIsNone(debug)

    def test_hard_case_maps_left_order_and_drops_rear_bottle(self) -> None:
        mask = png_base64((120, 100))
        names = ["脉动观梅止渴饮", "脉动芒果口味", "脉动菠萝口味", "脉动猫薄荷瓶"]
        instances = []
        for index in range(4):
            instances.append(
                main.LocatedInstance(
                    bbox=[index * 25 + 2, 20, index * 25 + 20, 90],
                    mask=mask,
                    score=0.9,
                    source_qwen_index=index,
                )
            )
        instances.append(
            main.LocatedInstance(
                bbox=[28, 5, 42, 55],
                mask=mask_base64_with_pixel_count((120, 100), 2000),
                score=0.8,
                source_qwen_index=1,
            )
        )
        with patch.object(
            main,
            "lookup_sku_row",
            return_value=[{"name": name} for name in names],
        ):
            filtered, debug = main.apply_hard_case_ordering(
                instances,
                product={
                    "name": "脉动芒果口味",
                    "locations": ["H2_F_L5_C02", "H2_F_L4_C02"],
                },
                task_type="SORTING",
                level="L4",
                hand="left",
            )
        self.assertIsNotNone(debug)
        self.assertEqual(debug.target_location, "H2_F_L4_C02")
        self.assertEqual([group.mapped_product_name for group in debug.groups], names)
        self.assertEqual(len(filtered), 4)
        selected = [item for item in filtered if item.is_selected]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].mapped_product_name, "脉动芒果口味")

    def test_red_shelf_front_line_keeps_perspective_edge_product(self) -> None:
        image = Image.new("RGB", (240, 140), "#202830")
        pixels = image.load()
        for x in range(image.width):
            shelf_y = round(112 - 0.08 * x)
            for y in range(shelf_y - 2, shelf_y + 3):
                pixels[x, y] = (220, 30, 20)
        shelf_line = main.detect_red_shelf_front_line(image)
        self.assertIsNotNone(shelf_line)

        mask = png_base64((240, 140))
        front_left = main.LocatedInstance(
            bbox=[10, 25, 65, 109], mask=mask,
        )
        front_right_truncated = main.LocatedInstance(
            bbox=[180, 38, 239, 98], mask=mask,
        )
        rear = main.LocatedInstance(
            bbox=[90, 20, 145, 66], mask=mask,
        )
        filtered = main.keep_front_depth_row(
            [front_left, front_right_truncated, rear], shelf_line,
        )
        self.assertEqual(filtered, [front_left, front_right_truncated])

    def test_hard_case_filters_rear_bridge_before_overlap_deduplication(self) -> None:
        mask = png_base64((140, 120))
        front_left = main.LocatedInstance(
            bbox=[10, 20, 60, 100], mask=mask, score=0.9,
        )
        rear_bridge = main.LocatedInstance(
            bbox=[40, 10, 90, 70], mask=mask, score=0.95,
        )
        front_right = main.LocatedInstance(
            bbox=[70, 20, 120, 100], mask=mask, score=0.9,
        )
        groups = main.split_instances_into_display_groups(
            [front_left, rear_bridge, front_right],
            shelf_front_line=(0.0, 105.0),
        )
        self.assertEqual(groups, [[front_left], [front_right]])

    def test_left_camera_maps_partial_visible_prefix(self) -> None:
        mask = png_base64((100, 100))
        names = ["脉动观梅止渴饮", "脉动芒果口味", "脉动菠萝口味", "脉动猫薄荷瓶"]
        instances = [
            main.LocatedInstance(
                bbox=[index * 30 + 2, 10, index * 30 + 25, 90],
                mask=mask,
                source_qwen_index=index,
            )
            for index in range(3)
        ]
        with patch.object(
            main,
            "lookup_sku_row",
            return_value=[{"name": name} for name in names],
        ):
            filtered, debug = main.apply_hard_case_ordering(
                instances,
                product={
                    "name": "脉动菠萝口味",
                    "locations": ["H2_F_L4_C03"],
                },
                task_type="SORTING",
                level="L4",
                hand="left",
            )
        self.assertEqual(
            [group.mapped_product_name for group in debug.groups],
            names[:3],
        )
        self.assertEqual(len(filtered), 3)
        self.assertEqual(
            [item.mapped_product_name for item in filtered if item.is_selected],
            ["脉动菠萝口味"],
        )

    def test_right_camera_maps_partial_visible_suffix(self) -> None:
        mask = png_base64((100, 100))
        names = ["镇江香醋", "蒸鱼豉油", "薄盐生抽"]
        instances = [
            main.LocatedInstance(
                bbox=[index * 35 + 2, 10, index * 35 + 28, 90],
                mask=mask,
                source_qwen_index=index,
            )
            for index in range(2)
        ]
        with patch.object(
            main,
            "lookup_sku_row",
            return_value=[{"name": name} for name in names],
        ):
            _, debug = main.apply_hard_case_ordering(
                instances,
                product={"name": "蒸鱼豉油", "locations": ["H1_B_L5_C05"]},
                task_type="SORTING",
                level="L5",
                hand="right",
            )
        self.assertEqual(
            [group.mapped_product_name for group in debug.groups],
            ["薄盐生抽", "蒸鱼豉油"],
        )

    def test_partial_visible_columns_reject_target_outside_view(self) -> None:
        mask = png_base64((100, 100))
        names = ["脉动观梅止渴饮", "脉动芒果口味", "脉动菠萝口味", "脉动猫薄荷瓶"]
        instances = [
            main.LocatedInstance(
                bbox=[2, 10, 25, 90],
                mask=mask,
                source_qwen_index=0,
            )
        ]
        with (
            patch.object(
                main,
                "lookup_sku_row",
                return_value=[{"name": name} for name in names],
            ),
            self.assertRaises(main.HTTPException) as raised,
        ):
            main.apply_hard_case_ordering(
                instances,
                product={
                    "name": "脉动菠萝口味",
                    "locations": ["H2_F_L4_C03"],
                },
                task_type="SORTING",
                level="L4",
                hand="left",
            )
        self.assertEqual(raised.exception.status_code, 404)

    def test_non_hard_case_keeps_original_instances(self) -> None:
        instance = main.LocatedInstance(
            bbox=[1, 2, 3, 4],
            mask=png_base64((10, 10)),
        )
        filtered, debug = main.apply_hard_case_ordering(
            [instance],
            product={"name": "可口可乐罐装", "locations": ["H2_F_L1_C01"]},
            task_type="SORTING",
            level="L1",
            hand="left",
        )
        self.assertEqual(filtered, [instance])
        self.assertIsNone(debug)

    def test_hard_case_maps_right_camera_from_right_edge(self) -> None:
        mask = png_base64((100, 100))
        names = ["镇江香醋", "蒸鱼豉油", "薄盐生抽"]
        instances = [
            main.LocatedInstance(
                bbox=[index * 30 + 2, 10, index * 30 + 24, 90],
                mask=mask,
                source_qwen_index=index,
            )
            for index in range(3)
        ]
        with patch.object(
            main,
            "lookup_sku_row",
            return_value=[{"name": name} for name in names],
        ):
            filtered, debug = main.apply_hard_case_ordering(
                instances,
                product={
                    "name": "蒸鱼豉油",
                    "locations": ["H1_B_L5_C05"],
                },
                task_type="SORTING",
                level="L5",
                hand="right",
            )
        self.assertEqual(debug.order_direction, "right_to_left")
        self.assertEqual(
            [group.mapped_product_name for group in debug.groups],
            list(reversed(names)),
        )
        selected = [item for item in filtered if item.is_selected]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].mapped_product_name, "蒸鱼豉油")

    def test_alien_zero_sugar_uses_right_camera_override(self) -> None:
        mask = png_base64((100, 100))
        names = ["外星人电解质水青柠口味0糖", "外星人电解质水青柠口味"]
        instances = [
            main.LocatedInstance(
                bbox=[index * 35 + 2, 10, index * 35 + 28, 90],
                mask=mask,
                source_qwen_index=index,
            )
            for index in range(2)
        ]
        with patch.object(
            main,
            "lookup_sku_row",
            return_value=[{"name": name} for name in names],
        ):
            filtered, debug = main.apply_hard_case_ordering(
                instances,
                product={
                    "name": "外星人电解质水青柠口味0糖",
                    "locations": ["H2_F_L4_C05"],
                },
                task_type="SORTING",
                level="L4",
                hand="right",
            )
        self.assertEqual(debug.preferred_hand, "right")
        self.assertEqual(debug.order_direction, "right_to_left")
        self.assertEqual(debug.target_location, "H2_F_L4_C05")
        selected = [item for item in filtered if item.is_selected]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].mapped_product_name, names[0])

    def test_maiydong_catnip_is_not_in_hard_case_scope(self) -> None:
        hard_case = main.hard_case_group_for_product(
            "脉动猫薄荷瓶",
            "SORTING",
            "L4",
            "right",
        )
        self.assertIsNone(hard_case)

    def test_bbq_products_use_left_camera(self) -> None:
        group_id, config = main.hard_case_group_for_product(
            "草原红太阳烧烤料香辣味",
            "SORTING",
            "L1",
            "left",
        )
        self.assertEqual(group_id, "bbq_seasoning")
        self.assertEqual(config.hand_for_product("草原红太阳烧烤料香辣味"), "left")
        sauce_group_id, sauce_config = main.hard_case_group_for_product(
            "草原红太阳烧烤酱香辣",
            "SORTING",
            "L1",
            "left",
        )
        self.assertEqual(sauce_group_id, "bbq_sauce_spicy")
        self.assertEqual(sauce_config.hand_for_product("草原红太阳烧烤酱香辣"), "left")
        original_hard_case = main.hard_case_group_for_product(
            "草原红太阳烧烤酱原味",
            "SORTING",
            "L1",
            "left",
        )
        self.assertIsNone(original_hard_case)

    def test_hard_case_response_uses_selected_instead_of_image_center(self) -> None:
        mask = png_base64((100, 100))
        selected = main.LocatedInstance(
            bbox=[1, 10, 10, 30],
            mask=mask,
            is_selected=True,
        )
        center = main.LocatedInstance(bbox=[45, 45, 55, 55], mask=mask)
        debug = main.LocateDebugResponse(
            sku_id="SKU_085",
            product_name="脉动芒果口味",
            image_name="frame.jpg",
            image_path="/tmp/frame.jpg",
            image_base64="aW1hZ2U=",
            image_media_type="image/jpeg",
            image_size=[100, 100],
            instances=[selected, center],
            hard_case=main.HardCaseDebugInfo(
                group_id="maiydong",
                preferred_hand="left",
                actual_hand="left",
                order_direction="left_to_right",
                target_location="H2_F_L4_C02",
                target_level=4,
                target_column=2,
                standard_order=["脉动观梅止渴饮", "脉动芒果口味"],
                selected_group_index=2,
            ),
        )
        response = main.make_locate_response(debug)
        self.assertEqual(response.mask, selected.mask)
        self.assertEqual(response.bbox, main.normalize_bbox_to_1_1000(selected.bbox, [100, 100]))

    def test_debug_passes_postprocess_error_capture_to_pipeline(self) -> None:
        request = main.LocateRequest(
            task_type="SORTING",
            product_name="脉动芒果口味",
            level="L4",
            hand="left",
        )
        expected = main.LocateDebugResponse(
            sku_id="SKU_085",
            product_name="脉动芒果口味",
            image_name="frame.jpg",
            image_path="/tmp/frame.jpg",
            image_base64="aW1hZ2U=",
            image_media_type="image/jpeg",
            image_size=[100, 100],
            error="第一排候选不足",
            error_status_code=422,
        )
        with (
            patch.object(
                main,
                "lookup_sku_by_name",
                return_value={
                    "sku_id": "SKU_085",
                    "name": "脉动芒果口味",
                    "locations": ["H2_F_L4_C02"],
                },
            ),
            patch.object(main, "get_latest_rgb", return_value=Path("/tmp/frame.jpg")),
            patch.object(main, "locate_product_in_image", return_value=expected) as locate_mock,
        ):
            result = main.locate_product_debug(request, capture_inference_errors=True)
        self.assertEqual(result.error_status_code, 422)
        self.assertTrue(locate_mock.call_args.kwargs["capture_postprocess_errors"])


if __name__ == "__main__":
    unittest.main()
