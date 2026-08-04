import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from receipt_recognizer.api import OpenAICompatibleClient
from receipt_recognizer.config import Settings
from receipt_recognizer.errors import APIResponseError


class StubHandler(BaseHTTPRequestHandler):
    seen_authorization = None
    last_path = None

    def do_GET(self):
        type(self).seen_authorization = self.headers.get("Authorization")
        type(self).last_path = self.path
        if self.path == "/v1/models":
            self._json(200, {"data": [{"id": "Qwen3-VL-4B-Instruct"}]})
        elif self.path == "/openapi.json":
            self._json(200, {"paths": {}})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        type(self).seen_authorization = self.headers.get("Authorization")
        type(self).last_path = self.path
        length = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(length))
        if self.path == "/v1/chat/completions":
            self._json(
                200,
                {
                    "choices": [
                        {
                            "message": {"content": "连接成功"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"total_tokens": 3},
                },
            )
        else:
            self._json(404, {"error": {"message": "not found"}})

    def log_message(self, format, *args):
        return

    def _json(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        cls.thread = threading.Thread(
            target=cls.server.serve_forever,
            daemon=True,
        )
        cls.thread.start()
        host, port = cls.server.server_address
        cls.base_url = f"http://{host}:{port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_omits_authorization_when_key_is_absent(self):
        client = OpenAICompatibleClient(
            Settings(base_url=self.base_url, api_key=None)
        )
        client.list_models()
        self.assertIsNone(StubHandler.seen_authorization)

    def test_sends_authorization_when_key_is_present(self):
        client = OpenAICompatibleClient(
            Settings(base_url=self.base_url, api_key="secret")
        )
        client.list_models()
        self.assertEqual(
            StubHandler.seen_authorization,
            "Bearer secret",
        )

    def test_openapi_url_is_outside_v1(self):
        client = OpenAICompatibleClient(
            Settings(base_url=self.base_url)
        )
        self.assertEqual(client.get_openapi(), {"paths": {}})
        self.assertEqual(StubHandler.last_path, "/openapi.json")

    def test_chat_content_is_extracted(self):
        client = OpenAICompatibleClient(
            Settings(base_url=self.base_url)
        )
        response = client.create_chat_completion(
            [{"role": "user", "content": "hello"}]
        )
        self.assertEqual(response.content, "连接成功")
        self.assertEqual(response.finish_reason, "stop")

    def test_http_error_is_classified(self):
        client = OpenAICompatibleClient(
            Settings(base_url=f"{self.base_url}/wrong")
        )
        with self.assertRaises(APIResponseError) as caught:
            client.list_models()
        self.assertEqual(caught.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()

