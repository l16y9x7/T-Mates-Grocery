from setuptools import find_packages, setup


setup(
    name="receipt-recognizer",
    version="0.1.0",
    description=(
        "A minimal Qwen3-VL receipt recognizer for an "
        "OpenAI-compatible API."
    ),
    packages=find_packages(include=["receipt_recognizer*"]),
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.110",
        "pillow>=10",
        "python-multipart>=0.0.9",
        "uvicorn[standard]>=0.29",
    ],
    extras_require={
        "ocr": ["paddleocr>=2.7"],
    },
    entry_points={
        "console_scripts": [
            "receipt-recognizer=receipt_recognizer.cli:main",
            "receipt-evaluate=receipt_recognizer.evaluation:main",
            "receipt-ocr=receipt_recognizer.ocr:main",
            "qwen-probe=receipt_recognizer.probe:main",
        ]
    },
)
