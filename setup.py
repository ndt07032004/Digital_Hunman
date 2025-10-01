from setuptools import find_packages, setup

setup(
    name="Digital_Human",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="TourGuideBot - AI assistant for Vietnamese travel and culture",
    url="https://your-repo.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "flask", "langchain", "pinecone-client", "huggingface-hub", "transformers",
        "faster-whisper", "torch", "requests", "python-dotenv", "pyaudio", "websockets",
        "flask-cors", "f5-tts"
    ],
)