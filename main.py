import os
import subprocess
import requests
import yaml
from dotenv import load_dotenv
from google import genai

load_dotenv()

LEMONFOX_API_URL = "https://api.lemonfox.ai/v1/audio/transcriptions"


def transcribe(audio_path: str) -> str:
    """Transcribe audio file and return transcript text."""
    api_key = os.getenv("LEMONFOX_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}

    with open(audio_path, "rb") as f:
        response = requests.post(
            LEMONFOX_API_URL,
            headers=headers,
            files={"file": f},
        )

    response.raise_for_status()
    return response.json()["text"]


def _load_prompt(name: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "prompts", f"{name}.txt")
    with open(path) as f:
        return f.read()


def edit(transcript: str) -> dict:
    """Send transcript to Gemini and return edit config as dict."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=_load_prompt("edit").format(transcript=transcript),
    )
    return yaml.safe_load(response.text)


def search(edit_config: dict) -> dict:
    """Find and download sources referenced in edit config."""
    return {}


def _time_to_seconds(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _position_to_x(position: str) -> str:
    return {
        "center": "(w-text_w)/2",
        "left": "50",
        "right": "w-text_w-50",
    }[position]


def interpret(audio_path: str, edit_config: dict, sources: dict, output_path: str = "output.mp4") -> str:
    """Use ffmpeg to render final video, return output path."""
    filters = []
    for item in edit_config["timeline"]:
        start = _time_to_seconds(item["start"])
        end = _time_to_seconds(item["end"])
        x = _position_to_x(item["position"])
        content = item["content"].replace("'", "\\'")
        filters.append(
            f"drawtext=text='{content}':x={x}:y=(h-text_h)/2"
            f":fontsize=48:fontcolor=white:enable='between(t,{start},{end})'"
        )

    vf = ",".join(filters)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:size=1280x720",
        "-i", audio_path,
        "-shortest",
        "-vf", vf,
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    return output_path


def run(audio_path: str, output_path: str = "output.mp4") -> str:
    transcript = transcribe(audio_path)
    edit_config = edit(transcript)
    sources = search(edit_config)
    return interpret(audio_path, edit_config, sources, output_path=output_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <audio_file>")
        sys.exit(1)
    run(sys.argv[1])
