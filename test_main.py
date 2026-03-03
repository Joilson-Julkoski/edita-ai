import pytest
from unittest.mock import patch, MagicMock
from main import transcribe, edit, search, interpret, run


def test_transcribe_returns_transcript_text(tmp_path):
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake audio data")

    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "hello world"}
    mock_response.raise_for_status.return_value = None

    with patch("main.requests.post", return_value=mock_response) as mock_post:
        result = transcribe(str(audio_file))

    assert result == "hello world"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs[0][0] == "https://api.lemonfox.ai/v1/audio/transcriptions"


def test_transcribe_sends_auth_header(tmp_path, monkeypatch):
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake audio data")
    monkeypatch.setenv("LEMONFOX_API_KEY", "test-key")

    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "hello"}
    mock_response.raise_for_status.return_value = None

    with patch("main.requests.post", return_value=mock_response) as mock_post:
        transcribe(str(audio_file))

    headers = mock_post.call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer test-key"


def test_transcribe_raises_on_api_error(tmp_path):
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake audio data")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("API error")

    with patch("main.requests.post", return_value=mock_response):
        with pytest.raises(Exception, match="API error"):
            transcribe(str(audio_file))


FAKE_YAML = """
timeline:
  - start: "00:00:01"
    end: "00:00:05"
    type: text
    content: "Hello world"
    position: center
  - start: "00:00:06"
    end: "00:00:10"
    type: text
    content: "Second point"
    position: left
"""


def test_edit_returns_dict_with_timeline():
    mock_response = MagicMock()
    mock_response.text = FAKE_YAML

    with patch("main.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = mock_response
        result = edit("some transcript text")

    assert isinstance(result, dict)
    assert "timeline" in result


def test_edit_timeline_items_have_required_fields():
    mock_response = MagicMock()
    mock_response.text = FAKE_YAML

    with patch("main.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = mock_response
        result = edit("some transcript text")

    for item in result["timeline"]:
        assert "start" in item
        assert "end" in item
        assert "type" in item
        assert "content" in item
        assert "position" in item


def test_edit_position_is_valid():
    mock_response = MagicMock()
    mock_response.text = FAKE_YAML

    with patch("main.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = mock_response
        result = edit("some transcript text")

    valid_positions = {"center", "left", "right"}
    for item in result["timeline"]:
        assert item["position"] in valid_positions


def test_edit_sends_transcript_in_prompt():
    mock_response = MagicMock()
    mock_response.text = FAKE_YAML

    with patch("main.genai.Client") as mock_client_cls:
        mock_generate = mock_client_cls.return_value.models.generate_content
        mock_generate.return_value = mock_response
        edit("my unique transcript content")

    call_args = mock_generate.call_args
    prompt = call_args[1]["contents"]
    assert "my unique transcript content" in prompt


def test_search_returns_empty_dict():
    result = search({"timeline": []})
    assert result == {}


FAKE_EDIT_CONFIG = {
    "timeline": [
        {"start": "00:00:01", "end": "00:00:05", "type": "text", "content": "Hello world", "position": "center"},
        {"start": "00:00:06", "end": "00:00:10", "type": "text", "content": "Second point", "position": "left"},
    ]
}


def test_interpret_returns_output_path(tmp_path):
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake audio")
    output_file = str(tmp_path / "output.mp4")

    with patch("main.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = interpret(str(audio_file), FAKE_EDIT_CONFIG, {}, output_path=output_file)

    assert result == output_file


def test_interpret_calls_ffmpeg(tmp_path):
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake audio")

    with patch("main.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        interpret(str(audio_file), FAKE_EDIT_CONFIG, {}, output_path=str(tmp_path / "out.mp4"))

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffmpeg"


def test_interpret_builds_drawtext_filters(tmp_path):
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake audio")

    with patch("main.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        interpret(str(audio_file), FAKE_EDIT_CONFIG, {}, output_path=str(tmp_path / "out.mp4"))

    cmd = " ".join(mock_run.call_args[0][0])
    assert "Hello world" in cmd
    assert "Second point" in cmd
    assert "between(t" in cmd


def test_interpret_maps_positions_to_coordinates(tmp_path):
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake audio")
    config = {
        "timeline": [
            {"start": "00:00:01", "end": "00:00:03", "type": "text", "content": "A", "position": "center"},
            {"start": "00:00:03", "end": "00:00:06", "type": "text", "content": "B", "position": "left"},
            {"start": "00:00:06", "end": "00:00:09", "type": "text", "content": "C", "position": "right"},
        ]
    }

    with patch("main.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        interpret(str(audio_file), config, {}, output_path=str(tmp_path / "out.mp4"))

    cmd = " ".join(mock_run.call_args[0][0])
    assert "(w-text_w)/2" in cmd  # center x
    assert "x=50" in cmd          # left x
    assert "w-text_w-50" in cmd   # right x


def test_interpret_raises_on_ffmpeg_error(tmp_path):
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake audio")

    with patch("main.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="ffmpeg error")
        with pytest.raises(RuntimeError, match="ffmpeg"):
            interpret(str(audio_file), FAKE_EDIT_CONFIG, {}, output_path=str(tmp_path / "out.mp4"))


def test_run_pipeline(tmp_path):
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_bytes(b"fake audio")
    output_file = str(tmp_path / "output.mp4")

    mock_http = MagicMock()
    mock_http.json.return_value = {"text": "hello transcript"}
    mock_http.raise_for_status.return_value = None

    mock_gemini = MagicMock()
    mock_gemini.text = FAKE_YAML

    with patch("main.requests.post", return_value=mock_http), \
         patch("main.genai.Client") as mock_client_cls, \
         patch("main.subprocess.run") as mock_run:
        mock_client_cls.return_value.models.generate_content.return_value = mock_gemini
        mock_run.return_value = MagicMock(returncode=0)
        result = run(str(audio_file), output_path=output_file)

    assert result == output_file
