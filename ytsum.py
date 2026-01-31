#!/usr/bin/env python3
"""YouTube video summarization tool."""

import argparse
from datetime import datetime
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import openai
from faster_whisper import WhisperModel


def get_openai_api_key() -> str:
    """Get OpenAI API key from 1Password."""
    result = subprocess.run(
        ["op", "item", "get", "openai-apikey", "--fields", "notesPlain"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to get OpenAI API key from 1Password. "
            "Make sure 'op' CLI is installed and you're signed in."
        )
    return result.stdout.strip()


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def split_audio(audio_path: Path, chunk_duration: int = 600, verbose: bool = False) -> list[Path]:
    """Split audio file into chunks of specified duration.

    Args:
        audio_path: Path to the audio file
        chunk_duration: Duration of each chunk in seconds (default: 600 = 10 minutes)
        verbose: Print detailed output

    Returns:
        List of paths to the chunk files
    """
    duration = get_audio_duration(audio_path)

    if duration <= chunk_duration:
        return [audio_path]

    if verbose:
        print(f"Splitting {duration:.0f}s audio into {chunk_duration}s chunks", file=sys.stderr)

    chunks = []
    chunk_index = 0
    start_time = 0

    while start_time < duration:
        chunk_path = audio_path.with_stem(f"{audio_path.stem}_chunk{chunk_index:03d}")
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", str(audio_path),
            "-ss", str(start_time),
            "-t", str(chunk_duration),
            "-acodec", "copy",
            str(chunk_path),
        ]

        if verbose:
            print(f"Creating chunk {chunk_index}: {start_time}s - {min(start_time + chunk_duration, duration):.0f}s", file=sys.stderr)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        chunks.append(chunk_path)
        chunk_index += 1
        start_time += chunk_duration

    return chunks


def get_video_metadata(url: str, verbose: bool = False) -> dict:
    """Get video metadata from YouTube URL using yt-dlp.

    Args:
        url: YouTube video URL
        verbose: Print detailed output

    Returns:
        Dictionary containing video metadata
    """
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        url,
    ]

    if verbose:
        print(f"Fetching metadata...", file=sys.stderr)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata fetch failed: {result.stderr}")

    data = json.loads(result.stdout)

    return {
        "title": data.get("title", "Unknown"),
        "channel": data.get("channel", data.get("uploader", "Unknown")),
        "upload_date": data.get("upload_date", "Unknown"),  # YYYYMMDD形式
        "duration": data.get("duration", 0),
        "view_count": data.get("view_count", 0),
        "url": url,
        "video_id": data.get("id", "Unknown"),
    }


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """Sanitize string for use in filename.

    Args:
        name: Original string
        max_length: Maximum length of the result

    Returns:
        Sanitized string safe for filenames
    """
    # ファイル名に使えない文字を置換
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    # 連続するアンダースコアを1つに
    sanitized = re.sub(r'_+', '_', sanitized)
    # 前後の空白とアンダースコアを除去
    sanitized = sanitized.strip().strip('_')
    # 長さ制限
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')
    return sanitized


def download_audio(url: str, output_dir: Path, verbose: bool = False) -> Path:
    """Download audio from YouTube URL using yt-dlp.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the audio file
        verbose: Print detailed output

    Returns:
        Path to the downloaded audio file
    """
    output_template = str(output_dir / "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "mp3",
        "--audio-quality", "0",  # Best quality
        "-o", output_template,
        "--print", "after_move:filepath",  # Print final path
        url,
    ]

    if verbose:
        print(f"Running: {' '.join(cmd)}", file=sys.stderr)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    # Get the output file path from stdout
    audio_path = Path(result.stdout.strip().split("\n")[-1])

    if not audio_path.exists():
        raise RuntimeError(f"Downloaded file not found: {audio_path}")

    return audio_path


def transcribe_audio(
    audio_path: Path,
    mode: str = "api",
    model: str = "whisper-1",
    language: str | None = None,
    verbose: bool = False,
) -> str:
    """Transcribe audio file using Whisper.

    Args:
        audio_path: Path to the audio file
        mode: 'api' for OpenAI API, 'faster' for faster-whisper, 'local' for whisper CLI
        model: Model name (whisper-1 for API, large-v3/medium/small/base for faster/local)
        language: Language code (e.g., 'ja', 'en'), None for auto-detect
        verbose: Print detailed output

    Returns:
        Transcribed text
    """
    if mode == "api":
        # Split audio if longer than 10 minutes
        chunks = split_audio(audio_path, chunk_duration=600, verbose=verbose)

        if len(chunks) > 1:
            print(f"Audio split into {len(chunks)} chunks for API processing", file=sys.stderr)

        transcripts = []
        for i, chunk_path in enumerate(chunks):
            if len(chunks) > 1:
                print(f"Transcribing chunk {i + 1}/{len(chunks)}...", file=sys.stderr)
            transcripts.append(_transcribe_api(chunk_path, model, language, verbose))

        return "\n".join(transcripts)
    elif mode == "faster":
        return _transcribe_faster(audio_path, model, language, verbose)
    else:
        return _transcribe_local(audio_path, model, language, verbose)


def _transcribe_api(
    audio_path: Path,
    model: str,
    language: str | None,
    verbose: bool,
) -> str:
    """Transcribe using OpenAI Whisper API."""
    if verbose:
        print(f"Transcribing with OpenAI API (model: {model})", file=sys.stderr)

    # Get API key from 1Password
    api_key = get_openai_api_key()
    client = openai.OpenAI(api_key=api_key)

    with open(audio_path, "rb") as audio_file:
        kwargs = {"model": model, "file": audio_file}
        if language:
            kwargs["language"] = language

        transcript = client.audio.transcriptions.create(**kwargs)

    return transcript.text


def _transcribe_faster(
    audio_path: Path,
    model: str,
    language: str | None,
    verbose: bool,
) -> str:
    """Transcribe using faster-whisper."""
    if verbose:
        print(f"Transcribing with faster-whisper (model: {model})", file=sys.stderr)

    # Determine device and compute type
    device = "auto"  # auto-detect cuda/cpu
    compute_type = "auto"  # auto-select based on device

    if verbose:
        print(f"Loading model {model}...", file=sys.stderr)

    whisper_model = WhisperModel(model, device=device, compute_type=compute_type)

    if verbose:
        print("Transcribing...", file=sys.stderr)

    segments, info = whisper_model.transcribe(
        str(audio_path),
        beam_size=5,
        language=language,
        vad_filter=True,  # Filter out silence
    )

    if verbose:
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})", file=sys.stderr)

    # Collect all segments
    texts = []
    for segment in segments:
        texts.append(segment.text.strip())

    return "\n".join(texts)


def _transcribe_local(
    audio_path: Path,
    model: str,
    language: str | None,
    verbose: bool,
) -> str:
    """Transcribe using local whisper CLI."""
    if verbose:
        print(f"Transcribing with local whisper (model: {model})", file=sys.stderr)

    cmd = [
        "whisper",
        str(audio_path),
        "--model", model,
        "--output_format", "txt",
        "--output_dir", str(audio_path.parent),
    ]

    if language:
        cmd.extend(["--language", language])

    if verbose:
        print(f"Running: {' '.join(cmd)}", file=sys.stderr)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"whisper failed: {result.stderr}")

    # Read the output text file
    txt_path = audio_path.with_suffix(".txt")
    if not txt_path.exists():
        raise RuntimeError(f"Transcription output not found: {txt_path}")

    return txt_path.read_text()


def summarize_text(transcript: str, format: str = "text", verbose: bool = False) -> str:
    """Summarize text using claude CLI.

    Args:
        transcript: Text to summarize
        format: Output format ('text' or 'markdown')
        verbose: Print detailed output

    Returns:
        Summary text
    """
    if format == "markdown":
        prompt = f"""以下の文字起こしテキストを日本語で要約してください。
Markdown形式で出力してください。

## 出力形式
- 概要（2-3文で簡潔に）
- 主要なポイント（箇条書き）
- 重要な引用やキーワード（あれば）

## 文字起こしテキスト
{transcript}
"""
    else:
        prompt = f"""以下の文字起こしテキストを日本語で要約してください。

主要なポイントを箇条書きで、概要を2-3文で出力してください。

文字起こしテキスト:
{transcript}
"""

    if verbose:
        print("Summarizing with claude CLI", file=sys.stderr)

    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"claude failed: {result.stderr}")

    return result.stdout


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Summarize YouTube videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://youtu.be/VIDEO_ID"
  %(prog)s "https://youtu.be/VIDEO_ID" --whisper-mode faster
  %(prog)s "https://youtu.be/VIDEO_ID" --whisper-mode faster --whisper-model large-v3
  %(prog)s "https://youtu.be/VIDEO_ID" --format markdown
""",
    )

    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--whisper-mode",
        choices=["api", "faster", "local"],
        default="api",
        help="Whisper mode: api (OpenAI), faster (faster-whisper), local (whisper CLI)",
    )
    parser.add_argument(
        "--whisper-model",
        default=None,
        help="Whisper model (api: whisper-1, faster/local: large-v3/medium/small/base)",
    )
    parser.add_argument(
        "--language",
        help="Language code (e.g., ja, en). Default: auto-detect",
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("log"),
        help="Log directory (default: ./log)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Set default model based on mode
    if args.whisper_model is None:
        if args.whisper_mode == "api":
            args.whisper_model = "whisper-1"
        elif args.whisper_mode == "faster":
            args.whisper_model = "large-v3"
        else:  # local
            args.whisper_model = "base"

    try:
        start_time = datetime.now()

        # Step 1: Get video metadata
        print("Fetching video metadata...", file=sys.stderr)
        metadata = get_video_metadata(args.url, args.verbose)
        if args.verbose:
            print(f"Title: {metadata['title']}", file=sys.stderr)
            print(f"Channel: {metadata['channel']}", file=sys.stderr)

        # Create log directory with timestamp and title
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        sanitized_title = sanitize_filename(metadata["title"])
        output_dir = args.log_dir / f"{timestamp}_{sanitized_title}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create temp directory for audio processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Step 2: Download audio
            print("Downloading audio...", file=sys.stderr)
            audio_path = download_audio(args.url, temp_path, args.verbose)
            if args.verbose:
                print(f"Downloaded: {audio_path}", file=sys.stderr)

            # Step 3: Transcribe
            print("Transcribing audio...", file=sys.stderr)
            transcript = transcribe_audio(
                audio_path,
                mode=args.whisper_mode,
                model=args.whisper_model,
                language=args.language,
                verbose=args.verbose,
            )
            if args.verbose:
                print(f"Transcript length: {len(transcript)} chars", file=sys.stderr)

            # Save transcript
            transcript_path = output_dir / "transcript.txt"
            transcript_path.write_text(transcript)
            print(f"Transcript saved to: {transcript_path}", file=sys.stderr)

            # Step 4: Summarize
            print("Summarizing...", file=sys.stderr)
            summary = summarize_text(transcript, args.format, args.verbose)

            # Save summary with metadata header
            summary_path = output_dir / "summary.txt"
            elapsed_time = datetime.now() - start_time
            elapsed_seconds = int(elapsed_time.total_seconds())
            elapsed_str = f"{elapsed_seconds // 60}分{elapsed_seconds % 60}秒"
            summary_with_metadata = f"""タイトル: {metadata['title']}
チャンネル: {metadata['channel']}
公開日: {metadata['upload_date']}
URL: {metadata['url']}
取得日: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
処理時間: {elapsed_str}

---

{summary}"""
            summary_path.write_text(summary_with_metadata)
            print(f"Summary saved to: {summary_path}", file=sys.stderr)

            # Also print summary to stdout
            print(summary)

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
