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


def get_video_title(url: str) -> str:
    """yt-dlpで動画タイトルを取得する。"""
    result = subprocess.run(
        ["yt-dlp", "--get-title", url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"動画タイトルの取得に失敗: {result.stderr}")
    return result.stdout.strip()


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


def is_channel_url(url: str) -> bool:
    """Check if URL is a YouTube channel URL.

    Args:
        url: URL to check

    Returns:
        True if URL is a channel URL
    """
    channel_patterns = [
        r"youtube\.com/@[\w-]+",
        r"youtube\.com/channel/[\w-]+",
        r"youtube\.com/c/[\w-]+",
        r"youtube\.com/user/[\w-]+",
    ]
    return any(re.search(pattern, url) for pattern in channel_patterns)


def get_channel_info(url: str, verbose: bool = False) -> dict:
    """Get channel info from YouTube channel URL.

    Args:
        url: YouTube channel URL
        verbose: Print detailed output

    Returns:
        Dictionary containing channel info
    """
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        "--playlist-end", "1",
        url,
    ]

    if verbose:
        print("Fetching channel info...", file=sys.stderr)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp channel info failed: {result.stderr}")

    # 最初の動画情報からチャンネル名を取得
    first_line = result.stdout.strip().split("\n")[0]
    data = json.loads(first_line)

    # チャンネル名を取得（URLから抽出も試みる）
    channel_name = data.get("channel") or data.get("uploader")
    if not channel_name:
        # URLから@以降を抽出
        match = re.search(r"@([\w-]+)", url)
        if match:
            channel_name = match.group(1)
        else:
            channel_name = "unknown_channel"

    return {
        "channel_name": channel_name,
        "channel_url": url,
    }


def get_channel_videos(url: str, limit: int, verbose: bool = False) -> list[dict]:
    """Get video list from YouTube channel.

    Args:
        url: YouTube channel URL
        limit: Maximum number of videos to fetch
        verbose: Print detailed output

    Returns:
        List of video info dictionaries
    """
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        "--playlist-end", str(limit),
        url,
    ]

    if verbose:
        print(f"Fetching up to {limit} videos from channel...", file=sys.stderr)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp channel fetch failed: {result.stderr}")

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        data = json.loads(line)
        videos.append({
            "video_id": data.get("id"),
            "title": data.get("title"),
            "url": data.get("url") or f"https://www.youtube.com/watch?v={data.get('id')}",
            "duration": data.get("duration"),
        })

    return videos


def get_processed_video_ids(channel_dir: Path) -> set[str]:
    """Get set of already processed video IDs from channel directory.

    Args:
        channel_dir: Path to channel log directory

    Returns:
        Set of processed video IDs
    """
    if not channel_dir.exists():
        return set()

    processed = set()
    for subdir in channel_dir.iterdir():
        if subdir.is_dir():
            # ディレクトリ名からvideo_idを抽出（video_id_タイトル形式）
            video_id = subdir.name.split("_")[0]
            if video_id:
                processed.add(video_id)
    return processed


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


def sanitize_filename(name: str, max_length: int = 50) -> str:
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
    output_template = str(output_dir / "%(id)s.%(ext)s")
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
    elif mode == "mlx":
        return _transcribe_mlx(audio_path, model, language, verbose)
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
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed. "
            "Install with: uv pip install faster-whisper"
        )

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


def _transcribe_mlx(
    audio_path: Path,
    model: str,
    language: str | None,
    verbose: bool,
) -> str:
    """Transcribe using mlx-whisper (Apple Silicon最適化)."""
    try:
        import mlx_whisper
    except ImportError:
        raise RuntimeError(
            "mlx-whisper is not installed. "
            "Install with: uv pip install 'ytsum[mlx]'"
        )

    # モデル名をHuggingFaceリポジトリ名にマッピング
    model_map = {
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        "large-v2": "mlx-community/whisper-large-v2-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "tiny": "mlx-community/whisper-tiny-mlx",
    }
    repo = model_map.get(model, model)

    if verbose:
        print(f"Transcribing with mlx-whisper (model: {repo})", file=sys.stderr)
        print(f"Loading model...", file=sys.stderr)

    kwargs = {"path_or_hf_repo": repo}
    if language:
        kwargs["language"] = language

    result = mlx_whisper.transcribe(str(audio_path), **kwargs)

    if verbose and "language" in result:
        print(f"Detected language: {result['language']}", file=sys.stderr)

    return result["text"]


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


def process_single_video(args, output_dir: Path | None = None) -> str | None:
    """Process a single YouTube video.

    Args:
        args: Parsed command line arguments
        output_dir: Optional output directory override

    Returns:
        Summary text or None if failed
    """
    start_time = datetime.now()

    # Step 1: Get video metadata
    print("Fetching video metadata...", file=sys.stderr)
    metadata = get_video_metadata(args.url, args.verbose)
    if args.verbose:
        print(f"Title: {metadata['title']}", file=sys.stderr)
        print(f"Channel: {metadata['channel']}", file=sys.stderr)

    # Create log directory with timestamp and title
    if output_dir is None:
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

        if args.no_summary:
            # metadata.jsonをlog/に保存（transcript全文は含めない）
            metadata_for_save = {
                "title": metadata["title"],
                "channel": metadata["channel"],
                "upload_date": metadata["upload_date"],
                "duration": metadata["duration"],
                "url": metadata["url"],
                "video_id": metadata["video_id"],
            }
            metadata_path = output_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(metadata_for_save, ensure_ascii=False, indent=2)
            )
            print(f"Metadata saved to: {metadata_path}", file=sys.stderr)

            # stdoutには軽量JSONのみ出力（transcript全文を含めない）
            output = json.dumps(
                {
                    "log_dir": str(output_dir),
                    "title": metadata["title"],
                    "video_id": metadata["video_id"],
                },
                ensure_ascii=False,
            )
            print(output)
            return None

        # Step 4: Summarize
        print("Summarizing...", file=sys.stderr)
        summary = summarize_text(transcript, args.format, args.verbose)

        # Save summary with metadata header
        summary_path = output_dir / "summary.md"
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

        return summary


def process_channel(args) -> None:
    """Process videos from a YouTube channel.

    Args:
        args: Parsed command line arguments
    """
    # チャンネル情報を取得
    channel_info = get_channel_info(args.url, args.verbose)
    channel_name = sanitize_filename(channel_info["channel_name"])
    channel_dir = args.log_dir / "channel" / channel_name

    print(f"Channel: {channel_info['channel_name']}", file=sys.stderr)
    print(f"Log directory: {channel_dir}", file=sys.stderr)

    # 処理済み動画IDを取得
    processed_ids = get_processed_video_ids(channel_dir)
    if processed_ids:
        print(f"Already processed: {len(processed_ids)} videos", file=sys.stderr)

    # 動画一覧を取得
    videos = get_channel_videos(args.url, args.limit, args.verbose)
    print(f"Found {len(videos)} videos", file=sys.stderr)

    # 処理対象をフィルタリング
    to_process = [v for v in videos if v["video_id"] not in processed_ids]
    skipped = len(videos) - len(to_process)
    if skipped > 0:
        print(f"Skipping {skipped} already processed videos", file=sys.stderr)

    if not to_process:
        print("No new videos to process", file=sys.stderr)
        return

    print(f"Processing {len(to_process)} videos...", file=sys.stderr)
    print("-" * 50, file=sys.stderr)

    # 各動画を処理
    for i, video in enumerate(to_process, 1):
        video_id = video["video_id"]
        title = video["title"] or "unknown"
        sanitized_title = sanitize_filename(title, max_length=80)
        output_dir = channel_dir / f"{video_id}_{sanitized_title}"

        print(f"\n[{i}/{len(to_process)}] {title}", file=sys.stderr)

        # argsをコピーしてURLを上書き
        video_args = argparse.Namespace(**vars(args))
        video_args.url = video["url"]

        try:
            process_single_video(video_args, output_dir)
        except Exception as e:
            print(f"Error processing {video_id}: {e}", file=sys.stderr)
            continue

    print("\n" + "=" * 50, file=sys.stderr)
    print(f"Completed: {len(to_process)} videos processed", file=sys.stderr)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Summarize YouTube videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 単一動画の要約
  %(prog)s "https://youtu.be/VIDEO_ID"
  %(prog)s "https://youtu.be/VIDEO_ID" --whisper-mode faster

  # チャンネルから最新N件を要約
  %(prog)s "https://www.youtube.com/@CHANNEL" --limit 5
  %(prog)s "https://www.youtube.com/@CHANNEL" --limit 10 --whisper-mode faster
""",
    )

    parser.add_argument("url", help="YouTube video or channel URL")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of videos to process for channel URLs (default: 5)",
    )
    parser.add_argument(
        "--whisper-mode",
        choices=["api", "faster", "mlx", "local"],
        default="api",
        help="Whisper mode: api (OpenAI), faster (faster-whisper), mlx (Apple Silicon), local (whisper CLI)",
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
        "--no-summary",
        action="store_true",
        help="文字起こしのみ実行し、要約をスキップ。JSON形式で出力",
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
        elif args.whisper_mode in ("faster", "mlx"):
            args.whisper_model = "large-v3"
        else:  # local
            args.whisper_model = "base"

    try:
        # チャンネルURLの場合
        if is_channel_url(args.url):
            process_channel(args)
        else:
            process_single_video(args)

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
