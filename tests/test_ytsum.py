"""ytsum.pyのユニットテスト。"""

import json
from unittest.mock import patch, MagicMock

import pytest

from ytsum import get_video_title, main


MOCK_METADATA = {
    "title": "テスト動画",
    "channel": "テストチャンネル",
    "upload_date": "20260214",
    "duration": 120,
    "view_count": 1000,
    "url": "https://youtu.be/test123",
    "video_id": "test123",
}


class TestGetVideoTitle:
    """get_video_title()のテスト。"""

    @patch("ytsum.subprocess.run")
    def test_正常にタイトルを取得(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="テスト動画タイトル\n",
        )
        title = get_video_title("https://youtu.be/test123")
        assert title == "テスト動画タイトル"
        mock_run.assert_called_once_with(
            ["yt-dlp", "--get-title", "https://youtu.be/test123"],
            capture_output=True,
            text=True,
        )

    @patch("ytsum.subprocess.run")
    def test_yt_dlp失敗時にエラー(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="ERROR: Video unavailable",
        )
        with pytest.raises(RuntimeError, match="動画タイトルの取得に失敗"):
            get_video_title("https://youtu.be/invalid")


class TestNoSummaryFlag:
    """--no-summaryフラグのテスト。"""

    @patch("ytsum.get_video_metadata", return_value=MOCK_METADATA)
    @patch("ytsum.transcribe_audio", return_value="これはテストの文字起こしです。")
    @patch("ytsum.download_audio")
    def test_no_summary時にJSON出力(self, mock_download, mock_transcribe, mock_metadata, tmp_path, capsys):
        mock_download.return_value = tmp_path / "audio.mp3"
        (tmp_path / "audio.mp3").touch()

        with patch("sys.argv", ["ytsum", "https://youtu.be/test123", "--no-summary", "--log-dir", str(tmp_path / "log")]):
            main()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["title"] == "テスト動画"
        assert output["transcript"] == "これはテストの文字起こしです。"

    @patch("ytsum.get_video_metadata", return_value=MOCK_METADATA)
    @patch("ytsum.transcribe_audio", return_value="テスト")
    @patch("ytsum.download_audio")
    def test_no_summary時にsummarize_textは呼ばれない(self, mock_download, mock_transcribe, mock_metadata, tmp_path):
        mock_download.return_value = tmp_path / "audio.mp3"
        (tmp_path / "audio.mp3").touch()

        with patch("ytsum.summarize_text") as mock_summarize, \
             patch("sys.argv", ["ytsum", "https://youtu.be/test123", "--no-summary", "--log-dir", str(tmp_path / "log")]):
            main()
            mock_summarize.assert_not_called()

    @patch("ytsum.get_video_metadata", return_value=MOCK_METADATA)
    @patch("ytsum.summarize_text", return_value="要約結果")
    @patch("ytsum.transcribe_audio", return_value="テスト")
    @patch("ytsum.download_audio")
    def test_no_summaryなしではsummarize_textが呼ばれる(self, mock_download, mock_transcribe, mock_summarize, mock_metadata, tmp_path):
        mock_download.return_value = tmp_path / "audio.mp3"
        (tmp_path / "audio.mp3").touch()

        with patch("sys.argv", ["ytsum", "https://youtu.be/test123", "--log-dir", str(tmp_path / "log")]):
            main()
            mock_summarize.assert_called_once()
