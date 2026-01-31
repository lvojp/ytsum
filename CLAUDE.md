# ytsum プロジェクト設定

## YouTube動画要約の手順（厳守）

**必ず ytsum.py を使用すること。他の方法は禁止。**

### ワークフロー

```
YouTube URL → yt-dlp (音声DL) → faster-whisper (文字起こし) → Claude (要約)
```

### 実行コマンド

```bash
uv run python ytsum.py "<YouTube URL>" --whisper-mode faster
```

### 禁止事項

- `youtube_transcript_api` は使用禁止（IPブロックされやすい）
- `yt-dlp --write-auto-sub` による字幕取得は使用禁止
- 字幕APIへの直接アクセスは禁止
- 独自のスクリプトで処理しようとしない

### 理由

- 字幕APIはYouTubeからIPブロック（429エラー）を受けやすい
- 音声ダウンロード → Whisper文字起こし が最も安定した方法
- ytsum.py に全ての処理が実装済み

### トラブルシューティング

依存関係エラーが出た場合：
```bash
uv sync
```
