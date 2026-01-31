# ytsum

YouTube動画を自動で文字起こし・要約するCLIツール

## ワークフロー

```
YouTube URL → yt-dlp (音声DL) → Whisper (文字起こし) → Claude (要約) → 出力
```

## インストール

```bash
git clone https://github.com/yourname/ytsum.git
cd ytsum
uv sync
```

### 必要な外部ツール

- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [ffmpeg](https://ffmpeg.org/)
- [Claude CLI](https://github.com/anthropics/claude-code)
- [1Password CLI](https://developer.1password.com/docs/cli/) (APIモード使用時)

## 使い方

```bash
# 基本（OpenAI API使用）
uv run python ytsum.py "https://youtu.be/VIDEO_ID"

# faster-whisper使用（無料・ローカル処理）
uv run python ytsum.py "https://youtu.be/VIDEO_ID" --whisper-mode faster

# モデル指定
uv run python ytsum.py "https://youtu.be/VIDEO_ID" --whisper-mode faster --whisper-model medium

# Markdown形式で出力
uv run python ytsum.py "https://youtu.be/VIDEO_ID" --format markdown
```

## オプション

| オプション | 説明 | デフォルト |
|------------|------|------------|
| `--whisper-mode` | `api` / `faster` / `local` | `api` |
| `--whisper-model` | モデル名 | api: `whisper-1`, faster: `large-v3`, local: `base` |
| `--language` | 言語コード (ja, en等) | 自動検出 |
| `--format` | `text` / `markdown` | `text` |
| `--log-dir` | ログ保存先 | `./log` |
| `-v, --verbose` | 詳細出力 | off |

## 出力

処理結果は `log/YYMMDD-HHMMSS/` に保存されます：

```
log/
└── 260131-170447/
    ├── transcript.txt  # 文字起こし
    └── summary.txt     # 要約
```

## Whisperモード比較

| モード | 処理場所 | コスト | 精度 |
|--------|----------|--------|------|
| `api` | OpenAI サーバー | 有料 | 高 |
| `faster` | ローカル | 無料 | 高 |
| `local` | ローカル | 無料 | 中 |

## 設定

### OpenAI API キー

1Passwordに`openai-apikey`という名前でAPIキーを保存してください。

## License

MIT
