# 音声文字起こし＆要約ツール

このプロジェクトは、Kotoba Whisper モデルを使用して日本語音声ファイルを文字起こしし、さらにGemini APIを利用してその内容を要約・タスク抽出するツールです。

## セットアップ

1.  **依存パッケージのインストール**:
    必要な Python パッケージをインストールします。

    ```bash
    uv sync
    ```
    

2.  **FFmpeg のインストール**:
    長時間の音声ファイルを自動で分割するために FFmpeg が必要です。お使いの OS に合わせてインストールしてください。
    *   **macOS (Homebrew)**: `brew install ffmpeg`
    *   **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
    *   **Windows**: [公式サイト](https://ffmpeg.org/download.html) からダウンロードし、パスを通してください。

3.  **Hugging Face 認証**:
    話者分離機能を利用する場合、Hugging Face で以下のモデルの利用規約に同意し、CLI でログインする必要があります。

    ```bash
    huggingface-cli login
    ```
    *   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
    *   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

4.  **環境変数ファイル (.env) の作成**:
    プロジェクトのルートディレクトリに `.env` ファイルを作成し、以下の内容を記述します。

    ```dotenv
    # Google Gemini API キー
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

    # 各種設定 (JSON形式で記述)
    CONFIG='''{
        "audio_file": "audio/your_audio.mp3",
        "transcription_output_dir": "output/transcriptions",
        "summary_output_dir": "output/summaries"
    }'''
    ```
    *   `GEMINI_API_KEY`: Google AI Studio などで取得した API キーを設定してください。
    *   `CONFIG`:
        *   `audio_file`: 文字起こししたい音声ファイルのパス（リポジトリルートからの相対パスまたは絶対パス）。
        *   `transcription_output_dir`: 文字起こし結果（.txt）を保存するディレクトリのパス。
        *   `summary_output_dir`: 要約結果（.md）を保存するディレクトリのパス。
    *   **注意**: `CONFIG` の値は JSON 文字列として記述するため、シングルクォート (`'`) で囲み、内部のダブルクォート (`"`) はそのまま使用してください。パス区切り文字は `/` を使用してください。

## 使い方

1.  **音声ファイルの準備**:
    `.env` ファイルの `CONFIG` で指定したパスに、文字起こししたい音声ファイル（例: `audio/your_audio.mp3`）を配置します。必要に応じて `audio` ディレクトリを作成してください。
2.  **スクリプトの実行**:
    以下のコマンドを実行します。

    ```bash
    uv run main.py
    ```
    スクリプトは `.env` ファイルの設定を読み込み、以下の処理を自動で行います。
    *   音声ファイルの分割（必要な場合）
    *   文字起こし（タイムスタンプ、話者分離付き）
    *   文字起こし結果のテキストファイル (.txt) 保存
    *   Gemini API を利用した要約・タスク抽出
    *   要約結果の Markdown ファイル (.md) 保存

## 機能

-   **日本語音声の文字起こし**: Kotoba Whisper モデルを使用。
-   **長時間音声の自動分割**: FFmpeg を利用して1時間ごとにファイルを分割し、処理後に結合。
-   **句読点の自動追加**: Whisper の機能を利用。
-   **タイムスタンプ付き出力**: 各発言の開始・終了時刻を記録。ファイル名に基づいて絶対時刻も付与。
-   **LLM による要約・タスク抽出**: Gemini API を利用して文字起こし結果から要約と次の日のタスク候補を生成。
-   **出力形式**:
    -   文字起こし結果: `.txt` ファイル（指定ディレクトリに保存）
    -   要約・タスク: `.md` ファイル（指定ディレクトリに保存）

## 活用例

-   会議や打ち合わせの議事録作成補助
-   インタビューや取材の文字起こし
-   講演、セミナー、ポッドキャストの内容記録
-   日々の音声メモからのタスク整理・日報作成
