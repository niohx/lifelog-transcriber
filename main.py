import torch
from transformers import pipeline
import os
import datetime
import pickle
import sys
import numpy as np
import tempfile
import subprocess
import shutil
import re
from pathlib import Path
from query_llm import compose_summary

# 設定の読み込み
import dotenv

dotenv.load_dotenv()
#.envファイルにはCONFIGというキーがあり、その値はjson形式で記載されています。
# json形式は以下のように記載します。
# {
#     "audio_file": "audio/sample.mp3",
#     "transcription_output_dir": "transcription_directory_path",
#     "summary_output_dir": "summary_directory_path "
# }

CONFIG = dotenv.get_key(".env", "CONFIG")

# パスの設定


def setup_paths():
    """パスの設定を行う"""
    try:
        audio_path = Path(CONFIG["audio_file"]).resolve()
        transcription_dir = Path(CONFIG["transcription_output_dir"]).resolve()
        summary_dir = Path(CONFIG["summary_output_dir"]).resolve()

        # 出力ディレクトリの存在確認と作成
        transcription_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)

        return audio_path, transcription_dir, summary_dir
    except Exception as e:
        print(f"パスの設定中にエラーが発生しました: {e}")
        sys.exit(1)


def check_ffmpeg_installed():
    """FFmpegがインストールされているか確認"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("FFmpegがインストールされていません。インストールしてください。")
        return False


def get_audio_duration(audio_path):
    """FFmpegを使用して音声ファイルの長さを取得（秒単位）"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"音声ファイルの長さを取得できませんでした: {e}")
        return 0


def split_audio_file_ffmpeg(audio_path, segment_length_sec=3600):  # デフォルトは1時間（3600秒）
    """FFmpegを使用して音声ファイルを指定された長さのセグメントに分割する"""
    if not check_ffmpeg_installed():
        return [audio_path], None  # 一時ディレクトリはなし

    temp_dir = None  # 初期化
    try:
        print(f"音声ファイルの長さを確認中: {audio_path}")
        duration_sec = get_audio_duration(audio_path)

        # 音声が指定された長さより短い場合は分割しない
        if duration_sec <= segment_length_sec:
            return [audio_path], None  # 一時ディレクトリはなし

        # 分割したファイルのパスを保存するリスト
        segment_files = []

        # 音声を分割
        num_segments = int(np.ceil(duration_sec / segment_length_sec))
        print(
            f"音声を {num_segments} 個のセグメントに分割します（各 {segment_length_sec/60:.1f} 分）")

        temp_dir = tempfile.mkdtemp()  # ここでtemp_dirを生成
        base_filename = Path(audio_path).stem
        file_ext = Path(audio_path).suffix

        for i in range(num_segments):
            start_sec = i * segment_length_sec
            segment_path = os.path.join(
                temp_dir, f"{base_filename}_part{i+1}{file_ext}")

            # FFmpegコマンドを実行して分割
            cmd = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-ss", str(start_sec),
                "-t", str(segment_length_sec),
                "-c", "copy",  # コーデックをコピー（高速）
                segment_path
            ]

            print(f"セグメント {i+1}/{num_segments} を作成中...")
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                segment_files.append(segment_path)
                print(f"セグメント {i+1}/{num_segments} を保存しました: {segment_path}")
            else:
                print(f"セグメント {i+1}/{num_segments} の作成に失敗しました")

        return segment_files, temp_dir
    except Exception as e:
        print(f"音声ファイルの分割中にエラーが発生しました: {e}")
        # エラー発生時でも作成されていたらクリーンアップ対象として返す
        if temp_dir and os.path.exists(temp_dir):
            # ただし、このケースでは部分的に作成されたファイルが残る可能性がある
            return [audio_path], temp_dir  # 元のファイルと、作成された一時ディレクトリを返す
        return [audio_path], None  # エラーが発生した場合は元のファイルを返し、一時ディレクトリはなし


def transcribe_audio(audio_path, model_id="kotoba-tech/kotoba-whisper-v2.2", add_punctuation=True, add_diarization=True):
    # デバイスとデータ型の設定
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {
    }

    print(f"Using device: {device}, torch_dtype: {torch_dtype}")

    # モデルの読み込み
    pipe = pipeline(
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs=model_kwargs,
        batch_size=8,
        trust_remote_code=True,
    )

    # 推論の実行
    result = pipe(
        audio_path,
        chunk_length_s=15,
        add_punctuation=add_punctuation,
        add_silence_start=0.5,
        add_silence_end=0.5
    )

    return result


def format_timestamp(seconds):
    """秒数を [HH:MM:SS.mm] 形式に変換する"""
    return datetime.timedelta(seconds=seconds)


def extract_date_time_from_filename(filename):
    """ファイル名（YYMMDD_HHMM形式）から日付と時刻を抽出する"""
    match = re.match(r'(\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})', filename)
    if match:
        year, month, day, hour, minute = match.groups()
        # 年は20を先頭に付ける（例：25→2025年）
        full_year = f"20{year}"
        return f"{full_year}-{month}-{day} {hour}:{minute}:00"
    return None


def adjust_timestamp(timestamp, offset_seconds):
    """タイムスタンプを指定された秒数だけ調整する"""
    if timestamp is None:
        return None
    return timestamp + offset_seconds


def save_transcription_to_txt(result, output_path, time_offset=0):
    with open(output_path, "w", encoding="utf-8") as f:
        # 時系列順にソート
        chunks = result.get("chunks", [])
        chunks.sort(key=lambda x: x["timestamp"][0]
                    if x["timestamp"][0] is not None else 0)

        # ファイル名から日時情報を抽出
        base_name = Path(output_path).stem
        date_time_str = extract_date_time_from_filename(base_name)

        # 時系列順に出力
        for chunk in chunks:
            if chunk["timestamp"][0] is None or chunk["timestamp"][1] is None:
                continue

            # タイムスタンプを調整（時間オフセットを適用）
            start_seconds = adjust_timestamp(
                chunk["timestamp"][0], time_offset)
            end_seconds = adjust_timestamp(chunk["timestamp"][1], time_offset)

            # フォーマットされたタイムスタンプ
            start_time = format_timestamp(start_seconds)
            end_time = format_timestamp(end_seconds)

            # 日付情報を含むタイムスタンプを作成
            if date_time_str:
                # ファイル名から抽出した日時＋タイムスタンプの秒数を使用
                start_dt = datetime.datetime.strptime(
                    date_time_str, "%Y-%m-%d %H:%M:%S")
                end_dt = datetime.datetime.strptime(
                    date_time_str, "%Y-%m-%d %H:%M:%S")

                # 秒数を追加
                start_dt = start_dt + datetime.timedelta(seconds=start_seconds)
                end_dt = end_dt + datetime.timedelta(seconds=end_seconds)

                # フォーマット
                formatted_start = start_dt.strftime(
                    "%Y-%m-%d %H:%M:%S.%f")[:-3]
                formatted_end = end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                # 出力行を作成
                if "speaker_id" in chunk:
                    f.write(
                        f"[{formatted_start} --> {formatted_end}] 話者 {chunk['speaker_id']}: {chunk['text']}\n")
                else:
                    f.write(
                        f"[{formatted_start} --> {formatted_end}] {chunk['text']}\n")
            else:
                # 日付情報が取得できない場合は従来の形式を使用
                if "speaker_id" in chunk:
                    f.write(
                        f"[{start_time} --> {end_time}] 話者 {chunk['speaker_id']}: {chunk['text']}\n")
                else:
                    f.write(f"[{start_time} --> {end_time}] {chunk['text']}\n")

    print(f"文字起こし結果を {output_path} に保存しました。")
    return output_path


def merge_transcription_files(file_paths, output_path):
    """複数の文字起こしファイルを1つのファイルにマージする"""
    with open(output_path, "w", encoding="utf-8") as out_file:
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as in_file:
                out_file.write(in_file.read())
                out_file.write("\n")

    print(f"文字起こし結果を {output_path} にマージしました。")
    return output_path


def cleanup_temp_files(file_paths, temp_dir_path=None):
    """一時ファイルを削除する"""
    for file_path in file_paths:
        try:
            # 一時ディレクトリ内のファイルのみ削除
            if os.path.exists(file_path) and temp_dir_path and file_path.startswith(temp_dir_path):
                os.remove(file_path)
        except Exception as e:
            print(f"ファイル {file_path} の削除中にエラーが発生しました: {e}")

    # 一時ディレクトリを削除
    if temp_dir_path and os.path.exists(temp_dir_path) and temp_dir_path.startswith(tempfile.gettempdir()):
        try:
            shutil.rmtree(temp_dir_path, ignore_errors=True)
            print(f"一時ディレクトリ {temp_dir_path} を削除しました。")
        except Exception as e:
            print(f"一時ディレクトリ {temp_dir_path} の削除中にエラーが発生しました: {e}")


def process_segment(segment_file, base_name, segment_index, segment_length_sec):
    """個別のセグメントを処理する"""
    print(f"セグメント {segment_index+1} の文字起こしを実行中...")

    # 各セグメントの出力ファイルパス
    segment_output_path = f"{base_name}_part{segment_index+1}_transcription.txt"

    # pickleファイルのパス
    pickle_path = f"{base_name}_part{segment_index+1}_result.pkl"

    try:
        # 文字起こしの実行（1時間 = 3600秒のオフセットを適用）
        result = transcribe_audio(segment_file)

        # 結果を一時的にpickleで保存
        with open(pickle_path, "wb") as f:
            pickle.dump(result, f)

        # 時間オフセットを計算
        time_offset = segment_index * segment_length_sec

        # 結果をテキストファイルに保存（時間オフセットを適用）
        segment_output = save_transcription_to_txt(
            result, segment_output_path, time_offset)

        # エラーがなければpickleファイルを削除
        os.remove(pickle_path)

        return segment_output

    except Exception as e:
        print(f"セグメント {segment_index+1} の処理中にエラーが発生しました: {e}", file=sys.stderr)
        print(f"pickleファイルは {pickle_path} に保存されています。")
        return None


def process_all_segments(segment_files, base_name, segment_length_sec):
    """全てのセグメントを処理する"""
    segment_outputs = []

    for i, segment_file in enumerate(segment_files):
        segment_output = process_segment(
            segment_file, base_name, i, segment_length_sec)
        if segment_output:
            segment_outputs.append(segment_output)

    return segment_outputs


def handle_segment_outputs(segment_outputs, final_output_path):
    """セグメントの出力ファイルを処理する"""
    if len(segment_outputs) > 1:
        merge_transcription_files(segment_outputs, final_output_path)

        # マージ後、個別のセグメント出力ファイルを削除
        for segment_output in segment_outputs:
            if os.path.exists(segment_output):
                os.remove(segment_output)
    elif len(segment_outputs) == 1:
        # セグメントが1つだけの場合はファイル名を変更
        if os.path.exists(segment_outputs[0]) and segment_outputs[0] != final_output_path:
            os.replace(segment_outputs[0], final_output_path)


def process_audio_file(audio_path, output_directory_path=None):
    """音声ファイルを処理するメイン関数

    Args:
        audio_path (str | Path): 入力音声ファイルの絶対パス
        output_directory_path (str | Path, optional): 出力ディレクトリの絶対パス。指定がない場合は入力ファイルと同じディレクトリに出力

    Raises:
        ValueError: パスが絶対パスでない場合
    """
    # Pathオブジェクトに変換
    audio_path = Path(audio_path)
    if output_directory_path is not None:
        output_directory_path = Path(output_directory_path)

    # 絶対パスのチェック
    if not audio_path.is_absolute():
        raise ValueError("audio_pathは絶対パスである必要があります")

    if output_directory_path is not None and not output_directory_path.is_absolute():
        raise ValueError("output_directory_pathは絶対パスである必要があります")

    # 出力ディレクトリの設定
    if output_directory_path is None:
        output_directory_path = audio_path.parent
    else:
        # 出力ディレクトリが存在しない場合は作成
        output_directory_path.mkdir(parents=True, exist_ok=True)

    # 出力ファイルのパス
    base_name = audio_path.stem
    final_output_path = output_directory_path / \
        f"{base_name}_transcription.txt"

    temp_dir_for_segments = None  # 初期化
    segment_length_for_processing = 3600  # デフォルトのセグメント長（秒）

    try:
        # 音声ファイルを分割（FFmpegを使用）
        # split_audio_file_ffmpeg に segment_length_for_processing を渡す
        segment_files, temp_dir_for_segments = split_audio_file_ffmpeg(
            str(audio_path), segment_length_sec=segment_length_for_processing)

        # 分割したセグメントごとに文字起こしを実行
        # process_all_segments に segment_length_for_processing を渡す
        segment_outputs = process_all_segments(
            segment_files, base_name, segment_length_for_processing)

        # 全セグメントの文字起こし結果をマージ
        handle_segment_outputs(segment_outputs, str(final_output_path))

        # 一時ファイルのクリーンアップ
        # segment_filesが元のオーディオパスと異なる（つまり分割が行われた）場合のみクリーンアップ
        if segment_files and (len(segment_files) > 1 or (len(segment_files) == 1 and segment_files[0] != str(audio_path))):
            # temp_dir_for_segments も渡す
            cleanup_temp_files(segment_files, temp_dir_for_segments)

    except Exception as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)

    return final_output_path


if __name__ == "__main__":
    # パスの設定
    audio_path, transcription_dir, summary_dir = setup_paths()

    # 文字起こしの実行
    transcription_output_path = process_audio_file(
        audio_path, transcription_dir)
    print(f"文字起こし結果を {transcription_output_path} に保存しました。")

    # サマリーの作成
    summary_output_path = compose_summary(
        transcription_output_path, summary_dir)
    print(f"サマリーを {summary_output_path} に保存しました。")

    # pickleファイルから結果を読み込む処理（コメントアウト）
    """
    try:
        # pickleファイルから結果を読み込む
        pickle_path = os.path.join(audio_dir, f"{base_name}_result.pkl")
        with open(pickle_path, "rb") as f:
            result = pickle.load(f)
        
        # 結果をテキストファイルに保存
        save_transcription_to_txt(result, final_output_path)
        
    except Exception as e:
        print(f"pickleファイルの読み込み中にエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)
    """
