import os
import subprocess
from pathlib import Path

def merge_mp3_files(input_dir, output_file):
    # 入力ディレクトリ内のMP3ファイルを取得
    mp3_files = [
        Path("audio/250514_0738_01.mp3"),
        Path("audio/250514_1302.mp3"),
        Path("audio/250514_1308.mp3")
    ]
    
    # ファイルの存在確認
    for mp3_file in mp3_files:
        if not mp3_file.exists():
            print(f"エラー: ファイル {mp3_file} が見つかりません。")
            return
    
    if not mp3_files:
        print("MP3ファイルが見つかりませんでした。")
        return
    
    # 一時ファイルのパスを作成
    temp_list_file = Path("temp_file_list.txt")
    
    # ファイルリストを作成
    try:
        with open(temp_list_file, "w", encoding="utf-8") as f:
            for mp3_file in mp3_files:
                f.write(f"file '{mp3_file.absolute()}'\n")
        
        # ffmpegコマンドを実行
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(temp_list_file),
            "-c", "copy",
            output_file
        ]
        
        print("実行コマンド:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"MP3ファイルの結合が完了しました。出力ファイル: {output_file}")
    
    except subprocess.CalledProcessError as e:
        print(f"エラーが発生しました: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
    
    finally:
        # 一時ファイルを削除
        if temp_list_file.exists():
            temp_list_file.unlink()

if __name__ == "__main__":
    input_directory = "audio"  # audioディレクトリを使用
    output_file = "audio/250514_0738.mp3"  # 出力ファイル名
    
    merge_mp3_files(input_directory, output_file)

