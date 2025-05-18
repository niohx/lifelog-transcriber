import sys
from pathlib import Path
from google import genai
import dotenv
import re

dotenv.load_dotenv()

gemini_api_key = dotenv.get_key(".env", "GEMINI_API_KEY")

def query_llm(transcription_text_path:Path):
    # APIキーの存在チェック
    if not gemini_api_key:
        print("エラー: 環境変数ファイル (.env) に GEMINI_API_KEY が設定されていません。", file=sys.stderr)
        return None
        
    try:
        client = genai.Client(api_key=gemini_api_key)
    except Exception as e:
        print(f"エラー: genai.Client の初期化に失敗しました: {e}", file=sys.stderr)
        return None

    model = "gemini-2.5-flash-preview-04-17"
    
    try:
        # テキストファイルの内容をアップロードする
        print(f"文字起こしファイルをアップロード中: {transcription_text_path}")
        transcription = client.files.upload(file=transcription_text_path)
    except FileNotFoundError:
        print(f"エラー: 指定された文字起こしファイルが見つかりません: {transcription_text_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"エラー: ファイルのアップロード中に予期せぬエラーが発生しました: {e}", file=sys.stderr)
        return None
    
    # プロンプトとテキスト内容を組み合わせる
    prompt = f"""
    ファイルは１日の文字起こしです。
    こちらの文字起こしを簡潔にまとめた後、次の日以降のタスクになりそうな部分をピックアップしてもらえませんか？ 

    また、返信のフォーマットはmarkdownで返信してください。
    返信のフォーマットは以下の通りです。
    
    ## まとめ
    (まとめの内容)

    ## タスク
    (タスクの内容)

    """
    
    try:
        # コンテンツを生成
        print("LLMにサマリー生成をリクエスト中...")
        response = client.models.generate_content(
            model=model,
            contents=[prompt,transcription]
        )
        print("サマリー生成完了。")
        print(f"レスポンスの型: {type(response)}")  # デバッグ用
        print(f"レスポンスの内容: {response}")  # デバッグ用
        
        # レスポンスからテキストを取得
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            print("エラー: レスポンスからテキストを取得できませんでした。", file=sys.stderr)
            return None
    except Exception as e:
        print(f"エラー: サマリー生成中に予期せぬエラーが発生しました: {e}", file=sys.stderr)
        # GoogleAPIErrorもここで捕捉される
        if hasattr(e, 'message'): # エラーオブジェクトにmessage属性があれば表示
             print(f"サマリー生成エラー: API関連のエラーが発生しました ({e.message})。", file=sys.stderr)
        else:
            print("サマリー生成エラー: 不明なエラーが発生しました。", file=sys.stderr)
        return None

def compose_summary(transcription_text_path:Path, output_directory_path:Path=None):
    summary = query_llm(transcription_text_path)
    
    if summary is None:
        print("エラー: サマリーの生成に失敗しました。", file=sys.stderr)
        return None
    
    # ファイル名から日付を抽出（YYMMDD形式）
    base_name = transcription_text_path.stem
    date_match = re.match(r'(\d{6})', base_name)
    if date_match:
        date_str = date_match.group(1)
        # YYMMDD形式をYYYYMMDD形式に変換
        year = f"20{date_str[:2]}"
        month = date_str[2:4]
        day = date_str[4:6]
        summary_filename = f"{year}{month}{day}_summary.md"
    else:
        # 日付が抽出できない場合は従来の形式を使用
        summary_filename = base_name.replace('_transcription', '') + '_summary.md'
    
    # 出力ディレクトリが指定されていない場合は、入力ファイルの親ディレクトリの親ディレクトリを使用
    if output_directory_path is None:
        output_directory_path = transcription_text_path.parent.parent
    
    # 出力パスを生成
    summary_path = output_directory_path / summary_filename
    
    # 日報のタイトルを追加して保存
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('#日報\n\n')
        f.write(summary)
    
    return summary_path



# if __name__ == "__main__":
#     # .envファイルに設定されたパスを使用するか、以下で直接指定
#     # config_str = dotenv.get_key(".env", "CONFIG")
#     # if config_str:
#     #     import json
#     #     config = json.loads(config_str)
#     #     base_path = Path(config["transcription_output_dir"]) 
#     # else:
#     #     # .envがない場合やキーが存在しない場合のフォールバック（例）
#     #     base_path = Path("fallback_path")
    
#     # または、テスト用に直接パスを指定
#     base_path = Path("./output/transcriptions") # 例: リポジトリ内のパス
#     example_filename = "YYYYMMDD_HHMM_transcription.txt" # 一般的なファイル名形式
#     transcription_text_path = base_path / example_filename
    
#     # 出力ディレクトリも同様に設定可能
#     summary_output_dir = Path("./output/summaries") # 例: リポジトリ内のパス
#     summary_output_dir.mkdir(parents=True, exist_ok=True) # 存在しない場合に作成
    
#     # テスト実行（ファイルが存在しない場合はエラーになる点に注意）
#     if transcription_text_path.exists():
#         compose_summary(transcription_text_path, summary_output_dir)
#         print(f"テスト用のサマリーを {summary_output_dir} に出力試行しました。")
#     else:
#         print(f"エラー: テスト用の文字起こしファイルが見つかりません: {transcription_text_path}", file=sys.stderr)
#         print("テストを実行するには、上記パスにダミーファイルを作成するか、実際のファイルパスを指定してください。")

