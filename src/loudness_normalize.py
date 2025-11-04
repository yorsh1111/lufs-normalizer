"""
ラウドネス正規化スクリプト
単一の音声ファイル（wav、mp3、flac形式）をITU-R BS.1770-3準拠で正規化します。
"""

import sys
import os
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import warnings
import pathlib

# librosaの警告を抑制
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


def get_file_extension(file_path):
    """
    ファイルの拡張子を取得

    Args:
        file_path: ファイルパス

    Returns:
        str: 小文字の拡張子（.を含まない）
    """
    return pathlib.Path(file_path).suffix.lower().lstrip('.')


def check_supported_format(file_path):
    """
    対応しているファイル形式かチェック

    Args:
        file_path: ファイルパス

    Returns:
        bool: 対応している形式であればTrue
    """
    supported_formats = ['wav', 'mp3', 'flac']
    extension = get_file_extension(file_path)
    return extension in supported_formats


def normalize_loudness(
    input_path,
    output_path,
    target_loudness=-14.0,
    true_peak_limit=-2.0,
):
    """
    音声ファイルをラウドネス正規化

    Args:
        input_path: 入力ファイルパス
        output_path: 出力ファイルパス
        target_loudness: 目標ラウドネス（LUFS）
        true_peak_limit: True Peakリミット（dBTP）

    Returns:
        dict: 処理結果の辞書
    """
    try:
        # ファイル形式をチェック
        if not check_supported_format(input_path):
            return {
                "status": "error",
                "error": f"対応していないファイル形式です: {input_path}。サポートされている形式: wav, mp3, flac"
            }
        
        # 音声ファイルを読み込み（ステレオを保持）
        audio_data, sr = librosa.load(input_path, sr=None, mono=False)

        # チャンネル数を確認
        if audio_data.ndim == 1:
            # モノラル音声
            is_stereo = False
            channels = 1
        else:
            # ステレオまたはマルチチャンネル音声
            is_stereo = True
            channels = audio_data.shape[0]
            # pyloudnorm用にshapeを(samples, channels)に変換
            audio_data = audio_data.T

        # pyloudnormのメーターを作成
        meter = pyln.Meter(sr, block_size=0.400)

        # 元のラウドネスを測定
        original_loudness = meter.integrated_loudness(audio_data)

        # 音声を正規化
        normalized_audio = pyln.normalize.loudness(
            audio_data, original_loudness, target_loudness
        )

        # True Peakリミッターを適用
        peak_limit_linear = 10 ** (true_peak_limit / 20)
        current_peak = np.max(np.abs(normalized_audio))

        limiter_applied = False
        # リミッター適用時のラウドネス変化を補正するための係数を追加
        if current_peak > peak_limit_linear:
            limiter_gain = peak_limit_linear / current_peak
            normalized_audio = normalized_audio * limiter_gain
            limiter_applied = True
            
            # リミッター適用後のラウドネスを再測定
            limited_loudness = meter.integrated_loudness(normalized_audio)
            
            # リミッターによるラウドネス低下を補正するための追加ゲイン
            loudness_compensation = target_loudness - limited_loudness
            
            # 補正係数（急激な変化を防ぐため0.8を掛ける）
            compensation_gain = 0.8 * (10 ** (loudness_compensation / 20))
            
            # 補正ゲインを適用（ピーク値を再確認）
            normalized_audio = normalized_audio * compensation_gain
            
            # 再度ピーク値をチェックして制限
            new_peak = np.max(np.abs(normalized_audio))
            if new_peak > peak_limit_linear:
                final_limiter_gain = peak_limit_linear / new_peak
                normalized_audio = normalized_audio * final_limiter_gain

        # 正規化後のラウドネスを測定
        normalized_loudness = meter.integrated_loudness(normalized_audio)

        # 出力ディレクトリが存在しない場合は作成
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 正規化された音声を保存
        # soundfileは(samples, channels)の形式を期待
        sf.write(output_path, normalized_audio, sr)

        return {
            "status": "success",
            "original_loudness": float(original_loudness),
            "normalized_loudness": float(normalized_loudness),
            "loudness_gain": float(normalized_loudness - original_loudness),
            "limiter_applied": limiter_applied,
            "true_peak_limit": true_peak_limit,
            "channels": channels,
            "is_stereo": is_stereo,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def print_usage():
    """
    使い方を表示
    """
    print("使い方: python loudness_normalize.py <入力ファイル> [出力ファイル] [オプション]")
    print("\n※出力ファイルを指定しない場合、outputフォルダを作成し、同名で出力します。")
    print("\nオプション:")
    print(
        "  -t, --target      目標ラウドネス（LUFS）（デフォルト: -14.0 [Spotify標準]）"
    )
    print(
        "  -p, --peak        True Peakリミット（dBTP）（デフォルト: -1.0 [Spotify推奨]）"
    )
    print("  -tol, --tolerance 目標ラウドネスからの許容値（LU）（デフォルト: 0.5）")
    print("  -m, --max-attempts 最大試行回数（デフォルト: 5）")
    print("\n対応ファイル形式:")
    print("  - WAV (.wav)")
    print("  - MP3 (.mp3)")
    print("  - FLAC (.flac)")
    
    print("\nプリセット例:")
    print("  標準 (Spotify): -14.0 LUFS / -1.0 dBTP")
    print("  大音量 (Spotify Premium): -11.0 LUFS / -1.0 dBTP")
    print("  小音量 (Spotify Premium): -19.0 LUFS / -1.0 dBTP")
    print("\n使用例:")
    print("  python loudness_normalize.py input.wav")
    print("  python loudness_normalize.py input.wav output.wav")
    print("  python loudness_normalize.py input.wav -t -16.0")
    print("  python loudness_normalize.py input.wav output.wav -t -14.0 -p -1.0")
    print("  python loudness_normalize.py input.wav -tol 0.3 -m 3")


def _get_option_value(i, option_name, convert_func=str):
    """
    コマンドラインオプションの値を取得する

    Args:
        i: 引数のインデックス
        option_name: オプション名
        convert_func: 変換関数（デフォルト: str）

    Returns:
        tuple: (値, 次のインデックス)
    """
    if i + 1 < len(sys.argv):
        try:
            value = convert_func(sys.argv[i + 1])
            return value, i + 2
        except ValueError:
            print(f"エラー: {option_name} オプションの値が無効です: {sys.argv[i + 1]}")
            sys.exit(1)
    else:
        print(f"エラー: {option_name} オプションには値が必要です")
        sys.exit(1)


def parse_arguments():
    """
    コマンドライン引数を解析

    Returns:
        tuple: (input_file, output_file, target_loudness, true_peak_limit, tolerance, max_attempts)
        注意: max_attemptsは1に固定されます
    """
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    input_file = sys.argv[1]
    
    # 入力ファイルの確認
    if not os.path.isfile(input_file):
        print(f"エラー: 入力ファイルが見つかりません: {input_file}")
        sys.exit(1)
        
    if not check_supported_format(input_file):
        print(f"エラー: 対応していないファイル形式です: {input_file}")
        print("サポートされている形式: wav, mp3, flac")
        sys.exit(1)
    
    # 出力ファイルパスの設定
    if len(sys.argv) >= 3 and not sys.argv[2].startswith('-'):
        # コマンドライン引数で出力ファイルパスが指定されている場合
        output_file = sys.argv[2]
        arg_start_index = 3
    else:
        # 出力ファイルパスが指定されていない場合、デフォルトのパスを生成
        input_basename = os.path.basename(input_file)
        output_dir = os.path.join(os.path.dirname(input_file), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, input_basename)
        arg_start_index = 2

    # デフォルト設定（Spotify標準に準拠）
    target_loudness = -14.0  # Spotifyデフォルト: -14dB LUFS
    true_peak_limit = -1.0  # Spotifyのロッシー形式向け推奨: -1dB
    tolerance = 0.5  # デフォルトの許容値
    max_attempts = 1  # 常に1回だけの処理に変更

    # オプション引数の処理
    i = arg_start_index
    while i < len(sys.argv):
        if sys.argv[i] in ["-t", "--target"]:
            target_loudness, i = _get_option_value(i, "target", float)
        elif sys.argv[i] in ["-p", "--peak"]:
            true_peak_limit, i = _get_option_value(i, "peak", float)
        elif sys.argv[i] in ["-tol", "--tolerance"]:
            tolerance, i = _get_option_value(i, "tolerance", float)
        elif sys.argv[i] in ["-m", "--max-attempts"]:
            max_attempts, i = _get_option_value(i, "max-attempts", int)
        else:
            print(f"エラー: 不明なオプション: {sys.argv[i]}")
            sys.exit(1)

    return (
        input_file,
        output_file,
        target_loudness,
        true_peak_limit,
        tolerance,
        max_attempts,
    )


def print_processing_info(
    input_file, output_file, target_loudness, true_peak_limit, tolerance, max_attempts
):
    """
    処理情報を表示

    Args:
        input_file: 入力ファイルパス
        output_file: 出力ファイルパス
        target_loudness: 目標ラウドネス
        true_peak_limit: True Peakリミット
        tolerance: 許容値（LU）
        max_attempts: 最大試行回数
    """
    print("=" * 60)
    print("ラウドネス正規化")
    print("=" * 60)
    print(f"入力ファイル: {input_file}")
    print(f"出力ファイル: {output_file}")
    print(f"目標ラウドネス: {target_loudness} LUFS")
    print(f"True Peakリミット: {true_peak_limit} dBTP")
    print(f"許容値: ±{tolerance} LU")
    print(f"最大試行回数: {max_attempts}回")
    print("=" * 60)


def print_result(result):
    """
    処理結果を表示

    Args:
        result: normalize_loudness の戻り値

    Returns:
        bool: 成功したかどうか
    """
    if result["status"] == "success":
        print("\n✅ 正規化完了")
        print(f"元のラウドネス: {result['original_loudness']:.2f} LUFS")
        print(f"正規化後のラウドネス: {result['normalized_loudness']:.2f} LUFS")
        print(f"適用ゲイン: {result['loudness_gain']:+.2f} LU")
        if result.get("limiter_applied"):
            print(f"True Peakリミッター: 適用（{result['true_peak_limit']:.1f} dBTP）")
        return True
    else:
        print(f"\n❌ エラー: {result.get('error', '不明なエラー')}")
        return False


def is_within_tolerance(value, target, tolerance):
    """
    値が許容範囲内かどうかをチェック

    Args:
        value: 測定値
        target: 目標値
        tolerance: 許容値

    Returns:
        bool: 許容範囲内であればTrue
    """
    return abs(value - target) <= tolerance


def _process_normalization_attempt(
    input_path, output_path, target_loudness, true_peak_limit, attempt_num
):
    """
    一回の正規化処理を実行

    Args:
        input_path: 入力ファイルパス
        output_path: 出力ファイルパス
        target_loudness: 目標ラウドネス
        true_peak_limit: True Peakリミット
        attempt_num: 試行回数

    Returns:
        dict: 処理結果
    """
    print(f"\n📊 {attempt_num}回目の正規化を実行中...")

    # 前回の結果から調整した目標値を計算
    adjusted_target = target_loudness
    
    # 2回目以降の処理では、前回の結果と目標値の差を考慮して目標値を調整
    if attempt_num > 1:
        # 音声を読み込み、現在のラウドネスを測定
        audio_data, sr = librosa.load(input_path, sr=None, mono=False)
        
        # チャンネル数を確認してpyloudnorm用にフォーマット
        if audio_data.ndim == 1:
            # モノラル音声
            pass
        else:
            # ステレオまたはマルチチャンネル音声
            audio_data = audio_data.T
            
        # 現在のラウドネスを測定
        meter = pyln.Meter(sr, block_size=0.400)
        current_loudness = meter.integrated_loudness(audio_data)
        
        # 目標値との差を計算
        loudness_diff = target_loudness - current_loudness
        
        # 差に基づいて調整係数を適用（収束を早めるために差の1.5倍を適用）
        adjustment_factor = 1.5 * loudness_diff
        
        # 現在値に調整を加えた値を新たな目標とする
        adjusted_target = target_loudness + adjustment_factor
        print(f"目標値調整: 元の目標 {target_loudness:.2f} LUFS → 調整後 {adjusted_target:.2f} LUFS")

    # 正規化を実行
    result = normalize_loudness(
        input_path=input_path,
        output_path=output_path,
        target_loudness=adjusted_target,  # 調整された目標値を使用
        true_peak_limit=true_peak_limit,
    )

    # 結果を表示
    if result["status"] != "success":
        print(f"\n❌ エラー: {result.get('error', '不明なエラー')}")
        sys.exit(1)

    print(f"元のラウドネス: {result['original_loudness']:.2f} LUFS")
    print(f"正規化後のラウドネス: {result['normalized_loudness']:.2f} LUFS")
    print(f"適用ゲイン: {result['loudness_gain']:+.2f} LU")
    print(f"目標値との差: {result['normalized_loudness'] - target_loudness:+.2f} LU")

    if result.get("limiter_applied"):
        print(f"True Peakリミッター: 適用（{result['true_peak_limit']:.1f} dBTP）")

    return result


def _display_final_result(
    success, result, output_file, target_loudness, tolerance, max_attempts
):
    """
    最終結果を表示

    Args:
        success: 成功したかどうか
        result: 最後の正規化結果
        output_file: 出力ファイルパス
        target_loudness: 目標ラウドネス
        tolerance: 許容値
        max_attempts: 最大試行回数
    """
    if success:
        print("\n🎉 正規化が完了しました！")
    else:
        print(
            f"\n❌ 正規化に失敗しました。"
            f"目標値 {target_loudness} LUFS ± {tolerance} LU に収まりませんでした。"
        )
        print(f"最終的なラウドネス: {result['normalized_loudness']:.2f} LUFS")
        print(f"目標との差: {result['normalized_loudness'] - target_loudness:+.2f} LU")
    
    # 成功・失敗に関わらず、ファイルは出力されたことを表示
    print(f"\n出力ファイル: {output_file}")


def main():
    """
    メイン処理
    """
    # コマンドライン引数を解析
    (
        input_file,
        output_file,
        target_loudness,
        true_peak_limit,
        tolerance,
        max_attempts,
    ) = parse_arguments()

    # 入力ファイルの存在確認
    if not os.path.exists(input_file):
        print(f"エラー: 入力ファイルが見つかりません: {input_file}")
        sys.exit(1)

    # 処理情報を表示
    print_processing_info(
        input_file,
        output_file,
        target_loudness,
        true_peak_limit,
        tolerance,
        max_attempts,
    )

    try:
        # 常に1回だけの処理に変更
        print("\n📊 正規化を実行中...")
        result = normalize_loudness(
            input_path=input_file,
            output_path=output_file,
            target_loudness=target_loudness,
            true_peak_limit=true_peak_limit,
        )

        # 結果を表示
        if result["status"] != "success":
            print(f"\n❌ エラー: {result.get('error', '不明なエラー')}")
            sys.exit(1)

        success = is_within_tolerance(
            result["normalized_loudness"], target_loudness, tolerance
        )

        print(f"元のラウドネス: {result['original_loudness']:.2f} LUFS")
        print(f"正規化後のラウドネス: {result['normalized_loudness']:.2f} LUFS")
        print(f"適用ゲイン: {result['loudness_gain']:+.2f} LU")
        print(f"目標値との差: {result['normalized_loudness'] - target_loudness:+.2f} LU")

        if result.get("limiter_applied"):
            print(f"True Peakリミッター: 適用（{result['true_peak_limit']:.1f} dBTP）")

        # 結果をまとめて表示
        _display_final_result(
            success, result, output_file, target_loudness, tolerance, max_attempts
        )

    except Exception as e:
        print(f"\n❌ エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
