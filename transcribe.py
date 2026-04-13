import whisper
import time
from zhipuai import ZhipuAI
from jiwer import wer
import noisereduce as nr
import soundfile as sf
import tempfile
import os

def _load_env(path: str = ".env") -> None:
    """极简 .env 加载：KEY=VALUE 注入 os.environ（不覆盖已存在的环境变量）。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_env()
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "")
if not ZHIPU_API_KEY:
    raise RuntimeError(
        "未配置 ZHIPU_API_KEY：请在项目根目录 .env 写入 ZHIPU_API_KEY=xxx（参考 .env.example）"
    )
client = ZhipuAI(api_key=ZHIPU_API_KEY)


def denoise_audio(audio_path: str) -> str:
    import librosa
    data, sample_rate = librosa.load(audio_path, sr=None, mono=True)

    # 用能量最低的0.5秒作为噪声样本，而不是固定取开头
    frame_length = int(sample_rate * 0.5)
    energies = [
        np.sum(data[i:i + frame_length] ** 2)
        for i in range(0, len(data) - frame_length, frame_length // 2)
    ]
    quietest_idx = np.argmin(energies)
    start = quietest_idx * (frame_length // 2)
    noise_sample = data[start:start + frame_length]

    reduced = nr.reduce_noise(
        y=data,
        sr=sample_rate,
        y_noise=noise_sample,
        prop_decrease=0.8
    )

    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, reduced, sample_rate)
    return tmpdef denoise_audio(audio_path: str) -> str:
    import librosa
    data, sample_rate = librosa.load(audio_path, sr=None, mono=True)

    # 用能量最低的0.5秒作为噪声样本，而不是固定取开头
    frame_length = int(sample_rate * 0.5)
    energies = [
        np.sum(data[i:i+frame_length]**2)
        for i in range(0, len(data)-frame_length, frame_length//2)
    ]
    quietest_idx = np.argmin(energies)
    start = quietest_idx * (frame_length // 2)
    noise_sample = data[start:start + frame_length]

    reduced = nr.reduce_noise(
        y=data,
        sr=sample_rate,
        y_noise=noise_sample,
        prop_decrease=0.8
    )

    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, reduced, sample_rate)
    return tmp

def transcribe_audio(audio_path: str, language: str = "zh", denoise: bool = False) -> tuple[str, str]:
    if denoise:
        print("降噪处理中...")
        audio_path = denoise_audio(audio_path)

    print("加载 Whisper 模型...")
    model = whisper.load_model("base")
    print("转录中...")
    result = model.transcribe(audio_path, language=language)

    if denoise:
        os.remove(audio_path)

    raw_text = result["text"]
    segments = result["segments"]
    timestamped_lines = []
    for seg in segments:
        start = int(seg["start"])
        end = int(seg["end"])
        text = seg["text"].strip()
        mm_ss_start = f"{start//60:02d}:{start%60:02d}"
        mm_ss_end = f"{end//60:02d}:{end%60:02d}"
        timestamped_lines.append(f"[{mm_ss_start} → {mm_ss_end}] {text}")

    timestamped_text = "\n".join(timestamped_lines)
    return raw_text, timestamped_text

def polish_transcript(transcript: str) -> str:
    """用 LLM 对转录粗稿纠错并生成摘要"""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[{
                    "role": "user",
                    "content": f"""以下是一段语音转录的粗稿，可能有错别字和识别错误。
1. 根据上下文纠正明显的错误
2. 根据每句话的内容和语气，在适当位置插入 emoji
3. 用3-5句话概括主要内容

加 emoji 的原则：
- 不要每句话都加，只在情绪或内容明显的地方加
- 语气轻松的地方用 😄🎉，严肃的地方不加
- 提到具体事物时可以加对应 emoji，比如提到音乐加 🎵

转录原文：
{transcript}

请按以下格式输出：
纠错+emoji版本：
[内容]

摘要：
[内容]"""
                }]
            )
            return response.choices[0].message.content
        except Exception as e:
            if "1302" in str(e):
                print(f"触发限速，等待60秒后重试（第{attempt + 1}次）...")
                time.sleep(60)
            elif "1301" in str(e) or "content" in str(e).lower():
                return "⚠️ 内容审核未通过，以下为原始转录：\n\n" + transcript
            else:
                raise e
    return "重试失败，请稍后再试"


def process_audio(audio_path: str, denoise: bool = False) -> tuple[str, str, str]:
    raw_text, timestamped_text = transcribe_audio(audio_path, denoise=denoise)
    print("转录完成，开始 LLM 后处理...")
    polished = polish_transcript(raw_text)
    return raw_text, timestamped_text, polished


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    计算字错误率 CER，适合中文
    把每个字符当成一个 token 来比较
    """
    # 去掉标点和空格，只保留汉字
    import re
    def clean(text):
        return re.sub(r'[^\u4e00-\u9fff]', '', text)

    ref = list(clean(reference))
    hyp = list(clean(hypothesis))

    if len(ref) == 0:
        return 0.0

    # 用编辑距离计算
    import numpy as np
    d = np.zeros((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)

    cer = d[len(ref)][len(hyp)] / len(ref)
    return round(cer * 100, 2)


