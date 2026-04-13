import gradio as gr
from transcribe import process_audio, calculate_cer

def run(audio_path, denoise, reference_text):
    if audio_path is None:
        return "请上传音频文件", "", "", ""
    raw_text, timestamped_text, polished = process_audio(audio_path, denoise=denoise)

    cer_result = ""
    if reference_text.strip():
        error_rate = calculate_cer(reference_text, raw_text)
        cer_result = f"CER（字错误率）：{error_rate}%\n\n说明：越低越好，0% 为完美，人工转录约 5%"

    return raw_text, timestamped_text, polished, cer_result

demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Audio(label="上传音频", type="filepath", sources=["upload", "microphone"]),
        gr.Checkbox(label="开启降噪（背景噪音较多时勾选）", value=False),
        gr.Textbox(label="参考文本（可选，用于计算准确率）",
                   placeholder="粘贴正确的文字内容，留空则不计算 CER", lines=3),
    ],
    outputs=[
        gr.Textbox(label="Whisper 转录原文", lines=6),
        gr.Textbox(label="带时间戳的转录", lines=8),
        gr.Textbox(label="LLM 纠错 + 摘要", lines=8),
        gr.Textbox(label="准确率评估", lines=3),
    ],
    title="🎙️ 语音转录 + 智能摘要",
    description="上传一段音频，自动转录并生成带时间戳的文字和摘要",
)

if __name__ == "__main__":
    demo.launch()