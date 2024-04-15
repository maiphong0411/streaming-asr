import json
import gradio as gr
import numpy as np
import torch
import wenet

torch.manual_seed(777)  # for lint

# wenet.set_log_level(2)
model = wenet.load_model(model_dir='/home/ubuntu/Documents/ASR/model')

def recognition(audio):
    sr, y = audio
    assert sr in [48000, 16000]
    if sr == 48000:  # Optional resample to 16000
        y = (y / max(np.max(y), 1) * 32767)[::3].astype("int16")
    print(type(y))
    print(y.shape)
    y = y.reshape(1, -1)
    print(y.shape)
    ans = model.decode(y)
    # if ans['text'] == "":
    #     return ans
    # ans = json.loads(ans)
    # text = ans["nbest"][0]["sentence"]
    print(">>>>>>>>>> ",ans)
    return ans['text']

print("\n===> Loading the ASR model ...")
print("===> Warming up by 100 randomly-generated audios ... Please wait ...\n")
for i in range(10):
    audio_len = np.random.randint(16000 * 3, 16000 * 10)  # 3~10s
    audio = np.random.randint(-32768, 32768, size=(1,audio_len), dtype=np.int16)
    ans = model.decode(audio)
    print("Processed the {}-th audio.".format(i + 1))

with gr.Blocks() as demo:
    gr.Markdown("Streaming Speech Recognition in WeNet | 基于 WeNet 的流式语音识别")
    with gr.Row():
        inputs = gr.Microphone(streaming=True)
        outputs = gr.Textbox(label="Output:")
    inputs.stream(fn=recognition, inputs=inputs, outputs=outputs,
                  show_progress="hidden")
demo.launch()