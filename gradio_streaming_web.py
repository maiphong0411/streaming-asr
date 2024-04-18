import json
import gradio as gr
import numpy as np
import torch
import wenet

torch.manual_seed(777)  # for lint

# wenet.set_log_level(2)
path_model = '/home/ubuntu/Documents/ASR/model'
model = wenet.load_model(model_dir=path_model)

attn_cache = torch.zeros((0, 0, 0, 0))
cnn_cache = torch.zeros((0, 0, 0, 0))
offset = 0
result_list = []
def recognition(audio):
    global attn_cache
    global cnn_cache
    global offset
    # print(attn_cache)
    sr, y = audio
    assert sr in [48000, 16000]
    if sr == 48000:  # Optional resample to 16000
        y = (y / max(np.max(y), 1) * 32767)[::3].astype("int16")
    print(type(y))
    print(y.shape)
    y = y.reshape(1, -1)
    print(y.shape)
    ans, attn_cache, cnn_cache, offset = model.decode(y, att_cache=attn_cache, cnn_cache=cnn_cache, offset=offset)
    # if ans['text'] == "":
    #     return ans
    # ans = json.loads(ans)
    # text = ans["nbest"][0]["sentence"]
    print(">>>>>>>>>> ",ans)
    result_list.append(ans['text'])
    return " ".join(result_list)

print("\n===> Loading the ASR model ...")
print("===> Warming up by 10 randomly-generated audios ... Please wait ...\n")
for i in range(10):
    audio_len = np.random.randint(16000 * 3, 16000 * 10)  # 3~10s
    audio = np.random.randint(-32768, 32768, size=(1,audio_len), dtype=np.int16)
    ans, _, _, _ = model.decode(audio)
    print("Processed the {}-th audio.".format(i + 1))

with gr.Blocks() as demo:
    gr.Markdown("Streaming Speech Recognition in WeNet | 基于 WeNet 的流式语音识别")
    with gr.Row():
        inputs = gr.Microphone(streaming=True)
        outputs = gr.Textbox(label="Output:")
    inputs.stream(fn=recognition, inputs=inputs, outputs=outputs,
                  show_progress="hidden")
demo.launch()