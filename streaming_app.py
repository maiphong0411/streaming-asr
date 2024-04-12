# streamlit_audio_recorder by stefanrmmr (rs. analytics) - version January 2023

import streamlit as st
from st_audiorec import st_audiorec
# from pydub import AudioSegment
# import pydub
import numpy as np
import wenet
import soundfile as sf
import io
import librosa
import torchaudio
import torch
# DESIGN implement changes to the standard streamlit UI/UX
# --> optional, not relevant for the functionality of the component!
st.set_page_config(page_title="streamlit_audio_recorder")
# Design move app further up and remove top padding
st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''',
            unsafe_allow_html=True)
# Design change st.Audio to fixed height of 45 pixels
st.markdown('''<style>.stAudio {height: 45px;}</style>''',
            unsafe_allow_html=True)
# Design change hyperlink href link color
st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''',
            unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''',
            unsafe_allow_html=True)  # lightmode


def audiorec_demo_app():
    model = wenet.load_model(model_dir='/home/ubuntu/Documents/ASR/model')
    print("\n===> Loading the ASR model ...")
    print("===> Warming up by 100 randomly-generated audios ... Please wait ...\n")
    for i in range(10):
        audio_len = torch.randint(16000 * 3, 16000 * 10, (1,))  # 3~10s
        print(audio_len)
        audio = torch.randint(-32768, 32768, size=(1,audio_len.item()), dtype=torch.int16)
    #     print(audio.shape)
        audio = audio.to(torch.float)
        ans = model.decode(audio)
        # print(ans)
        print("Processed the {}-th audio.".format(i + 1))

    # TITLE and Creator information
    st.title('streamlit audio recorder')
    st.markdown('Implemented by '
        '[Stefan Rummer](https://www.linkedin.com/in/stefanrmmr/) - '
        'view project source code on '
                
        '[GitHub](https://github.com/stefanrmmr/streamlit-audio-recorder)')
    st.write('\n\n')

    # TUTORIAL: How to use STREAMLIT AUDIO RECORDER?
    # by calling this function an instance of the audio recorder is created
    # once a recording is completed, audio data will be saved to wav_audio_data

    wav_audio_data = st_audiorec() # bytes string
    print(type(wav_audio_data))

    audio_data, sample_rate = torchaudio.load(io.BytesIO(wav_audio_data), normalize=False)
    waveform = torch.sum(audio_data, dim=0)
    print(f"sample rate {sample_rate}")

    waveform = torch.unsqueeze(waveform, dim=0)
    print(waveform.size())
    result = model.decode(waveform, sample_rate=sample_rate)
    print(result)
    # add some spacing and informative messages
    col_info, col_space = st.columns([0.57, 0.43])
    with col_info:
        st.write('\n')  # add vertical spacer
        st.write('\n')  # add vertical spacer
        if result['text'] is not None:
            st.write(f'Text received: {result["text"]}')
            result['text'] = ""
        # st.write('The .wav audio data, as received in the backend Python code,'
        #          ' will be displayed below this message as soon as it has'
        #          ' been processed. [This informative message is not part of'
        #          ' the audio recorder and can be removed easily] 🎈')

    if wav_audio_data is not None:
        # display audio data as received on the Python side
        col_playback, col_space = st.columns([0.58,0.42])
        with col_playback:
            st.audio(wav_audio_data, format='audio/wav')


if __name__ == '__main__':
    # call main function
    audiorec_demo_app()
