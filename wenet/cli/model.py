# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from wenet.cli.hub import Hub
from wenet.utils.ctc_utils import (force_align, gen_ctc_peak_time,
                                   gen_timestamps_from_peak)
from wenet.utils.file_utils import read_symbol_table
from wenet.transformer.search import (attention_rescoring,
                                      ctc_prefix_beam_search, DecodeResult)
from wenet.utils.context_graph import ContextGraph


class Model:

    def __init__(self,
                 model_dir: str,
                 gpu: int = -1,
                 beam: int = 5,
                 context_path: str = None,
                 context_score: float = 6.0,
                 resample_rate: int = 16000,
                 frame_length: int = 25,
                 frame_shift: int = 10):
        model_path = os.path.join(model_dir, 'final.zip')
        units_path = os.path.join(model_dir, 'units.txt')
        self.model = torch.jit.load(model_path)
        self.resample_rate = resample_rate
        self.model.eval()
        if gpu >= 0:
            device = 'cuda:{}'.format(gpu)
        else:
            device = 'cpu'
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.symbol_table = read_symbol_table(units_path)
        self.char_dict = {v: k for k, v in self.symbol_table.items()}
        self.beam = beam
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.window_size = int(self.resample_rate * self.frame_length * 0.001)
        self.window_stride = int(self.resample_rate * self.frame_shift * 0.001)
        if context_path is not None:
            self.context_graph = ContextGraph(context_path,
                                              self.symbol_table,
                                              context_score=context_score)
        else:
            self.context_graph = None

    def compute_feats(self, audio_file: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_file, normalize=False)
        waveform = waveform.to(torch.float)
        if sample_rate != self.resample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.resample_rate)(waveform)
        waveform = waveform.to(self.device)
        feats = kaldi.fbank(waveform,
                            num_mel_bins=80,
                            frame_length=self.frame_length,
                            frame_shift=self.frame_shift,
                            energy_floor=0.0,
                            sample_frequency=self.resample_rate)
        feats = feats.unsqueeze(0)
        return feats
    
    def convert_audio_to_waveform(self, audio_file):
        waveform, sample_rate = torchaudio.load(audio_file, normalize=False)
        waveform = waveform.to(torch.float)
        if sample_rate != self.resample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.resample_rate)(waveform)
        return waveform
    
    @torch.no_grad()
    def _streaming_chunk_by_chunk(self, audio_file, att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)), chunk_size: int = 8, decoding_chunk_size: int = 35):
        list_feats = []
        result = {'text': []}
        ctc_probs = None
        encoder_cache = []
        ctc_probs_cache = []
        encoder_lens = None
        count = 0
        offset = 0
        tmp = ""
        subsampling_rate = self.model.subsampling_rate() # 4
        assert subsampling_rate == 4
        right_context = self.model.right_context() # 6
        assert right_context == 6
        stride = subsampling_rate * chunk_size
        context = right_context + 1 # add current frame
        decoding_window = (chunk_size - 1) * subsampling_rate + context
#         num_frames = xs.size(1)
        
        # load audio
        waveform = self.convert_audio_to_waveform(audio_file)
        num_steps = int((waveform.size(1) - self.window_size) // (self.window_stride)) + 1
        forward_count = 0
        start_chunk = 0
        end_chunk = 0
        
        print(f"num of frames {num_steps}")
        for i in range(0, num_steps):
            cur = i * self.window_stride
            end = cur + self.window_size
            wav = waveform[:, cur:end]
            feat = kaldi.fbank(wav,
                            num_mel_bins=80,
                            frame_length=self.frame_length,
                            frame_shift=self.frame_shift,
                            energy_floor=0.0,
                            sample_frequency=self.resample_rate)
            feat = feat.unsqueeze(0) # batch,frame ,mel-dim
            list_feats.append(feat.to(self.device))
            
            # decode
            if (end_chunk < num_steps) and (start_chunk < num_steps - context + 1):
                end_chunk = min(start_chunk + decoding_window, num_steps - 1)
                if end_chunk > i: # current steps
                    continue
                xs = torch.cat([x for x in list_feats[start_chunk:end_chunk]], dim=1) # batch, decoding_chunk_size, mel-dim
                if xs.size(1) < 8:
                    pad_tensor = torch.zeros(1, 8, 80).to(self.device)
                    xs = torch.cat([xs, pad_tensor], dim=1).to(self.device)
#                 print(start_chunk, end_chunk)
                start_chunk += stride
                encoder_out, att_cache, cnn_cache = self.model.forward_encoder_chunk(xs, offset, -1, att_cache, cnn_cache)
                encoder_cache.append(encoder_out)
                offset += encoder_out.size(1)
                ctc_probs = self.model.ctc_activation(encoder_out) # batch, frame, vocab_size
                ctc_probs_cache.append(ctc_probs)

                indices = torch.argmax(ctc_probs, dim=2)
                indices = indices.squeeze(0)
                index = [x.item() for x in indices if x != 0]

                s = ''.join([self.char_dict[x] for x in index])
                result['text'].append(s)
                tmp += s
                progress_output(tmp)
                
        
        encoder_outs = torch.cat(encoder_cache, dim=1)
        
        ctc_probs = self.model.ctc_activation(encoder_outs)
        encoder_lens = torch.tensor([encoder_outs.size(1)],
                                    dtype=torch.long,
                                    device=encoder_out.device)
        ctc_prefix_results = ctc_prefix_beam_search(
                ctc_probs,
                encoder_lens,
                self.beam,
                context_graph=self.context_graph)
        
        rescoring_results = attention_rescoring(self.model, ctc_prefix_results,
                                                encoder_outs, encoder_lens, 0.3,
                                                0.5)
        res = rescoring_results[0]
        output = ''.join([self.char_dict[x] for x in res.tokens])
        output = output.replace('▁', ' ')
        print()
        print("after rescoring: ",output)
        result['output'] = output
        return result
        
    @torch.no_grad()
    def _decode_with_streamming(self, audio_file: str,
                                simulate_streaming: bool = True,
                                decoding_chunk_size: int = 16,
                                num_decoding_left_chunks: int = -1,
                                tokens_info: bool = False,
                                label: str = None) -> dict:
        speech = self.compute_feats(audio_file)
        speech_lengths = torch.Tensor(speech.size(0))
        encoder_out, encoder_mask = self.model.forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)
        print(f"encoder out size {encoder_out.size()}")
        encoder_lens = encoder_mask.squeeze(1).sum(1)
        
        ctc_probs = self.model.ctc_activation(encoder_out)
        
        ctc_prefix_results = ctc_prefix_beam_search(
                ctc_probs,
                encoder_lens,
                self.beam,
                context_graph=self.context_graph)
        
        rescoring_results = attention_rescoring(self.model, ctc_prefix_results,
                                                encoder_out, encoder_lens, 0.3,
                                                0.5)
        res = rescoring_results[0]
        result = {}
        result['text'] = ''.join([self.char_dict[x] for x in res.tokens])
        result['confidence'] = res.confidence

        if tokens_info:
            frame_rate = self.model.subsampling_rate(
            ) * 0.01  # 0.01 seconds per frame
            max_duration = encoder_out.size(1) * frame_rate
            times = gen_timestamps_from_peak(res.times, max_duration,
                                             frame_rate, 1.0)
            tokens_info = []
            for i, x in enumerate(res.tokens):
                tokens_info.append({
                    'token': self.char_dict[x],
                    'start': times[i][0],
                    'end': times[i][1],
                    'confidence': res.tokens_confidence[i]
                })
            result['tokens'] = tokens_info
        return result
        
                
    @torch.no_grad()
    def _decode(self,
                audio_file: str,
                tokens_info: bool = False,
                label: str = None) -> dict:
        feats = self.compute_feats(audio_file)
        import time
        start = time.time()
        print(f"feats size {feats.size()}")
        encoder_out, _, _ = self.model.forward_encoder_chunk(feats, 0, -1) # xs, offset, cache_size 
        end = time.time()
        print(f"duration encode {end - start}")
        encoder_lens = torch.tensor([encoder_out.size(1)],
                                    dtype=torch.long,
                                    device=encoder_out.device)

        ctc_probs = self.model.ctc_activation(encoder_out)

        if label is None:
            ctc_prefix_results = ctc_prefix_beam_search(
                ctc_probs,
                encoder_lens,
                self.beam,
                context_graph=self.context_graph)
        else:  # force align mode, construct ctc prefix result from alignment
            label_t = self.tokenize(label)
            alignment = force_align(ctc_probs.squeeze(0),
                                    torch.tensor(label_t, dtype=torch.long))
            peaks = gen_ctc_peak_time(alignment)
            ctc_prefix_results = [
                DecodeResult(tokens=label_t,
                             score=0.0,
                             times=peaks,
                             nbest=[label_t],
                             nbest_scores=[0.0],
                             nbest_times=[peaks])
            ]
        rescoring_results = attention_rescoring(self.model, ctc_prefix_results,
                                                encoder_out, encoder_lens, 0.3,
                                                0.5)
        res = rescoring_results[0]
        result = {}
        result['text'] = ''.join([self.char_dict[x] for x in res.tokens])
        result['confidence'] = res.confidence

        if tokens_info:
            frame_rate = self.model.subsampling_rate(
            ) * 0.01  # 0.01 seconds per frame
            max_duration = encoder_out.size(1) * frame_rate
            times = gen_timestamps_from_peak(res.times, max_duration,
                                             frame_rate, 1.0)
            tokens_info = []
            for i, x in enumerate(res.tokens):
                tokens_info.append({
                    'token': self.char_dict[x],
                    'start': times[i][0],
                    'end': times[i][1],
                    'confidence': res.tokens_confidence[i]
                })
            result['tokens'] = tokens_info
        return result
    
    @torch.no_grad()
    def decode(self, waveform, sample_rate=16000, label=None):
        if type(waveform) == np.ndarray:
            waveform = torch.from_numpy(waveform)
        waveform = waveform.to(torch.float)
        # waveform = torch.from_numpy(waveform).to(torch.float)
        # waveform = waveform.unsqueeze(dim=0)
        # print(waveform.size())
        if sample_rate != self.resample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.resample_rate)(waveform)
        waveform = waveform.to(self.device)
        print(waveform.size())
        feats = kaldi.fbank(waveform,
                            num_mel_bins=80,
                            frame_length=self.frame_length,
                            frame_shift=self.frame_shift,
                            energy_floor=0.0,
                            sample_frequency=self.resample_rate)
        feats = feats.unsqueeze(0)
        self.model.eval()
        encoder_out, _, _ = self.model.forward_encoder_chunk(feats, 0, -1) # xs, offset, cache_size 
        encoder_lens = torch.tensor([encoder_out.size(1)],
                                    dtype=torch.long,
                                    device=encoder_out.device)

        ctc_probs = self.model.ctc_activation(encoder_out)

        if label is None:
            ctc_prefix_results = ctc_prefix_beam_search(
                ctc_probs,
                encoder_lens,
                self.beam,
                context_graph=self.context_graph)
        else:  # force align mode, construct ctc prefix result from alignment
            label_t = self.tokenize(label)
            alignment = force_align(ctc_probs.squeeze(0),
                                    torch.tensor(label_t, dtype=torch.long))
            peaks = gen_ctc_peak_time(alignment)
            ctc_prefix_results = [
                DecodeResult(tokens=label_t,
                             score=0.0,
                             times=peaks,
                             nbest=[label_t],
                             nbest_scores=[0.0],
                             nbest_times=[peaks])
            ]
        rescoring_results = attention_rescoring(self.model, ctc_prefix_results,
                                                encoder_out, encoder_lens, 0.3,
                                                0.5)
        res = rescoring_results[0]
        result = {}
        result['text'] = ''.join([self.char_dict[x] for x in res.tokens])
        result['confidence'] = res.confidence
        
        return result
        
        
        
    def transcribe(self, 
                   audio_file: str, 
                   chunk_size: int = 8,
                   tokens_info: bool = False, 
                   simulate_stream: bool = False) -> dict:
        if simulate_stream:
            return self._streaming_chunk_by_chunk(audio_file, chunk_size=chunk_size)
        else:
            return self._decode(audio_file, tokens_info)

    def tokenize(self, label: str):
        # TODO(Binbin Zhang): Support BPE
        tokens = []
        for c in label:
            if c == ' ':
                c = "▁"
            tokens.append(c)
        token_list = []
        for c in tokens:
            if c in self.symbol_table:
                token_list.append(self.symbol_table[c])
            elif '<unk>' in self.symbol_table:
                token_list.append(self.symbol_table['<unk>'])
        return token_list

    def align(self, audio_file: str, label: str) -> dict:
        return self._decode(audio_file, True, label)


def load_model(language: str = None,
               model_dir: str = None,
               gpu: int = -1,
               beam: int = 5,
               context_path: str = None,
               context_score: float = 6.0) -> Model:
    if model_dir is None:
        model_dir = Hub.get_model_by_lang(language)
    return Model(model_dir, gpu, beam, context_path, context_score)

prev_lines = 0
def progress_output(text):
    text = text.replace('▁', ' ')
    global prev_lines
    lines=['']
    for i in text:
        if len(lines[-1]) > 100:
            lines.append('')
        lines[-1] += i
    for i,line in enumerate(lines):
        if i == prev_lines:
            sys.stderr.write('\n\n\r')
        else:
            sys.stderr.write('\r\033[B\033[K')
        sys.stderr.write(line)

    prev_lines = len(lines)
    sys.stderr.flush()
    
def cut_audio_fixed_duration(audio_file, duration=1.0, resample_rate=16000):
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_file, normalize=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    waveform = waveform.to(torch.float)
    if sample_rate != resample_rate:
        waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    waveform = waveform.to(device)
    
    # Calculate the number of parts based on the desired duration
    num_parts = int(waveform.size(1) / (duration * sample_rate))
    
    if num_parts < 1:
        return [waveform]
    # Cut audio into fixed-duration parts
    audio_parts = []
    for i in range(num_parts):
        start_time = int(i * duration * sample_rate)
        end_time = int((i + 1) * duration * sample_rate)

        # Ensure end_time is within the audio length
        end_time = min(end_time, waveform.size(1))

        # Extract part
        part = waveform[:, start_time:end_time]

        audio_parts.append(part)

    # Check if there is remaining audio after the last chunk
    remaining_audio = waveform[:, int(num_parts*duration*sample_rate):]
    if remaining_audio.size(1) > 0:
        # Concatenate the remaining audio with the last chunk
        audio_parts[-1] = torch.cat([audio_parts[-1], remaining_audio], dim=1)

    return audio_parts
