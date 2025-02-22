# -*- coding:utf-8 -*-
import json
import os

import yaml
from scipy.io.wavfile import write
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils
import commons
import sys
import re
from torch import no_grad, LongTensor
import logging



logging.getLogger('numba').setLevel(logging.WARNING)


def ex_print(text, escape=False):
    if escape:
        print(text.encode('unicode_escape').decode())
    else:
        print(text)


def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def ask_if_continue():
    while True:
        answer = input('Continue? (y/n): ')
        if answer == 'y':
            break
        elif answer == 'n':
            sys.exit(0)


def print_speakers(speakers, escape=False):
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        ex_print(str(id) + '\t' + name, escape)


def get_speaker_id(message):
    '''speaker_id = input(message)
    try:
        speaker_id = int(speaker_id)
    except:
        print(str(speaker_id) + ' is not a valid ID!')
        sys.exit(1)
    return speaker_id'''
    return 0


def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text


def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text


MODEL_CACHE = {}


def load_model(model_path, config_path):
    """加载并缓存模型"""
    if (model_path, config_path) in MODEL_CACHE:
        print("模型已加载，复用")
        return MODEL_CACHE[(model_path, config_path)]
    print("模型初次加载，loading...")
    hps_ms = utils.get_hparams_from_file(config_path)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
    emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotion_embedding=emotion_embedding,
        **hps_ms.model
    )
    _ = net_g_ms.eval()
    utils.load_checkpoint(model_path, net_g_ms)

    MODEL_CACHE[(model_path, config_path)] = (net_g_ms, hps_ms)
    return net_g_ms, hps_ms


def vG(text, out_path, speaker_id: int, modelSelect: list):
    """文本转语音，避免重复加载模型"""
    model_path, config_path = modelSelect

    # 复用已加载的模型
    net_g_ms, hps_ms = load_model(model_path, config_path)

    length_scale, text = get_label_value(text, 'LENGTH', data["tts_config"]["speed"], 'length scale')
    noise_scale, text = get_label_value(text, 'NOISE', data["tts_config"]["noise"], 'noise scale')
    noise_scale_w, text = get_label_value(text, 'NOISEW', 0.7, 'deviation of noise')
    cleaned, text = get_label(text, 'CLEANED')

    stn_tst = get_text(text, hps_ms, cleaned=cleaned)


    with no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)])
        sid = LongTensor([speaker_id])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                               noise_scale_w=noise_scale_w, length_scale=length_scale)[0][
            0, 0].data.cpu().float().numpy()

    write(out_path, hps_ms.data.sampling_rate, audio)  # 将生成的语音文件写入本地
    return out_path
all_speakers={}
def load_add_speakers():
    """
    加载所有speaker并制作索引
    :return:
    """
    for i in os.listdir("voiceModel"):
        if os.path.isdir(os.path.join("voiceModel", i)):
            config_file = os.path.join("voiceModel", i, "config.json")
            for j in os.listdir(os.path.join("voiceModel", i)):
                if j.endswith(".pth"):
                    model_file = os.path.join("voiceModel", i, j)
            with open(config_file, "r", encoding="utf-8") as file:
                data = json.load(file)  # 解析 JSON 数据
            speakers = data["speakers"]
            index = 0
            for speaker in speakers:
                all_speakers[speaker] = {"index": index, "model":[model_file, config_file]}
                index += 1
    return all_speakers
load_add_speakers()

"""
读取配置文件
"""
with open("config.yaml", "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)  # 解析 YAML 数据

print(f"当前可用speaker:{list(all_speakers.keys())}")

"""
flask_api，供给外部请求
"""
from flask import Flask, send_file, request, abort, jsonify
import os
import uuid

app = Flask(__name__)

@app.route("/get_speakers", methods=["GET"])
def get_speakers():
    return jsonify(list(all_speakers.keys()))

@app.route("/get_audio", methods=["GET"])
def get_audio():
    text = request.args.get("text")
    speaker=request.args.get("speaker",data["speaker"])
    speaker_index=all_speakers[speaker]["index"]
    model=all_speakers[speaker]["model"]

    if not text:
        return abort(400, "Missing text parameter")

    base_dir = os.getcwd()  # 获取当前工作目录
    audio_dir = os.path.join(base_dir, "output")
    os.makedirs(audio_dir, exist_ok=True)  # 确保输出目录存在

    filename = f"{uuid.uuid4()}.mp3"
    file_path = os.path.join(audio_dir, filename)

    vG(text, file_path, int(speaker_index), model)  # 调用声码器生成音频文件

    return send_file(file_path, as_attachment=True, mimetype="audio/mpeg")


app.run(host="0.0.0.0", port=data["server"]["port"], debug=True)


