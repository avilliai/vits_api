import logging
from json import loads
from torch import load, FloatTensor
from numpy import float32
import librosa


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()


def load_checkpoint(checkpoint_path, model):
  checkpoint_dict = load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      logging.info("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)
  logging.info("Loaded checkpoint '{}' (iteration {})" .format(
    checkpoint_path, iteration))
  return


def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = loads(data)

  hparams = HParams(**config)
  return hparams


def load_audio_to_torch(full_path, target_sampling_rate):
  audio, sampling_rate = librosa.load(full_path, sr=target_sampling_rate, mono=True)
  return FloatTensor(audio.astype(float32))
import httpx
import requests
from langdetect import detect

def translate(text, mode="ZH_CN2JA"):
    try:
        URL = f"https://api.pearktrue.cn/api/translate/?text={text}&type={mode}"
        r=requests.get(url=URL,timeout=10)
        #print(r.json()["data"]["translate"])
        return r.json()["data"]["translate"]
    except:
        pass
        if mode != "ZH_CN2JA":
            return text
    try:
        url = f"https://findmyip.net/api/translate.php?text={text}&target_lang=ja"
        r = requests.get(url=url, timeout=10)
        return r.json()["data"]["translate_result"]
    except:
        pass
    try:
        url = f"https://translate.appworlds.cn?text={text}&from=zh-CN&to=ja"
        r = requests.get(url=url, timeout=10, verify=False)
        return r.json()["data"]
    except:
        pass
    return text
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # 让 langdetect 结果更稳定

def detect_language(text: str):
    """使用 langdetect 进行语言检测，返回 'zh' 或 'ja'，否则返回 'unknown'。"""
    try:
        lang_code = detect(text)
        if lang_code.startswith("zh"):
            return "zh"
        elif lang_code.startswith("ja"):
            return "ja"
    except Exception:
        return "zh"
    return "zh"

def process_text(text: str, cleaner: str, translate_func=None):
    """
    处理文本，根据 cleaner 类型决定是否加标签或翻译。
    :param text: 输入文本
    :param cleaner: 选择的 cleaner（如 'zh_ja_mixture_cleaners', 'japanese_cleaners', 'chinese_cleaners'）
    :param translate_func: 可选的翻译函数（如果需要翻译）
    :return: 处理后的文本
    """
    lang = detect(text)  # 检测语言
    print(f"Detected language: {lang} cleaner: {cleaner}")

    if cleaner == "zh_ja_mixture_cleaners":
        # 仅检测整体语言占比并在前后加上标签
        if lang == "zh":
            return f"[ZH]{text}[ZH]"
        elif lang == "ja":
            return f"[JA]{text}[JA]"
        return text  # 如果无法识别语言，则保持原样

    elif (cleaner == "japanese_cleaners" or cleaner=="japanese_cleaners2") and lang != "ja":
        return translate_func(text, target_lang="ja") if translate_func else text

    elif cleaner == "chinese_cleaners" and lang != "zh-cn":
        return translate_func(text, target_lang="zh-cn") if translate_func else text

    return text

def mock_translate(text, target_lang):
    print(f"[Translated to {target_lang}]: {text}")
    r=translate(text)
    return r