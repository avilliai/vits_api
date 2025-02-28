# vits服务端
**基本是在MoeGoe的基础上进行了一些小小的改动**

用例(启动服务端后)
```python
import requests

url = "http://localhost:5009/get_audio"
params = {
    "text": "老师好，今天过的怎么样",
    "speaker": "錦あすみ"
}

response = requests.get(url, params=params)

if response.status_code == 200:
    with open("downloaded_audio.mp3", "wb") as f:
        f.write(response.content)
    print("Audio downloaded successfully!")
else:
    print(f"Error: {response.status_code}, {response.text}")

```
# Links
- [MoeGoe](https://github.com/CjangCjengh/MoeGoe)
- [MoeGoe_GUI](https://github.com/CjangCjengh/MoeGoe_GUI)
- [Pretrained models](https://github.com/CjangCjengh/TTSModels)

