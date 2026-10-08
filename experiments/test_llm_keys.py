# -*- coding: utf-8 -*-
"""test_llm_keys.py —— 三个平台 API key 连通性测试。

每个平台发一条最小请求，打印 HTTP 状态与回复前 40 字。
用于确认 key ↔ 平台映射正确；若某平台 401，尝试换 key 映射。
"""
import json
import os
import urllib.request

ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
cfg = {}
for line in open(ENV, encoding="utf-8"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()

PROVIDERS = {
    "deepseek": {
        "key": cfg.get("DEEPSEEK_API_KEY", ""),
        "url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "key": cfg.get("DASHSCOPE_API_KEY", ""),
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen-plus",
    },
    "zhipu": {
        "key": cfg.get("ZHIPU_API_KEY", ""),
        "url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "model": "glm-4-air",
    },
}


def ping(name, key, url, model):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 8,
        "temperature": 0,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
            txt = data["choices"][0]["message"]["content"]
            print(f"[{name}] OK  model={model}  reply={txt[:40]!r}")
            return True
    except urllib.error.HTTPError as e:
        print(f"[{name}] HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:150]}")
    except Exception as e:
        print(f"[{name}] ERROR: {type(e).__name__}: {e}")
    return False


if __name__ == "__main__":
    ok = {n: ping(n, p["key"], p["url"], p["model"]) for n, p in PROVIDERS.items()}
    print("\nSUMMARY:", {k: ("OK" if v else "FAIL") for k, v in ok.items()})
