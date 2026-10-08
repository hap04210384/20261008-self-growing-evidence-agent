# Pipeline 原型说明

本目录是论文"模式库 × LLM 诊断"方法链路的原型代码。

## 结构

- `skab_spike.py` —— 最小端到端链路原型：SKAB 传感时序 → 分位数离散化 → 滑动窗事务化 → 极大频繁项集（MFI）挖掘 → 模式-故障对照分析。

## 重要说明（引擎口径）

- 本原型中的 MFI 挖掘器是**用于打通链路的参考实现**（垂直位图 DFS，精确、输出极大项集），性能不是目标；
- 论文正式版本的挖掘与计数将由**引擎①（AnyFIM，免阈值随时挖掘）与引擎②（TensorFIM，张量核 BMMA 加速）**替换，两者实现将放入 `code/engine_anyfim/` 与 `code/engine_tensorfim/`，并以论文伪代码为准一一对应；
- 替换点在本文件中以 `ENGINE_HOOK` 注释标出，保证"伪代码 + 真代码"双证承诺可兑现。

## 运行

```bash
python skab_spike.py --data ../data/SKAB_raw --out ../results/spike_v1
```
