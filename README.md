# AI-Fiber-NLC 🌊

> **利用人工智能突破光纖非線性瓶頸** — 開源、模組化、可複現嘅 AI 光纖非線性補償框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## 願景

現代光纖通信系統嘅傳輸容量，正逐步逼近由「非線性香農極限」所定義嘅物理天花板。本項目構建一個完全開源嘅 AI 非線性補償（NLC）框架，連接學術研究與工業應用。

**核心主張：** 喺唔改變現有光纖物理架構嘅前提下，僅升級兩端終端設備同 DSP 算法，係最具成本效益嘅速率提升路線。

## 快速開始

### Colab 體驗

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-fiber-nlc/ai-fiber-nlc/blob/main/notebooks/quickstart_demo.ipynb)

5 分鐘快速體驗：睇到 AI 補償前後嘅星座圖對比。

### 本地安裝

```bash
# 克隆倉庫
git clone https://github.com/ai-fiber-nlc/ai-fiber-nlc.git
cd ai-fiber-nlc

# 安裝依賴
pip install -r requirements.txt

# 或者可編輯安裝（開發模式）
pip install -e ".[dev]"
```

### 運行測試

```bash
pytest tests/ -v
```

### 運行 Benchmark

```bash
# 驗證協議
python -m src.benchmark.protocol result.json

# 或者手動觸發完整 Benchmark（需要 GitHub Actions）
```

## 項目結構

```
ai-fiber-nlc/
├── src/
│   ├── benchmark/     # Benchmark 驗證協議（核心）
│   │   └── protocol.py    # 場景定義 + 評分公式 + 結果驗證
│   ├── models/        # NLC 模型實現
│   │   ├── mlp_nlc.py      # 多層感知機
│   │   ├── baseline_dbp.py # 傳統 DBP 基準
│   │   └── ...
│   ├── data/          # 光纖模擬數據生成
│   └── utils/         # 可視化、配置
├── tests/             # 單元測試
├── benchmark/         # Benchmark 場景配置 + 結果
├── notebooks/         # 互動式 Demo
└── docs/              # 文檔
```

## Benchmark 協議

本項目採用 **結果導向驗證協議**：

| 指標 | 說明 |
|------|------|
| Q-factor 提升 (dB) | AI 模型相對於無補償嘅提升 |
| 歸一化效果 (ΔQ_norm) | 相對於 DBP 基準嘅歸一化得分 |
| 歸一化效率 (E_norm) | 計算成本相對於 DBP 嘅歸一化得分 |
| 複合評分 | 0.7×效果 + 0.3×效率 |

詳細協議見 [Benchmark Protocol](docs/benchmark-protocol.md)。

## 貢獻

我哋歡迎所有形式嘅貢獻！請先閱讀 [CONTRIBUTING.md](CONTRIBUTING.md)。

**快速指引：**

1. Fork 倉庫
2. 創建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交變更 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

**提交 Benchmark 結果？** 用 `Benchmark Submission` Issue 模板，確保包含所有必需欄位。

## 授權

本項目採用 [MIT License](LICENSE)。

## 參考

- Meta-DSP: [arXiv:2311.10416](https://arxiv.org/abs/2311.10416)
- FONTE-EID: [GitHub](https://github.com/FONTE-EID/fiber-optic-transmission-system-modeling)
- OptiCommPy: [GitHub](https://github.com/edsonportosilva/OptiCommPy)
