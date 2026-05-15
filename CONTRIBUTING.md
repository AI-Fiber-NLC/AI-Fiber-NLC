# CONTRIBUTING.md — AI-Fiber-NLC

> 歡迎貢獻！以下係快速指引。

## 開發者設置

```bash
# 克隆同安裝
git clone https://github.com/ai-fiber-nlc/ai-fiber-nlc.git
cd ai-fiber-nlc
pip install -e ".[dev]"

# 驗證安裝
pytest tests/ -v
ruff check src/ tests/
```

## 提交類型

### 1. 代碼貢獻（模型 / Bug 修復 / 功能）

1. 開 Issue 討論（如果唔係 trivial fix）
2. Fork → 分支 → 開發
3. 確保 `pytest` + `ruff` 通過
4. 開 PR，填好 PR 模板

### 2. Benchmark 提交

1. 用 `Benchmark Submission` Issue 模板
2. 確保包含所有必需欄位
3. 結果必須由官方 benchmark script 生成
4. 附 contribution signature（見下方）

### 3. 文檔貢獻

直接開 PR，改 `docs/` 或者 `README.md`。

## Contribution Signature（Ed25519 簽名）

Phase 0-1 使用 Ed25519 本地簽名方案：

```bash
# 生成金鑰對（首次）
python tools/sign.py keygen --author "Your Name <email>"

# 為模型文件生成簽名
python tools/sign.py sign --model-file models/my_model.pt
```

簽名文件（`*.sig`）隨 PR 一併提交。Maintainer 驗證後合併。

## 代碼風格

- **Lint**: `ruff`（自動格式化）
- **Type check**: `mypy`
- **測試**: `pytest`

```bash
# 格式化
ruff format src/ tests/

# 檢查
ruff check src/ tests/
mypy src/
```

## Benchmark 提交 Checklist

- [ ] 結果由官方 benchmark script 生成
- [ ] 包含至少 3 個功率點
- [ ] 使用場景配置中指定嘅 seed
- [ ] 附 contribution signature
- [ ] 填好 Benchmark Submission 模板所有欄位

## 行為準則

請閱讀 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

## 問題？

開 [Issue](https://github.com/ai-fiber-nlc/ai-fiber-nlc/issues) 或者喺 [Discussion](https://github.com/ai-fiber-nlc/ai-fiber-nlc/discussions) 提問。
