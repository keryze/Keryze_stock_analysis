# GitHub Pages 部署说明

把选股结果做成一个自动更新的网页：**GitHub Actions 跑分析 → 生成 JSON → 发布到 GitHub Pages**。

## 架构

```
GitHub Actions (定时/手动触发)
  └─ python scripts/run_selectors.py
       ├─ update_all_market_data.py   # 刷新全市场缓存
       ├─ short_term_selector.py      # 短线 TOP10 → scripts/site/data/short_term.json
       └─ long_term_selector.py       # 中长线 TOP10 → scripts/site/data/long_term.json
  └─ 部署 scripts/site/ 到 GitHub Pages
GitHub Pages
  └─ scripts/site/index.html          # 读取 data/*.json 渲染榜单
```

- **TUSHARE_TOKEN 存放在仓库 Secrets 里**，不会出现在网页上，也不会被别人用你的额度。
- 网页是静态结果展示页，不在浏览器里直接拉数据（Tushare 有跨域限制、计算量也太大）。

## 一次性配置（3 步）

### 1. 配置 Secrets
仓库 → **Settings → Secrets and variables → Actions → New repository secret**，至少添加：

| Name | 说明 |
|---|---|
| `TUSHARE_TOKEN` | 必填。tushare.pro 的 token |
| `EASTMONEY_COOKIE` | 可选。东方财富资金流 cookie，缺失只是少一部分资金分 |
| `EMAIL_SENDER` / `EMAIL_PASSWORD` / `EMAIL_RECEIVER` | 可选。要邮件推送才需要 |

### 2. 开启 Pages
仓库 → **Settings → Pages → Build and deployment → Source** 选 **GitHub Actions**。

### 3. 跑一次
仓库 → **Actions → Analyze and Deploy Pages → Run workflow**。
跑完后访问 `https://<你的用户名>.github.io/<仓库名>/` 即可看到榜单。

## 自动更新

`.github/workflows/pages.yml` 里已配置每个交易日北京时间约 15:30（收盘后）自动运行：

```yaml
schedule:
  - cron: "30 7 * * 1-5"   # UTC 07:30 = 北京 15:30，周一至周五
```

不需要自动运行就删掉这两行，只保留手动触发。

## 本地预览

```bash
cd scripts/site
python -m http.server 8000
# 浏览器打开 http://127.0.0.1:8000
```

（本地 `data/*.json` 是空占位，页面会显示"暂无数据"提示；真实数据由 Actions 运行后生成。）

## 二阶段（AI 复筛）

本页只做第一阶段量化初筛。拿到 TOP10 后，仍按原流程把 `scripts/analysis_longterm_prompt.md` /
`analysis_shortterm_prompt.md` + 榜单丢给 Claude/ChatGPT/通义/DeepSeek 做二次筛选。
