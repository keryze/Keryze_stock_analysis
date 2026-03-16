#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, List

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user

from env_settings import FLASK_SECRET_KEY
from stock_cache_db import StockCache


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["SECRET_KEY"] = FLASK_SECRET_KEY

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "请先登录"


ROLES = {
    "admin": {
        "name": "管理员",
        "permissions": ["read", "create", "update", "delete", "manage_users", "change_all_passwords"],
    },
    "developer": {
        "name": "开发者",
        "permissions": ["read", "create", "update", "change_own_password"],
    },
    "viewer": {
        "name": "访客",
        "permissions": ["read", "change_own_password"],
    },
}

USERS = {
    "admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "id": 1,
        "role": "admin",
    },
    "developer": {
        "password": hashlib.sha256("dev123".encode()).hexdigest(),
        "id": 2,
        "role": "developer",
    },
    "viewer": {
        "password": hashlib.sha256("view123".encode()).hexdigest(),
        "id": 3,
        "role": "viewer",
    },
}

WATCHLIST_CONFIGS = {
    "short_term": {
        "label": "短期监控列表",
        "filename": "watchlist_short_term.json",
        "stocks": [],
    },
    "long_term": {
        "label": "长期监控列表",
        "filename": "watchlist_long_term.json",
        "stocks": [],
    },
    "advanced_long_term": {
        "label": "超长期监控列表",
        "filename": "watchlist_long_advanced_long_term.json",
        "stocks": [],
    },
}

SEARCH_CACHE: Dict[str, List[Dict]] = {}
SEARCH_CACHE_TIME: Dict[str, float] = {}


class User(UserMixin):
    def __init__(self, user_id: int, username: str, role: str = "viewer"):
        self.id = user_id
        self.username = username
        self.role = role

    def has_permission(self, permission: str) -> bool:
        return permission in ROLES.get(self.role, {}).get("permissions", [])

    def can_delete(self) -> bool:
        return self.has_permission("delete")

    def can_create(self) -> bool:
        return self.has_permission("create")

    def can_update(self) -> bool:
        return self.has_permission("update")

    def can_manage_users(self) -> bool:
        return self.has_permission("manage_users")

    def can_change_password(self, target_user: str | None = None) -> bool:
        if self.has_permission("change_all_passwords"):
            return True
        return self.has_permission("change_own_password") and target_user == self.username


@login_manager.user_loader
def load_user(user_id: str):
    for username, data in USERS.items():
        if data["id"] == int(user_id):
            return User(data["id"], username, data.get("role", "viewer"))
    return None


def permission_required(permission: str):
    def decorator(func):
        @wraps(func)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for("login"))
            if not current_user.has_permission(permission):
                return jsonify({"status": "error", "message": "权限不足"}), 403
            return func(*args, **kwargs)

        return decorated_function

    return decorator


def normalize_code(code: str) -> str:
    value = str(code).strip()
    if value.isdigit():
        value = value.zfill(6)
    return value


def watchlist_file(list_type: str) -> Path:
    return BASE_DIR / WATCHLIST_CONFIGS[list_type]["filename"]


def get_watchlist_codes(list_type: str | None = None) -> List[str]:
    if list_type:
        config = WATCHLIST_CONFIGS.get(list_type)
        return list(config["stocks"]) if config else []

    seen = set()
    result = []
    for config in WATCHLIST_CONFIGS.values():
        for code in config["stocks"]:
            if code not in seen:
                seen.add(code)
                result.append(code)
    return result


def load_watchlist(list_type: str) -> List[str]:
    path = watchlist_file(list_type)
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return []

    if not isinstance(raw, list):
        return []

    codes = []
    for item in raw:
        code = normalize_code(item)
        if code:
            codes.append(code)
    return list(dict.fromkeys(codes))


def load_all_watchlists():
    for list_type in WATCHLIST_CONFIGS:
        WATCHLIST_CONFIGS[list_type]["stocks"] = load_watchlist(list_type)


def save_watchlist(list_type: str):
    path = watchlist_file(list_type)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(WATCHLIST_CONFIGS[list_type]["stocks"], f, ensure_ascii=False, indent=2)


def save_users():
    with open(BASE_DIR / "users.json", "w", encoding="utf-8") as f:
        json.dump(USERS, f, ensure_ascii=False, indent=2)


def build_stock_snapshot(cache: StockCache, code: str, include_details: bool = True):
    stock = cache.get_stock(code)
    if not stock:
        return None

    if include_details:
        fund = cache.get_fund_flow(code, max_age_hours=48)
        if fund:
            stock["fund_flow"] = fund

        tech = cache.get_tech_indicators(code, max_age_hours=48)
        if tech:
            stock["tech_indicators"] = tech

    return stock


def build_watchlist_groups(cache: StockCache):
    groups = {}
    all_stocks = []
    seen = set()

    for list_type, config in WATCHLIST_CONFIGS.items():
        stocks = []
        for code in config["stocks"]:
            stock = build_stock_snapshot(cache, code, include_details=True)
            if not stock:
                continue

            stock["list_type"] = list_type
            stocks.append(stock)

            if code not in seen:
                seen.add(code)
                all_stocks.append(dict(stock))

        groups[list_type] = {"label": config["label"], "stocks": stocks}

    return groups, all_stocks


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username in USERS:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            if password_hash == USERS[username]["password"]:
                user = User(USERS[username]["id"], username, USERS[username].get("role", "viewer"))
                login_user(user, remember=True)
                return redirect(request.args.get("next") or url_for("index"))
        return render_template("login.html", error="用户名或密码错误")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html", username=current_user.username)


@app.route("/stock/<code>")
@login_required
def stock_detail(code):
    return render_template("stock_detail.html", code=code)


@app.route("/backtest")
@login_required
def backtest_page():
    return render_template("backtest.html")


@app.route("/optimize")
@login_required
def optimize_page():
    return render_template("optimize.html")


@app.route("/stocks-manage")
@login_required
def stocks_manage_page():
    return render_template("stocks_manage.html")


@app.route("/users-manage")
@login_required
def users_manage_page():
    if not current_user.can_manage_users():
        return redirect(url_for("index"))
    return render_template("users_manage.html")


@app.route("/profile")
@login_required
def profile_page():
    return render_template("profile.html")


@app.route("/long-term-select")
@login_required
def long_term_select_page():
    return render_template("long_term_select.html")


@app.route("/stock-selector")
@login_required
def stock_selector_page():
    if not current_user.can_manage_users():
        flash("仅管理员可访问选股中心", "danger")
        return redirect(url_for("index"))
    return render_template("stock_selector.html")


@app.route("/api/stocks")
@login_required
def api_stocks():
    cache = StockCache()
    groups, all_stocks = build_watchlist_groups(cache)
    cache.close()
    return jsonify(
        {
            "status": "success",
            "data": {"groups": groups, "all": all_stocks},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


@app.route("/api/stocks/realtime")
@login_required
def api_stocks_realtime():
    cache = StockCache()
    stocks = []
    for code in get_watchlist_codes():
        stock = cache.get_stock(code)
        if stock:
            stocks.append(
                {
                    "code": stock["code"],
                    "name": stock["name"],
                    "price": stock["price"],
                    "change_pct": stock["change_pct"],
                    "update_time": stock.get("update_time"),
                }
            )
    cache.close()
    return jsonify(
        {
            "status": "success",
            "data": stocks,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


@app.route("/api/stock/<code>")
@login_required
def api_stock_detail(code):
    cache = StockCache()
    stock = build_stock_snapshot(cache, normalize_code(code), include_details=True)
    cache.close()
    if not stock:
        return jsonify({"status": "error", "message": "股票不存在"})
    return jsonify({"status": "success", "data": stock})


@app.route("/api/history/<code>")
@login_required
def api_history(code):
    from tech_indicators import TechIndicatorCalculator

    calc = TechIndicatorCalculator()
    history = calc.get_stock_history(normalize_code(code), days=request.args.get("days", 60, type=int))
    calc.close()
    if history is None or history.empty:
        return jsonify({"status": "error", "message": "获取历史数据失败"})

    data = []
    for date, row in history.iterrows():
        data.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "close": float(row["close"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "volume": float(row["volume"]),
            }
        )
    return jsonify({"status": "success", "data": data})


@app.route("/api/backtest", methods=["POST"])
@login_required
def api_backtest():
    from backtest_engine import BacktestEngine

    data = request.get_json(silent=True) or {}
    symbol = normalize_code(data.get("symbol", ""))
    strategy = data.get("strategy")
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    initial_capital = data.get("initial_capital", 100000)

    if not all([symbol, strategy, start_date, end_date]):
        return jsonify({"status": "error", "message": "参数不完整"})

    engine = BacktestEngine()
    result = engine.backtest(
        symbol=symbol,
        strategy_name=strategy,
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        initial_capital=initial_capital,
    )
    engine.close()
    if result is None:
        return jsonify({"status": "error", "message": "回测失败"})

    trades = []
    for trade in result["trades"]:
        item = dict(trade)
        if hasattr(item["date"], "strftime"):
            item["date"] = item["date"].strftime("%Y-%m-%d")
        trades.append(item)
    result["trades"] = trades
    return jsonify({"status": "success", "data": result})


@app.route("/api/cache/stats")
@login_required
def api_cache_stats():
    cache = StockCache()
    stats = cache.get_cache_stats()
    cache.close()
    stats["watchlist_counts"] = {
        key: len(config["stocks"]) for key, config in WATCHLIST_CONFIGS.items()
    }
    stats["watchlist_total"] = len(get_watchlist_codes())
    return jsonify({"status": "success", "data": stats})


@app.route("/api/stock/<code>/refresh", methods=["POST"])
@login_required
def api_refresh_stock(code):
    def refresh_in_background(stock_code: str):
        try:
            from stock_async_fetcher import StockAsyncFetcher
            from tech_indicators import TechIndicatorCalculator

            fetcher = StockAsyncFetcher()
            fetcher.fetch_and_cache([stock_code])
            fetcher.fetch_fund_flow(stock_code)
            fetcher.close()

            calc = TechIndicatorCalculator()
            result = calc.calculate_indicators(stock_code)
            if result:
                calc.cache.save_tech_indicators(stock_code, result)
            calc.close()
        except Exception as e:
            print(f"refresh failed {stock_code}: {e}")

    thread = threading.Thread(target=refresh_in_background, args=(normalize_code(code),), daemon=True)
    thread.start()
    return jsonify({"status": "success", "message": f"正在后台刷新 {code} 的数据"})


@app.route("/api/watchlist", methods=["GET"])
@login_required
def api_get_watchlist():
    cache = StockCache()
    groups = {}
    for list_type, config in WATCHLIST_CONFIGS.items():
        stocks = []
        for code in config["stocks"]:
            stock = cache.get_stock(code)
            stocks.append(
                {
                    "code": code,
                    "name": stock["name"] if stock else "加载中...",
                    "price": stock.get("price", 0) if stock else 0,
                    "change_pct": stock.get("change_pct", 0) if stock else 0,
                }
            )
        groups[list_type] = {"label": config["label"], "stocks": stocks}
    cache.close()
    return jsonify({"status": "success", "data": groups})


@app.route("/api/watchlist", methods=["POST"])
@login_required
def api_add_to_watchlist():
    data = request.get_json(silent=True) or {}
    code = normalize_code(data.get("code", ""))
    list_type = data.get("list_type", "short_term")

    if not code:
        return jsonify({"status": "error", "message": "股票代码不能为空"})
    if not code.isdigit() or len(code) != 6:
        return jsonify({"status": "error", "message": "股票代码格式错误"})
    if list_type not in WATCHLIST_CONFIGS:
        return jsonify({"status": "error", "message": "监控列表类型无效"})
    if code in WATCHLIST_CONFIGS[list_type]["stocks"]:
        return jsonify({"status": "error", "message": "该股票已在监控列表中"})

    WATCHLIST_CONFIGS[list_type]["stocks"].append(code)
    save_watchlist(list_type)
    return jsonify(
        {
            "status": "success",
            "message": f"成功添加 {code}",
            "data": {"code": code, "name": "待加载", "list_type": list_type},
        }
    )


@app.route("/api/watchlist/<code>", methods=["DELETE"])
@login_required
def api_remove_from_watchlist(code):
    list_type = request.args.get("list_type", "").strip()
    target_types = [list_type] if list_type in WATCHLIST_CONFIGS else list(WATCHLIST_CONFIGS.keys())
    code = normalize_code(code)

    removed = False
    for current_type in target_types:
        if code in WATCHLIST_CONFIGS[current_type]["stocks"]:
            WATCHLIST_CONFIGS[current_type]["stocks"].remove(code)
            save_watchlist(current_type)
            removed = True

    if not removed:
        return jsonify({"status": "error", "message": "该股票不在监控列表中"})
    return jsonify({"status": "success", "message": f"已移除 {code}"})


@app.route("/api/stock/search")
@login_required
def api_search_stock():
    import time
    import akshare as ak

    keyword = request.args.get("q", "").strip()
    if not keyword:
        return jsonify({"status": "error", "message": "搜索关键词不能为空"})

    now = time.time()
    if keyword in SEARCH_CACHE and now - SEARCH_CACHE_TIME.get(keyword, 0) < 300:
        return jsonify({"status": "success", "data": SEARCH_CACHE[keyword], "cached": True})

    try:
        df = ak.stock_zh_a_spot_em()
        mask = df["代码"].astype(str).str.contains(keyword, na=False) | df["名称"].astype(str).str.contains(keyword, na=False)
        results = df[mask].head(10)
        stocks = []
        for _, row in results.iterrows():
            stocks.append(
                {
                    "code": str(row["代码"]),
                    "name": row["名称"],
                    "price": float(row["最新价"]),
                    "change_pct": float(row["涨跌幅"]),
                }
            )
        SEARCH_CACHE[keyword] = stocks
        SEARCH_CACHE_TIME[keyword] = now
        return jsonify({"status": "success", "data": stocks})
    except Exception as e:
        return jsonify({"status": "error", "message": f"搜索失败: {e}"})


@app.route("/api/users", methods=["GET"])
@login_required
def api_get_users():
    if not current_user.can_manage_users():
        return jsonify({"status": "error", "message": "权限不足"}), 403
    users_list = []
    for username, data in USERS.items():
        users_list.append(
            {
                "username": username,
                "role": data.get("role", "viewer"),
                "role_name": ROLES.get(data.get("role", "viewer"), {}).get("name", "未知"),
            }
        )
    return jsonify({"status": "success", "data": users_list})


@app.route("/api/users", methods=["POST"])
@login_required
def api_create_user():
    if not current_user.can_manage_users():
        return jsonify({"status": "error", "message": "权限不足"}), 403

    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    role = data.get("role", "viewer")
    if not username or not password:
        return jsonify({"status": "error", "message": "用户名和密码不能为空"})
    if username in USERS:
        return jsonify({"status": "error", "message": "用户已存在"})
    if role not in ROLES:
        return jsonify({"status": "error", "message": "无效的角色"})

    USERS[username] = {
        "password": hashlib.sha256(password.encode()).hexdigest(),
        "id": max([u["id"] for u in USERS.values()]) + 1,
        "role": role,
    }
    save_users()
    return jsonify({"status": "success", "message": f"用户 {username} 创建成功"})


@app.route("/api/users/<username>", methods=["DELETE"])
@login_required
def api_delete_user(username):
    if not current_user.can_delete():
        return jsonify({"status": "error", "message": "权限不足"}), 403
    if username not in USERS:
        return jsonify({"status": "error", "message": "用户不存在"})
    if username == "admin":
        return jsonify({"status": "error", "message": "不能删除 admin 用户"})
    if username == current_user.username:
        return jsonify({"status": "error", "message": "不能删除自己"})

    del USERS[username]
    save_users()
    return jsonify({"status": "success", "message": f"用户 {username} 已删除"})


@app.route("/api/change-password", methods=["POST"])
@login_required
def api_change_password():
    data = request.get_json(silent=True) or {}
    target_user = data.get("username", current_user.username)
    old_password = data.get("old_password", "")
    new_password = data.get("new_password", "")
    confirm_password = data.get("confirm_password", "")

    if not current_user.can_change_password(target_user):
        return jsonify({"status": "error", "message": "权限不足"})
    if not new_password:
        return jsonify({"status": "error", "message": "新密码不能为空"})
    if len(new_password) < 6:
        return jsonify({"status": "error", "message": "密码长度不能少于6位"})
    if new_password != confirm_password:
        return jsonify({"status": "error", "message": "两次输入的密码不一致"})
    if target_user == current_user.username and not current_user.has_permission("change_all_passwords"):
        old_hash = hashlib.sha256(old_password.encode()).hexdigest()
        if old_hash != USERS[target_user]["password"]:
            return jsonify({"status": "error", "message": "原密码错误"})
    if target_user not in USERS:
        return jsonify({"status": "error", "message": "用户不存在"})

    USERS[target_user]["password"] = hashlib.sha256(new_password.encode()).hexdigest()
    save_users()
    return jsonify({"status": "success", "message": "密码修改成功"})


@app.route("/api/roles", methods=["GET"])
@login_required
def api_get_roles():
    data = []
    for role_id, role_data in ROLES.items():
        data.append({"id": role_id, "name": role_data["name"], "permissions": role_data["permissions"]})
    return jsonify({"status": "success", "data": data})


@app.route("/api/long-term-select", methods=["POST"])
@login_required
def api_long_term_select():
    from long_term_selector import LongTermSelector

    data = request.get_json(silent=True) or {}
    selector = LongTermSelector()
    try:
        stocks = selector.select_top_stocks(top_n=data.get("top_n", 5))
        return jsonify({"status": "success", "data": stocks})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        selector.close()


@app.route("/api/long-term-report", methods=["POST"])
@login_required
def api_long_term_report():
    from long_term_selector import LongTermSelector

    data = request.get_json(silent=True) or {}
    stocks = data.get("stocks", [])
    if not stocks:
        return jsonify({"status": "error", "message": "无数据"})

    selector = LongTermSelector()
    try:
        report = selector.generate_report(stocks)
        return jsonify({"status": "success", "report": report})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        selector.close()


@app.route("/api/selector/run", methods=["POST"])
@login_required
def api_run_selector():
    if not current_user.can_manage_users():
        return jsonify({"status": "error", "message": "权限不足"}), 403

    data = request.get_json(silent=True) or {}
    selector_type = data.get("type", "long")
    top_n = data.get("top_n", 5)
    try:
        if selector_type == "short":
            from short_term_selector import ShortTermSelector

            selector = ShortTermSelector()
        else:
            from long_term_selector import LongTermSelector

            selector = LongTermSelector()
        try:
            stocks = selector.select_top_stocks(top_n=top_n)
            return jsonify({"status": "success", "data": stocks})
        finally:
            selector.close()
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/selector/report", methods=["POST"])
@login_required
def api_get_selector_report():
    if not current_user.can_manage_users():
        return jsonify({"status": "error", "message": "权限不足"}), 403

    data = request.get_json(silent=True) or {}
    selector_type = data.get("type", "long")
    stocks = data.get("stocks", [])
    if not stocks:
        return jsonify({"status": "error", "message": "无数据"})
    try:
        if selector_type == "short":
            from short_term_selector import ShortTermSelector

            selector = ShortTermSelector()
        else:
            from long_term_selector import LongTermSelector

            selector = LongTermSelector()
        try:
            report = selector.generate_report(stocks)
            return jsonify({"status": "success", "report": report})
        finally:
            selector.close()
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/market/overview", methods=["GET"])
@login_required
def api_market_overview():
    from market_analysis import MarketAnalysis

    cache = StockCache()
    stocks = []
    for code in get_watchlist_codes():
        stock = cache.get_stock(code)
        if stock:
            stocks.append(stock)
    cache.close()

    analyzer = MarketAnalysis()
    return jsonify({"status": "success", "data": analyzer.get_market_overview(stocks)})


@app.route("/api/market/sentiment")
def api_market_sentiment():
    try:
        from market_sentiment import calculate_market_sentiment

        sentiment = calculate_market_sentiment(use_demo_data=False)
        if sentiment["stats"]["total"] == 0:
            sentiment = calculate_market_sentiment(use_demo_data=True)
            sentiment["demo_mode"] = True
        return jsonify({"status": "success", "data": sentiment})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/enhanced-selector/run", methods=["POST"])
@login_required
def api_run_enhanced_selector():
    if not current_user.can_manage_users():
        return jsonify({"status": "error", "message": "权限不足"}), 403

    from enhanced_long_term_selector import EnhancedLongTermSelector

    selector = EnhancedLongTermSelector()
    try:
        stocks = selector.select_top_stocks(top_n=(request.get_json(silent=True) or {}).get("top_n", 5))
        return jsonify({"status": "success", "data": stocks})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        selector.close()


if __name__ == "__main__":
    load_all_watchlists()
    total = len(get_watchlist_codes())
    print("A股量化监控系统 Web界面")
    for key, config in WATCHLIST_CONFIGS.items():
        print(f"  {config['label']}: {len(config['stocks'])} 只")
    print(f"  合计唯一股票: {total} 只")
    print("  访问: http://localhost:5000")
    print("  默认账号: admin / admin123")
    app.run(host="0.0.0.0", port=5000, debug=True)
