"""Agent 工具环境（官方 train_agent.py 的工具/模拟器/安全求值重构）。"""
from __future__ import annotations

import ast
import json
import math
import operator
import re

# ===== 工具 schema =====
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"), "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"), "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"), "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00", "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好", ("今天天气真好", "english"): "The weather is nice today", ("I love programming", "chinese"): "我喜欢编程", ("机器学习很有趣", "english"): "Machine learning is interesting", ("Happy birthday", "chinese"): "生日快乐"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048, "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556}

MOCK_RESULTS = {
    "calculate_math": lambda a: {"result": str(safe_math_eval(a.get("expression", "0")))},
    "unit_converter": lambda a: {"result": round(float(a.get("value", 0)) * UNIT_DATA.get(f"{a.get('from_unit', '').lower()}_{a.get('to_unit', '').lower()}", 1), 4)},
    "get_current_weather": lambda a: (lambda w: {"city": a.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(a.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda a: {"datetime": TIME_DATA.get(a.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": a.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda a: {"from": a.get("from_currency"), "to": a.get("to_currency"), "rate": EXCHANGE_DATA.get((a.get("from_currency"), a.get("to_currency")), 1.0)},
    "translate_text": lambda a: {"translated_text": TRANSLATE_DATA.get((a.get("text"), a.get("target_language")), a.get("text", ""))},
}

CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}


def parse_tool_calls(text: str) -> list[dict]:
    """从模型输出抽取 <tool_call>{json}</tool_call>；坏 JSON 静默跳过。"""
    calls = []
    for m in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            calls.append(json.loads(m.strip()))
        except json.JSONDecodeError:
            pass
    return calls


def execute_tool(name: str, args: dict) -> dict | None:
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return None
    try:
        return fn(args)
    except Exception:  # noqa: BLE001  工具失败返回 None 由奖励层扣分
        return None


def safe_math_eval(expression: str) -> float:
    """AST 白名单求值（替代 eval）：算术 + math 白名单 + pi/e/tau；幂运算防爆炸。"""
    def pow_guard(base, exp):
        if abs(exp) > 1e4 or (abs(base) > 1 and abs(exp) * math.log10(abs(base)) > 100):
            raise ValueError("幂运算结果过大")
        return base ** exp

    def resolve(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute) and getattr(node.value, "id", "") == "math":
            return node.attr
        return None

    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        name = resolve(node)
        if name in consts:
            return consts[name]
        if isinstance(node, ast.UnaryOp):
            return unary_ops[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp):
            return bin_ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Call) and not node.keywords:
            return funcs[resolve(node.func)](*map(walk, node.args))
        raise ValueError(f"不支持的表达式语法: {type(node).__name__}")

    consts = {"pi": math.pi, "e": math.e, "tau": math.tau}
    funcs = {"pow": pow_guard, **{n: getattr(math, n) for n in "sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil trunc fabs fmod hypot gcd degrees radians".split()}}
    bin_ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: pow_guard}
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    chars = str.maketrans({"^": "**", "×": "*", "÷": "/", "−": "-", "²": "**2", "³": "**3", "（": "(", "）": ")"})
    expr = str(expression).translate(chars).strip()
    if not expr or len(expr) > 512:
        raise ValueError("表达式为空或过长")
    try:
        return walk(ast.parse(expr, mode="eval").body)
    except (KeyError, SyntaxError, TypeError):
        raise ValueError("不支持的表达式语法") from None
