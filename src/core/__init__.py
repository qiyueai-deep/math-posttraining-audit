"""零第三方依赖的核心层: 表达式、验证器、数据生成、切分、偏好对、结果事实源。

不做 eager import, 这样 `python -m src.core.data_gen` 不会触发 runpy 重复导入警告。
"""

__all__ = ["expressions", "verifier", "data_gen", "splits", "preferences", "results"]
