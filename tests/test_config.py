"""配置层测试。

这一层是"看起来无关紧要、出问题却最难查"的地方，重点覆盖三条真实踩过的坑：
1. **YAML 与内置 DEFAULTS 漂移**：配置文件写错一个键名不会报错，只会静默失效；
2. **显式 `--config` 解析失败必须报错**（否则"配置没生效"会伪装成"实验跑成功了"）；
3. **自动发现的默认配置坏掉时要退回内置默认值**（保证"只装 numpy 也能跑"这条主线）。
"""

import importlib.util
import os
import unittest

from rag.config import (
    DEFAULTS,
    Config,
    default_config,
    find_project_root,
    load_config,
    parse_scalar,
)

try:  # 兼容两种 discover 调用方式
    from tests.support import PROJECT_ROOT, TempWorkspace
except ImportError:  # pragma: no cover
    from support import PROJECT_ROOT, TempWorkspace

HAS_YAML = importlib.util.find_spec("yaml") is not None


def flatten(node, prefix=""):
    out = {}
    for key, value in (node or {}).items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, full))
        else:
            out[full] = value
    return out


class ParseScalarTest(unittest.TestCase):
    def test_scalars(self):
        self.assertIs(parse_scalar("true"), True)
        self.assertIs(parse_scalar("False"), False)
        self.assertIsNone(parse_scalar("none"))
        self.assertEqual(parse_scalar("42"), 42)
        self.assertAlmostEqual(parse_scalar("0.25"), 0.25)
        self.assertEqual(parse_scalar("bge"), "bge")

    def test_inline_json(self):
        self.assertEqual(parse_scalar("[1, 3, 5]"), [1, 3, 5])
        self.assertEqual(parse_scalar('{"a": 1}'), {"a": 1})


class OverrideTest(unittest.TestCase):
    def test_overrides_apply_and_keep_defaults(self):
        cfg = Config.from_dict({}, overrides={"retrieval.rerank.enabled": True, "chunking.max_chars": 256})
        self.assertTrue(cfg.get("retrieval.rerank.enabled"))
        self.assertEqual(cfg.get("chunking.max_chars"), 256)
        self.assertEqual(cfg.get("chunking.overlap_chars"), DEFAULTS["chunking"]["overlap_chars"])

    def test_with_overrides_is_a_new_object(self):
        base = default_config()
        changed = base.with_overrides({"retrieval.top_k": 9})
        self.assertEqual(changed.get("retrieval.top_k"), 9)
        self.assertEqual(base.get("retrieval.top_k"), DEFAULTS["retrieval"]["top_k"])

    def test_falsy_values_are_not_treated_as_missing(self):
        cfg = Config.from_dict({"retrieval": {"weights": {"vector": 0.0, "bm25": 1.0}}})
        self.assertEqual(cfg.get("retrieval.weights.vector"), 0.0)
        self.assertFalse(cfg.get("retrieval.rerank.enabled"))
        self.assertEqual(cfg.get("retrieval.weights.missing", "fallback"), "fallback")

    def test_section_and_describe_masks_secrets(self):
        cfg = Config.from_dict({"generation": {"api_key_env": "MY_SECRET_KEY"}})
        self.assertIn("weights", cfg.section("retrieval"))
        described = cfg.describe()
        self.assertIn("generation.api_key_env", described)
        self.assertIn("***", described)
        self.assertNotIn("MY_SECRET_KEY", described)

    def test_resolve_uses_root(self):
        cfg = Config.from_dict({}, root=PROJECT_ROOT)
        self.assertEqual(cfg.resolve("data/corpus"), os.path.join(PROJECT_ROOT, "data", "corpus"))
        # 绝对路径原样保留（注意用 normpath 比较，Windows 下 Path 会统一成反斜杠）
        self.assertEqual(
            os.path.normpath(cfg.resolve("C:/abs/path")), os.path.normpath("C:/abs/path")
        )


class ConfigFileTest(unittest.TestCase):
    def test_project_yaml_matches_builtin_defaults(self):
        """防止 YAML 与 DEFAULTS 漂移：写错键名会静默失效，必须显式对齐。"""
        cfg = load_config()
        if not cfg.path or not str(cfg.path).endswith((".yaml", ".yml")):
            self.skipTest("当前没有 YAML 配置（可能未安装 pyyaml，退回内置默认值）")
        yaml_flat = flatten(cfg.merged())
        default_flat = flatten(default_config().merged())
        drift = {
            key: (yaml_flat.get(key, "<missing>"), default_flat.get(key, "<missing>"))
            for key in set(yaml_flat) | set(default_flat)
            if yaml_flat.get(key, "<missing>") != default_flat.get(key, "<missing>")
        }
        self.assertEqual(drift, {}, f"configs/default.yaml 与内置 DEFAULTS 不一致：{drift}")

    def test_json_config_is_supported(self):
        with TempWorkspace(prefix="config-") as root:
            path = os.path.join(root, "custom.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"chunking": {"max_chars": 321}, "retrieval": {"top_k": 7}}')
            cfg = load_config(path=path, root=root)
        self.assertEqual(cfg.get("chunking.max_chars"), 321)
        self.assertEqual(cfg.get("retrieval.top_k"), 7)
        self.assertEqual(cfg.get("chunking.overlap_chars"), DEFAULTS["chunking"]["overlap_chars"])

    def test_explicit_broken_config_raises(self):
        """显式 --config 解析失败必须抛错（BUG-8 回归）。"""
        with TempWorkspace(prefix="config-") as root:
            path = os.path.join(root, "broken.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("chunking: [1, 2\n")  # 语法错误
            with self.assertRaises(Exception):
                load_config(path=path, root=root)

    def test_missing_explicit_config_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config(path=os.path.join(PROJECT_ROOT, "configs", "definitely-missing.yaml"))

    def test_auto_discovered_broken_config_falls_back(self):
        """自动发现的默认配置坏掉时退回内置 DEFAULTS，而不是让 CLI 直接崩（BUG-1 回归）。"""
        with TempWorkspace(prefix="config-") as root:
            configs = os.path.join(root, "configs")
            os.makedirs(configs, exist_ok=True)
            with open(os.path.join(configs, "default.yaml"), "w", encoding="utf-8") as handle:
                handle.write("chunking: [1, 2\n")
            cfg = load_config(root=root)
        self.assertEqual(cfg.get("chunking.max_chars"), DEFAULTS["chunking"]["max_chars"])


class ProjectRootTest(unittest.TestCase):
    def test_find_project_root_locates_pyproject(self):
        root = find_project_root()
        self.assertTrue(os.path.isfile(os.path.join(root, "pyproject.toml")))

    def test_path_of_resolves_known_keys(self):
        cfg = default_config(root=PROJECT_ROOT)
        self.assertTrue(cfg.path_of("corpus").endswith(os.path.join("data", "corpus")))


if __name__ == "__main__":
    unittest.main()
