import unittest

from rag.device import resolve_device, device_report


class ResolveDeviceTest(unittest.TestCase):
    def test_explicit_device_passthrough(self):
        self.assertEqual(resolve_device("cpu"), "cpu")
        self.assertEqual(resolve_device("cuda"), "cuda")  # 可用性交给上层报错
        self.assertEqual(resolve_device("MPS"), "mps")

    def test_auto_returns_a_valid_device(self):
        self.assertIn(resolve_device("auto"), ("cpu", "cuda", "mps"))

    def test_empty_string_treated_as_auto(self):
        self.assertIn(resolve_device(""), ("cpu", "cuda", "mps"))

    def test_report_is_safe_without_torch(self):
        report = device_report()
        for key in ("selected", "torch", "cuda_available", "gpu", "reason"):
            self.assertIn(key, report)
        self.assertIn(report["selected"], ("cpu", "cuda", "mps"))


if __name__ == "__main__":
    unittest.main()
