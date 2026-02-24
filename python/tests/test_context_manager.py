"""
Unit tests for CactusModel and CactusIndex context managers.
Run with: python -m unittest python/tests/test_context_manager.py -v
"""
import sys
import unittest
from unittest.mock import MagicMock, patch

mock_lib = MagicMock()
mock_ctypes = MagicMock()
mock_ctypes.CDLL.return_value = mock_lib
mock_ctypes.CFUNCTYPE.return_value = MagicMock()
mock_ctypes.c_char_p = MagicMock()
mock_ctypes.c_void_p = MagicMock()
mock_ctypes.c_bool = MagicMock()
mock_ctypes.c_int = MagicMock()
mock_ctypes.c_uint32 = MagicMock()
mock_ctypes.c_float = MagicMock()
mock_ctypes.c_uint8 = MagicMock()
mock_ctypes.c_size_t = MagicMock()

with patch.dict(sys.modules, {"ctypes": mock_ctypes}):
    with patch("pathlib.Path.exists", return_value=True):
        sys.modules.pop("cactus", None)

        import os
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        import cactus

MOCK_HANDLE = MagicMock(name="mock_model_handle")
MOCK_INDEX_HANDLE = MagicMock(name="mock_index_handle")


class TestCactusModelContextManager(unittest.TestCase):

    @patch.object(cactus, "cactus_destroy")
    @patch.object(cactus, "cactus_init", return_value=MOCK_HANDLE)
    def test_with_calls_destroy(self, mock_init, mock_destroy):
        with cactus.CactusModel("weights/test") as model:
            self.assertIsNotNone(model._handle)

        mock_init.assert_called_once_with("weights/test", None, False)
        mock_destroy.assert_called_once_with(MOCK_HANDLE)

    @patch.object(cactus, "cactus_destroy")
    @patch.object(cactus, "cactus_init", return_value=MOCK_HANDLE)
    def test_with_calls_destroy_on_exception(self, mock_init, mock_destroy):
        with self.assertRaises(ValueError):
            with cactus.CactusModel("weights/test") as model:
                raise ValueError("simulated error")

        mock_destroy.assert_called_once_with(MOCK_HANDLE)

    @patch.object(cactus, "cactus_destroy")
    @patch.object(cactus, "cactus_init", return_value=MOCK_HANDLE)
    def test_double_destroy_safe(self, mock_init, mock_destroy):
        model = cactus.CactusModel("weights/test")
        model.destroy()
        model.destroy()

        mock_destroy.assert_called_once_with(MOCK_HANDLE)

    @patch.object(cactus, "cactus_destroy")
    @patch.object(cactus, "cactus_init", return_value=MOCK_HANDLE)
    def test_method_after_destroy_raises(self, mock_init, mock_destroy):
        model = cactus.CactusModel("weights/test")
        model.destroy()

        with self.assertRaises(RuntimeError):
            model.complete([])

        with self.assertRaises(RuntimeError):
            model.embed("text")

        with self.assertRaises(RuntimeError):
            model.transcribe("audio.wav")


class TestCactusIndexContextManager(unittest.TestCase):

    @patch.object(cactus, "cactus_index_destroy")
    @patch.object(cactus, "cactus_index_init", return_value=MOCK_INDEX_HANDLE)
    def test_with_calls_destroy(self, mock_init, mock_destroy):
        with cactus.CactusIndex("/tmp/index", 384) as index:
            self.assertIsNotNone(index._handle)

        mock_init.assert_called_once_with("/tmp/index", 384)
        mock_destroy.assert_called_once_with(MOCK_INDEX_HANDLE)

    @patch.object(cactus, "cactus_index_destroy")
    @patch.object(cactus, "cactus_index_init", return_value=MOCK_INDEX_HANDLE)
    def test_with_calls_destroy_on_exception(self, mock_init, mock_destroy):
        with self.assertRaises(RuntimeError):
            with cactus.CactusIndex("/tmp/index", 384) as index:
                raise RuntimeError("simulated error")

        mock_destroy.assert_called_once_with(MOCK_INDEX_HANDLE)

    @patch.object(cactus, "cactus_index_destroy")
    @patch.object(cactus, "cactus_index_init", return_value=MOCK_INDEX_HANDLE)
    def test_double_destroy_safe(self, mock_init, mock_destroy):
        index = cactus.CactusIndex("/tmp/index", 384)
        index.destroy()
        index.destroy()

        mock_destroy.assert_called_once_with(MOCK_INDEX_HANDLE)

    @patch.object(cactus, "cactus_index_destroy")
    @patch.object(cactus, "cactus_index_init", return_value=MOCK_INDEX_HANDLE)
    def test_method_after_destroy_raises(self, mock_init, mock_destroy):
        index = cactus.CactusIndex("/tmp/index", 384)
        index.destroy()

        with self.assertRaises(RuntimeError):
            index.query([0.1, 0.2])

        with self.assertRaises(RuntimeError):
            index.add([1], ["doc"], [[0.1]])

        with self.assertRaises(RuntimeError):
            index.compact()


if __name__ == "__main__":
    unittest.main()
