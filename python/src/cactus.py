"""
Cactus FFI Python Bindings

Python bindings for Cactus Engine via FFI. Provides access to:
- Text completion with LLMs (including cloud handoff detection)
- Audio transcription with Whisper models
- Voice Activity Detection (VAD) for speech segment detection
- Text, image, and audio embeddings
- RAG (Retrieval-Augmented Generation) queries
- Tool RAG (automatic tool selection based on query relevance)
- Streaming transcription
- Vector index for similarity search

Response Format:
All completion responses use a unified JSON format with all fields always present:
{
    "success": bool,        # True if generation succeeded
    "error": str|null,      # Error message if failed, null otherwise
    "cloud_handoff": bool,  # True if model recommends deferring to cloud
    "response": str|null,   # Generated text, null if cloud_handoff or error
    "function_calls": [],   # List of function calls if tools were used
    "confidence": float,    # Model confidence (1.0 - normalized_entropy)
    "time_to_first_token_ms": float,
    "total_time_ms": float,
    "prefill_tps": float,
    "decode_tps": float,
    "ram_usage_mb": float,
    "prefill_tokens": int,
    "decode_tokens": int,
    "total_tokens": int
}
"""
import ctypes
import json
import platform
from pathlib import Path

TokenCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p)

_DIR = Path(__file__).parent.parent.parent
if platform.system() == "Darwin":
    _LIB_PATH = _DIR / "cactus" / "build" / "libcactus.dylib"
else:
    _LIB_PATH = _DIR / "cactus" / "build" / "libcactus.so"

if not _LIB_PATH.exists():
    raise RuntimeError(
        f"Cactus library not found at {_LIB_PATH}\n"
        f"Please build first: cactus build --python"
    )

_lib = ctypes.CDLL(str(_LIB_PATH))

_lib.cactus_set_telemetry_environment.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.cactus_set_telemetry_environment.restype = None
_lib.cactus_set_telemetry_environment(b"python", None)

_lib.cactus_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool]
_lib.cactus_init.restype = ctypes.c_void_p

_lib.cactus_complete.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
    ctypes.c_char_p, ctypes.c_char_p, TokenCallback, ctypes.c_void_p
]
_lib.cactus_complete.restype = ctypes.c_int

_lib.cactus_transcribe.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_size_t, ctypes.c_char_p, TokenCallback, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
]
_lib.cactus_transcribe.restype = ctypes.c_int

_lib.cactus_embed.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), ctypes.c_bool
]
_lib.cactus_embed.restype = ctypes.c_int

_lib.cactus_image_embed.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
]
_lib.cactus_image_embed.restype = ctypes.c_int

_lib.cactus_audio_embed.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
]
_lib.cactus_audio_embed.restype = ctypes.c_int

_lib.cactus_vad.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
    ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
]
_lib.cactus_vad.restype = ctypes.c_int

_lib.cactus_reset.argtypes = [ctypes.c_void_p]
_lib.cactus_reset.restype = None

_lib.cactus_stop.argtypes = [ctypes.c_void_p]
_lib.cactus_stop.restype = None

_lib.cactus_destroy.argtypes = [ctypes.c_void_p]
_lib.cactus_destroy.restype = None

_lib.cactus_get_last_error.argtypes = []
_lib.cactus_get_last_error.restype = ctypes.c_char_p

_lib.cactus_tokenize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.cactus_tokenize.restype = ctypes.c_int

_lib.cactus_score_window.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_size_t,
]
_lib.cactus_score_window.restype = ctypes.c_int

_lib.cactus_rag_query.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_size_t, ctypes.c_size_t
]
_lib.cactus_rag_query.restype = ctypes.c_int

_lib.cactus_stream_transcribe_start.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.cactus_stream_transcribe_start.restype = ctypes.c_void_p

_lib.cactus_stream_transcribe_process.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.c_char_p, ctypes.c_size_t
]
_lib.cactus_stream_transcribe_process.restype = ctypes.c_int

_lib.cactus_stream_transcribe_stop.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t
]
_lib.cactus_stream_transcribe_stop.restype = ctypes.c_int

_lib.cactus_index_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
_lib.cactus_index_init.restype = ctypes.c_void_p

_lib.cactus_index_add.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.c_size_t,
    ctypes.c_size_t
]
_lib.cactus_index_add.restype = ctypes.c_int

_lib.cactus_index_delete.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_size_t
]
_lib.cactus_index_delete.restype = ctypes.c_int

_lib.cactus_index_query.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.c_size_t)
]
_lib.cactus_index_query.restype = ctypes.c_int

_lib.cactus_index_compact.argtypes = [ctypes.c_void_p]
_lib.cactus_index_compact.restype = ctypes.c_int

_lib.cactus_index_destroy.argtypes = [ctypes.c_void_p]
_lib.cactus_index_destroy.restype = None


def cactus_set_telemetry_environment(path):
    _lib.cactus_set_telemetry_environment(None, path.encode() if isinstance(path, str) else path)


class CactusModel:
    def __init__(self, model_path, corpus_dir=None, cache_index=False):
        self._handle = _lib.cactus_init(
            model_path.encode() if isinstance(model_path, str) else model_path,
            corpus_dir.encode() if corpus_dir else None,
            cache_index
        )
        if not self._handle:
            err = _lib.cactus_get_last_error()
            raise RuntimeError(f"Failed to load model: {err.decode() if err else 'unknown error'}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
        return False

    def _require_handle(self):
        if self._handle is None:
            raise RuntimeError("Model has been destroyed")

    def destroy(self):
        if self._handle is not None:
            _lib.cactus_destroy(self._handle)
            self._handle = None

    def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        stop_sequences=None,
        include_stop_sequences=None,
        force_tools=False,
        tool_rag_top_k=None,
        confidence_threshold=None,
        callback=None
    ):
        self._require_handle()
        
        if isinstance(messages, list):
            messages_json = json.dumps(messages)
        else:
            messages_json = messages

        tools_json = None
        if tools is not None:
            if isinstance(tools, list):
                tools_json = json.dumps(tools)
            else:
                tools_json = tools

        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        if stop_sequences is not None:
            options["stop_sequences"] = stop_sequences
        if include_stop_sequences is not None:
            options["include_stop_sequences"] = include_stop_sequences
        if force_tools:
            options["force_tools"] = True
        if tool_rag_top_k is not None:
            options["tool_rag_top_k"] = tool_rag_top_k
        if confidence_threshold is not None:
            options["confidence_threshold"] = confidence_threshold

        options_json = json.dumps(options) if options else None

        buf = ctypes.create_string_buffer(65536)
        cb = TokenCallback(callback) if callback else TokenCallback()
        _lib.cactus_complete(
            self._handle,
            messages_json.encode() if isinstance(messages_json, str) else messages_json,
            buf, len(buf),
            options_json.encode() if options_json else None,
            tools_json.encode() if tools_json else None,
            cb, None
        )
        return buf.value.decode("utf-8", errors="ignore")

    def transcribe(self, audio_path, prompt="", callback=None):
        self._require_handle()
        buf = ctypes.create_string_buffer(65536)
        cb = TokenCallback(callback) if callback else TokenCallback()
        _lib.cactus_transcribe(
            self._handle,
            audio_path.encode() if isinstance(audio_path, str) else audio_path,
            prompt.encode() if isinstance(prompt, str) else prompt,
            buf, len(buf),
            None, cb, None, None, 0
        )
        return buf.value.decode("utf-8", errors="ignore")

    def embed(self, text, normalize=False):
        self._require_handle()
        buf = (ctypes.c_float * 4096)()
        dim = ctypes.c_size_t()
        _lib.cactus_embed(
            self._handle,
            text.encode() if isinstance(text, str) else text,
            buf, ctypes.sizeof(buf), ctypes.byref(dim), normalize
        )
        return list(buf[:dim.value])

    def image_embed(self, image_path):
        self._require_handle()
        buf = (ctypes.c_float * 4096)()
        dim = ctypes.c_size_t()
        _lib.cactus_image_embed(
            self._handle,
            image_path.encode() if isinstance(image_path, str) else image_path,
            buf, ctypes.sizeof(buf), ctypes.byref(dim)
        )
        return list(buf[:dim.value])

    def audio_embed(self, audio_path):
        self._require_handle()
        buf = (ctypes.c_float * 4096)()
        dim = ctypes.c_size_t()
        _lib.cactus_audio_embed(
            self._handle,
            audio_path.encode() if isinstance(audio_path, str) else audio_path,
            buf, ctypes.sizeof(buf), ctypes.byref(dim)
        )
        return list(buf[:dim.value])

    def vad(self, audio_path=None, pcm_data=None, options=None):
        self._require_handle()
        if (audio_path is None) == (pcm_data is None):
            raise ValueError("Must provide either audio_path or pcm_data (not both)")
    
        options_json = None
        if options:
            options_json = json.dumps(options) if isinstance(options, dict) else options
    
        buf = ctypes.create_string_buffer(65536)
    
        if pcm_data is not None:
            if isinstance(pcm_data, bytes):
                arr = (ctypes.c_uint8 * len(pcm_data)).from_buffer_copy(pcm_data)
            else:
                arr = (ctypes.c_uint8 * len(pcm_data))(*pcm_data)
            _lib.cactus_vad(
                self._handle, None, buf, len(buf),
                options_json.encode() if options_json else None,
                arr, len(arr)
            )
        else:
            _lib.cactus_vad(
                self._handle,
                audio_path.encode() if isinstance(audio_path, str) else audio_path,
                buf, len(buf),
                options_json.encode() if options_json else None,
                None, 0
            )
    
        return buf.value.decode("utf-8", errors="ignore")

    def reset(self):
        self._require_handle()
        _lib.cactus_reset(self._handle)

    def stop(self):
        self._require_handle()
        _lib.cactus_stop(self._handle)

    def tokenize(self, text):
        self._require_handle()
        needed = ctypes.c_size_t(0)
        rc = _lib.cactus_tokenize(
            self._handle,
            text.encode("utf-8"),
            None,
            0,
            ctypes.byref(needed),
        )
        if rc != 0:
            raise RuntimeError(f"cactus_tokenize length query failed rc={rc}")
    
        n = needed.value
        arr = (ctypes.c_uint32 * n)()
    
        rc = _lib.cactus_tokenize(
            self._handle,
            text.encode("utf-8"),
            arr,
            n,
            ctypes.byref(needed),
        )
        if rc != 0:
            raise RuntimeError(f"cactus_tokenize fetch failed rc={rc}")
    
        return [arr[i] for i in range(n)]

    def score_window(self, tokens, start, end, context):
        self._require_handle()
        buf = ctypes.create_string_buffer(4096)
        n = len(tokens)
        arr = (ctypes.c_uint32 * n)(*tokens)
    
        _lib.cactus_score_window(
            self._handle,
            arr,
            n,
            start,
            end,
            context,
            buf,
            len(buf),
        )
        return json.loads(buf.value.decode("utf-8", errors="ignore"))

    def rag_query(self, query, top_k=5):
        self._require_handle()
        buf = ctypes.create_string_buffer(65536)
        result = _lib.cactus_rag_query(
            self._handle,
            query.encode() if isinstance(query, str) else query,
            buf, len(buf), top_k
        )
        if result != 0:
            return []
        return json.loads(buf.value.decode("utf-8", errors="ignore"))

    def stream_transcribe_start(self, options=None, language="en"):
        self._require_handle()
        if options is None:
            options = {}
        elif isinstance(options, str):
            options = json.loads(options)
    
        options["language"] = language
        options_json = json.dumps(options)
    
        return _lib.cactus_stream_transcribe_start(
            self._handle,
            options_json.encode()
        )

    def stream_transcribe_process(self, stream, pcm_data):
        if isinstance(pcm_data, bytes):
            arr = (ctypes.c_uint8 * len(pcm_data)).from_buffer_copy(pcm_data)
        else:
            arr = (ctypes.c_uint8 * len(pcm_data))(*pcm_data)
    
        buf = ctypes.create_string_buffer(65536)
        _lib.cactus_stream_transcribe_process(stream, arr, len(arr), buf, len(buf))
        return buf.value.decode("utf-8", errors="ignore")

    def stream_transcribe_stop(self, stream):
        buf = ctypes.create_string_buffer(65536)
        _lib.cactus_stream_transcribe_stop(stream, buf, len(buf))
        return buf.value.decode("utf-8", errors="ignore")

    @staticmethod
    def get_last_error():
        result = _lib.cactus_get_last_error()
        return result.decode() if result else None


class CactusIndex:
    def __init__(self, index_dir, embedding_dim):
        self._handle = _lib.cactus_index_init(
            index_dir.encode() if isinstance(index_dir, str) else index_dir,
            embedding_dim
        )
        if not self._handle:
            raise RuntimeError("Failed to initialize vector index")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
        return False

    def _require_handle(self):
        if self._handle is None:
            raise RuntimeError("Index has been destroyed")

    def destroy(self):
        if self._handle is not None:
            _lib.cactus_index_destroy(self._handle)
            self._handle = None

    def add(self, ids, documents, embeddings, metadatas=None):
        self._require_handle()
        count = len(ids)
        embedding_dim = len(embeddings[0]) if embeddings else 0
    
        ids_arr = (ctypes.c_int * count)(*ids)
    
        docs_arr = (ctypes.c_char_p * count)()
        for i, doc in enumerate(documents):
            docs_arr[i] = doc.encode() if isinstance(doc, str) else doc
    
        meta_arr = None
        if metadatas:
            meta_arr = (ctypes.c_char_p * count)()
            for i, meta in enumerate(metadatas):
                meta_arr[i] = meta.encode() if isinstance(meta, str) else meta
    
        emb_ptrs = (ctypes.POINTER(ctypes.c_float) * count)()
        emb_arrays = []
        for i, emb in enumerate(embeddings):
            arr = (ctypes.c_float * len(emb))(*emb)
            emb_arrays.append(arr)
            emb_ptrs[i] = ctypes.cast(arr, ctypes.POINTER(ctypes.c_float))
    
        return _lib.cactus_index_add(
            self._handle,
            ids_arr,
            docs_arr,
            meta_arr,
            emb_ptrs,
            count,
            embedding_dim
        )

    def delete(self, ids):
        self._require_handle()
        count = len(ids)
        ids_arr = (ctypes.c_int * count)(*ids)
        return _lib.cactus_index_delete(self._handle, ids_arr, count)

    def query(self, embedding, top_k=5, options=None):
        self._require_handle()
        embedding_dim = len(embedding)
    
        emb_arr = (ctypes.c_float * embedding_dim)(*embedding)
        emb_ptr = ctypes.cast(emb_arr, ctypes.POINTER(ctypes.c_float))
        emb_ptr_ptr = ctypes.pointer(emb_ptr)
    
        id_buffer = (ctypes.c_int * top_k)()
        score_buffer = (ctypes.c_float * top_k)()
    
        id_ptr = ctypes.cast(id_buffer, ctypes.POINTER(ctypes.c_int))
        score_ptr = ctypes.cast(score_buffer, ctypes.POINTER(ctypes.c_float))
    
        id_size = ctypes.c_size_t(top_k)
        score_size = ctypes.c_size_t(top_k)
    
        id_ptr_ptr = ctypes.pointer(id_ptr)
        score_ptr_ptr = ctypes.pointer(score_ptr)
    
        options_encoded = options.encode() if options else None
    
        result = _lib.cactus_index_query(
            self._handle,
            emb_ptr_ptr,
            1,
            embedding_dim,
            options_encoded,
            id_ptr_ptr,
            ctypes.byref(id_size),
            score_ptr_ptr,
            ctypes.byref(score_size)
        )
    
        if result < 0:
            return []
    
        return [
            {"id": id_buffer[i], "score": score_buffer[i]}
            for i in range(id_size.value)
        ]

    def compact(self):
        self._require_handle()
        return _lib.cactus_index_compact(self._handle)
