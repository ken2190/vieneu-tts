from pathlib import Path
from typing import Optional, Union, List, Generator, Any, Dict
import numpy as np
import torch
import gc
import logging
import queue
import threading
import asyncio
from collections import defaultdict
from .base import BaseVieneuTTS
from .utils import _compile_codec_with_triton, extract_speech_ids, _linear_overlap_add
from vieneu_utils.phonemize_text import phonemize_with_dict
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks
from neucodec import NeuCodec, DistillNeuCodec

logger = logging.getLogger("Vieneu.vLLM")


class VllmVieNeuTTS(BaseVieneuTTS):
    """
    GPU-optimized VieNeu-TTS using vLLM backend.
    Supports float16 automatically, making it compatible with V100/T4/RTX 20-series GPUs
    that do not support bfloat16 required by LMDeploy.
    """

    def __init__(
        self,
        backbone_repo: str = "pnnbao-ump/VieNeu-TTS",
        backbone_device: str = "cuda",
        codec_repo: str = "neuphonic/distill-neucodec",
        codec_device: str = "cuda",
        gpu_memory_utilization: float = 0.3,
        tp: int = 1,
        enable_prefix_caching: bool = True,
        enable_triton: bool = True,
        max_batch_size: int = 4,
        hf_token: Optional[str] = None,
    ):
        super().__init__()

        if backbone_device != "cuda" and not backbone_device.startswith("cuda:"):
            raise ValueError("vLLM backend requires CUDA device")

        # Streaming configuration (same as FastVieNeuTTS)
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 50
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        self.max_batch_size = max_batch_size
        self._ref_cache: Dict[str, Any] = {}
        self.stored_dict = defaultdict(dict)

        self._is_onnx_codec = False
        self._triton_enabled = False

        self._load_backbone_vllm(backbone_repo, gpu_memory_utilization, tp, enable_prefix_caching, hf_token)
        self._load_codec(codec_repo, codec_device, enable_triton)
        self._load_voices(backbone_repo, hf_token)
        self._warmup_model()

        logger.info("✅ VllmVieNeuTTS loaded successfully!")
        logger.info(f"   Max batch size: {self.max_batch_size}")

    def _load_backbone_vllm(self, repo, gpu_memory_utilization, tp, enable_prefix_caching, hf_token=None):
        logger.info(f"Loading backbone with vLLM from: {repo}")
        if hf_token:
            import os
            os.environ["HF_TOKEN"] = hf_token

        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "Failed to import `vllm`. Install with: pip install vieneu[vllm]"
            ) from e

        self.llm = LLM(
            model=repo,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tp,
            enable_prefix_caching=enable_prefix_caching,
            trust_remote_code=True,
        )
        self._backbone_repo = repo
        self._gpu_memory_utilization = gpu_memory_utilization
        self._tp = tp
        self._enable_prefix_caching = enable_prefix_caching

    def _load_codec(self, codec_repo, codec_device, enable_triton):
        logger.info(f"Loading codec from: {codec_repo} on {codec_device}")
        match codec_repo:
            case "neuphonic/neucodec":
                self.codec = NeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/distill-neucodec":
                self.codec = DistillNeuCodec.from_pretrained(codec_repo)
                self.codec.eval().to(codec_device)
            case "neuphonic/neucodec-onnx-decoder-int8":
                if codec_device != "cpu":
                    raise ValueError("ONNX decoder only runs on CPU")
                try:
                    from neucodec import NeuCodecOnnxDecoder
                except ImportError as e:
                    raise ImportError("Failed to import ONNX decoder.") from e
                self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
                self._is_onnx_codec = True
            case _:
                raise ValueError(f"Unsupported codec repository: {codec_repo}")

        if enable_triton and not self._is_onnx_codec and codec_device != "cpu":
            self._triton_enabled = _compile_codec_with_triton(self.codec)

    def _warmup_model(self):
        logger.info("🔥 Warming up vLLM model...")
        try:
            from vllm import SamplingParams
            dummy_codes = list(range(10))
            dummy_prompt = self._format_prompt(dummy_codes, "warmup", "test")
            params = SamplingParams(
                top_p=0.95, top_k=50, temperature=1.0,
                max_tokens=32, min_tokens=5,
            )
            _ = self.llm.generate([dummy_prompt], params)
            logger.info("   ✅ Warmup complete")
        except Exception as e:
            logger.warning(f"   ⚠️ Warmup failed: {e}")

    def _decode(self, codes_str: str) -> np.ndarray:
        speech_ids = extract_speech_ids(codes_str)
        if not speech_ids:
            raise ValueError(
                "No valid speech tokens found in the output. "
                "Hãy thử giảm temperature hoặc thay đổi văn bản đầu vào. "
                "Nếu vẫn gặp lỗi này, hãy thông báo với chúng tôi tại: https://discord.com/invite/yJt8kzjzWZ"
            )

        if self._is_onnx_codec:
            codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
            recon = self.codec.decode_code(codes)
        else:
            with torch.no_grad():
                codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(self.codec.device)
                recon = self.codec.decode_code(codes).cpu().numpy()
        return recon[0, 0, :]

    def _format_prompt(self, ref_codes: Union[List[int], torch.Tensor, np.ndarray], ref_text: str, input_text: str) -> str:
        if isinstance(ref_codes, (torch.Tensor, np.ndarray)):
            ref_codes_list = ref_codes.flatten().tolist()
        else:
            ref_codes_list = ref_codes

        ref_text_phones = phonemize_with_dict(ref_text)
        input_text_phones = phonemize_with_dict(input_text, skip_normalize=True)
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes_list])
        return (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text_phones} {input_text_phones}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

    def infer(self, text: str, ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, max_chars: int = 256, silence_p: float = 0.15, crossfade_p: float = 0.0, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False) -> np.ndarray:

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        if not skip_normalize:
            text = self.normalizer.normalize(text)

        from vllm import SamplingParams
        params = SamplingParams(
            top_p=0.95, top_k=top_k, temperature=temperature,
            max_tokens=2048, min_tokens=40,
        )

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        if len(chunks) == 1:
            prompt = self._format_prompt(ref_codes, ref_text, chunks[0])
            outputs = self.llm.generate([prompt], params)
            wav = self._decode(outputs[0].outputs[0].text)
            wav = self._apply_watermark(wav)
        else:
            all_wavs = self.infer_batch(chunks, ref_codes, ref_text, voice=voice, temperature=temperature, top_k=top_k, skip_normalize=True)
            wav = join_audio_chunks(all_wavs, self.sample_rate, silence_p, crossfade_p)

        return wav

    def infer_batch(self, texts: List[str], ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, max_batch_size: Optional[int] = None, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False) -> List[np.ndarray]:

        if not skip_normalize:
            texts = [self.normalizer.normalize(t) for t in texts]

        max_batch_size = max_batch_size or self.max_batch_size

        ref_codes, ref_text = self._resolve_ref_voice(voice, None, ref_codes, ref_text)

        from vllm import SamplingParams
        params = SamplingParams(
            top_p=0.95, top_k=top_k, temperature=temperature,
            max_tokens=2048, min_tokens=40,
        )

        all_wavs = []
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i : i + max_batch_size]
            prompts = [self._format_prompt(ref_codes, ref_text, text) for text in batch_texts]
            outputs = self.llm.generate(prompts, params)
            batch_codes = [output.outputs[0].text for output in outputs]
            batch_wavs = [self._decode(codes) for codes in batch_codes]
            batch_wavs = [self._apply_watermark(w) for w in batch_wavs]
            all_wavs.extend(batch_wavs)
        return all_wavs

    def infer_stream(self, text: str, ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, max_chars: int = 256, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False) -> Generator[np.ndarray, None, None]:

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        if not skip_normalize:
            text = self.normalizer.normalize(text)

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        for chunk in chunks:
            yield from self._infer_stream_single(chunk, ref_codes, ref_text, temperature, top_k)

    def _infer_stream_single(self, text: str, ref_codes: Union[np.ndarray, torch.Tensor, List[int]], ref_text: str, temperature: float = 1.0, top_k: int = 50) -> Generator[np.ndarray, None, None]:
        if isinstance(ref_codes, (torch.Tensor, np.ndarray)):
            ref_codes_list = ref_codes.flatten().tolist()
        else:
            ref_codes_list = ref_codes

        prompt = self._format_prompt(ref_codes_list, ref_text, text)
        audio_cache = []
        token_cache = [f"<|speech_{idx}|>" for idx in ref_codes_list]
        n_decoded_samples = 0
        n_decoded_tokens = len(ref_codes_list)

        # Use a queue to bridge async vLLM streaming to sync generator
        token_queue: queue.Queue = queue.Queue()
        error_holder: List[Exception] = []

        from vllm import SamplingParams

        params = SamplingParams(
            top_p=0.95, top_k=top_k, temperature=temperature,
            max_tokens=2048, min_tokens=40,
        )

        def _run_async_stream():
            """Run async vLLM streaming in a background thread with its own event loop."""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._async_stream_tokens(prompt, params, token_queue))
            except Exception as e:
                error_holder.append(e)
            finally:
                token_queue.put(None)  # Sentinel

        stream_thread = threading.Thread(target=_run_async_stream, daemon=True)
        stream_thread.start()

        # Read tokens from queue and process (same overlap-add logic as FastVieNeuTTS)
        while True:
            try:
                token_text = token_queue.get(timeout=30.0)
            except queue.Empty:
                break

            if token_text is None:
                break

            if token_text:
                token_cache.append(token_text)

            if len(token_cache[n_decoded_tokens:]) >= self.streaming_frames_per_chunk + self.streaming_lookforward:
                tokens_start = max(n_decoded_tokens - self.streaming_lookback - self.streaming_overlap_frames, 0)
                tokens_end = n_decoded_tokens + self.streaming_frames_per_chunk + self.streaming_lookforward + self.streaming_overlap_frames
                sample_start = (n_decoded_tokens - tokens_start) * self.hop_length
                sample_end = sample_start + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames) * self.hop_length
                curr_codes = token_cache[tokens_start:tokens_end]
                recon = self._decode("".join(curr_codes))
                recon = self._apply_watermark(recon)
                recon = recon[sample_start:sample_end]
                audio_cache.append(recon)

                processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                processed_recon = processed_recon[n_decoded_samples:new_samples_end]
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk
                yield processed_recon

        stream_thread.join(timeout=5.0)

        if error_holder:
            raise error_holder[0]

        # Flush remaining tokens
        remaining_tokens = len(token_cache) - n_decoded_tokens
        if remaining_tokens > 0:
            tokens_start = max(len(token_cache) - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens), 0)
            sample_start = (len(token_cache) - tokens_start - remaining_tokens - self.streaming_overlap_frames) * self.hop_length
            curr_codes = token_cache[tokens_start:]
            recon = self._decode("".join(curr_codes))
            recon = self._apply_watermark(recon)
            recon = recon[sample_start:]
            audio_cache.append(recon)
            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            processed_recon = processed_recon[n_decoded_samples:]
            yield processed_recon

    async def _async_stream_tokens(self, prompt: str, params, token_queue: queue.Queue):
        """Stream tokens from vLLM's async engine and put them into a queue."""
        try:
            from vllm import AsyncEngineArgs, AsyncLLM
            from vllm import RequestOutputKind

            # Create async engine using same config as sync LLM
            engine_args = AsyncEngineArgs(
                model=self._backbone_repo,
                dtype="auto",
                gpu_memory_utilization=self._gpu_memory_utilization,
                tensor_parallel_size=self._tp,
                enable_prefix_caching=self._enable_prefix_caching,
                trust_remote_code=True,
            )

            async_llm = AsyncLLM.from_engine_args(engine_args)

            # Use DELTA output kind for token-level streaming
            streaming_params = params.clone()
            streaming_params.output_kind = RequestOutputKind.DELTA

            async for output in async_llm.generate(prompt, streaming_params):
                for completion in output.outputs:
                    if completion.text:
                        token_queue.put(completion.text)

        except ImportError:
            # Fallback: use sync LLM generate (no streaming, put all tokens at once)
            logger.warning("AsyncLLM not available, falling back to non-streaming vLLM generation")
            outputs = self.llm.generate([prompt], params)
            full_text = outputs[0].outputs[0].text
            # Simulate token-by-token by splitting on speech token boundaries
            import re
            tokens = re.findall(r'<\|speech_\d+\|>', full_text)
            for token in tokens:
                token_queue.put(token)

    def cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def get_optimization_stats(self) -> Dict[str, Any]:
        return {
            'triton_enabled': self._triton_enabled,
            'max_batch_size': self.max_batch_size,
            'cached_references': len(self._ref_cache),
            'active_sessions': len(self.stored_dict),
            'prefix_caching': self._enable_prefix_caching,
        }
