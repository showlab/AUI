import os
import json
import time
import base64
from typing import Dict, Any, Optional, List, Callable
from openai import OpenAI, AzureOpenAI
from .providers.azure_openai import chat_completion as azure_chat
from .providers.azure_openai import chat_stream_completion as azure_chat_stream
from .providers.openai_generic import chat_completion as openai_chat
from .logging_utils import ts_print

class ModelClient:
    """Unified model client supporting multiple models (no extra quota/throttle layer here)."""
    
    def __init__(self):
        self.config = self._load_config()
        self._check_environment_variables()

    # Azure-specific throttling/quotas are removed; direct calls are used.
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and process model configuration (config file must exist and be parseable).

        - Supports environment-variable substitution for all strings of the form ${VARNAME}
        """
        import yaml
        config_path = 'configs/models.yaml'
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing model config: {config_path}")
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        if not isinstance(file_config, dict) or 'models' not in file_config:
            raise ValueError("Invalid models.yaml: missing 'models' key")
        models: Dict[str, Dict[str, Any]] = {}
        for model_name, model_config in file_config.get('models', {}).items():
            resolved: Dict[str, Any] = {}
            for k, v in model_config.items():
                if isinstance(v, str) and v.startswith('${') and v.endswith('}'):
                    env_var = v[2:-1]
                    resolved[k] = os.getenv(env_var)
                else:
                    resolved[k] = v
            models[model_name] = resolved
        return {'models': models}
    
    def _check_environment_variables(self):
        """Check required environment variables."""
        missing_vars = []
        
        for model_name, model_config in self.config['models'].items():
            api_key = model_config.get('api_key')
            if not api_key:
                if model_config['provider'] == 'openai':
                    missing_vars.append(f"OPENAI_API_KEY (for {model_name})")
                elif model_config['provider'] == 'azure_openai':
                    missing_vars.append(f"AZURE_OPENAI_API_KEY (for {model_name})")
                # Gemini uses GCP project authentication; do not enforce key check here
        
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
    
    def _get_client(self, model_name: str):
        """Get underlying SDK client for the configured provider."""
        model_config = self.config['models'][model_name]
        provider = model_config['provider']
        
        if provider == 'azure_openai':
            return AzureOpenAI(
                api_version=model_config.get('api_version', '2024-12-01-preview'),
                azure_endpoint=model_config['azure_endpoint'],
                api_key=model_config['api_key'],
            )
        if provider == 'local':
            return OpenAI(
                base_url=model_config['base_url'],
                api_key=model_config['api_key'],
            )
        if provider == 'gemini':
            import google.genai as genai  # type: ignore
            project = model_config.get('project')
            location = model_config.get('location', 'global')
            return genai.Client(vertexai=True, project=project, location=location)
        # OpenAI-compatible default
        return OpenAI(api_key=model_config['api_key'])
    
    def _build_openai_messages(self, prompt: str, images: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Build OpenAI-compatible messages (text + optional images)."""
        if not images:
            return [{"role": "user", "content": prompt}]
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img_path in images:
            if img_path.startswith("data:image"):
                base64_image = img_path
            elif (("/" in img_path or "\\" in img_path) and
                  not img_path.startswith(("iVBOR", "/9j", "UklG")) and
                  len(img_path) < 1000):
                with open(img_path, "rb") as f:
                    base64_data = base64.b64encode(f.read()).decode()
                base64_image = f"data:image/png;base64,{base64_data}"
            else:
                base64_image = f"data:image/png;base64,{img_path}"
            content.append({
                "type": "image_url",
                "image_url": {"url": base64_image}
            })
        return [{"role": "user", "content": content}]
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check whether the error corresponds to a 429 rate-limit response."""
        error_str = str(error)
        return '429' in error_str
    
    async def call_model_with_gpt5_params(self, model_name: str, prompt: str, 
                                         images: Optional[List[str]] = None,
                                         temperature: float = 0.3,
                                         verbosity: str = "medium", 
                                         reasoning_effort: str = "medium",
                                         stream_callback: Optional[Callable[[str], None]] = None) -> str:
        """Call model API for GPT-5 series with verbosity and reasoning_effort parameters."""
        import asyncio

        client = self._get_client(model_name)
        model_config = self.config['models'][model_name]
        is_local = model_config['provider'] == 'local'
        
        # Build messages payload
        messages = []
        if images:
            content = [{"type": "text", "text": prompt}]
            for image_base64 in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        for attempt in range(5):
            try:
                if model_config['provider'] == 'azure_openai':
                    # Offload synchronous SDK call to a thread to avoid blocking the event loop
                    model_type = model_config.get('type', '').lower()

                    # Streaming path (if callback provided)
                    if stream_callback is not None:
                        max_tokens = model_config.get('max_tokens', 16384)
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(
                            None,
                            lambda: azure_chat_stream(
                                client,
                                model_config['deployment'],
                                messages,
                                max_completion_tokens=max_tokens,
                                stream_callback=stream_callback,
                            ),
                        )

                    # Non-streaming path
                    def _do_call():
                        max_tokens = model_config.get('max_tokens', 16384)
                        if 'o1' in model_type or 'gpt-5' in model_type:
                            return azure_chat(
                                client,
                                model_config['deployment'],
                                messages,
                                max_completion_tokens=max_tokens,
                                temperature=None,
                            )
                        else:
                            return azure_chat(
                                client,
                                model_config['deployment'],
                                messages,
                                max_completion_tokens=max_tokens,
                                temperature=temperature,
                            )

                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, _do_call)
                else:
                    # Other providers use the generic call_model path
                    return await self.call_model(model_name, prompt, images, temperature)
                
            except Exception as e:
                import sys
                ts_print(f"GPT-5 API call error (attempt {attempt + 1}/5): {type(e).__name__}: {str(e)}", file=sys.stderr)
                if self._is_rate_limit_error(e):
                    await asyncio.sleep(2 ** attempt)
                    continue
                if attempt == 4:
                    raise e
                # For non-429 errors, continue retrying with a short delay
                await asyncio.sleep(1)
                continue
        
        raise Exception("Max retries exceeded")

    async def call_model(self, model_name: str, prompt: str, 
                   images: Optional[List[str]] = None,
                   temperature: float = 0.3) -> str:
        """Asynchronously call a model API."""
        import asyncio
        
        client = self._get_client(model_name)
        model_config = self.config['models'][model_name]
        provider = model_config.get('provider')
        
        is_local = provider == 'local'
        max_retries = float('inf') if is_local else 5
        
        attempt = 0
        while True:
            try:
                def _make_request():
                    # Gemini 3 via google-genai interface (supports text + screenshots)
                    if provider == 'gemini':
                        from google.genai import types  # type: ignore
                        parts: List[Any] = [types.Part(text=prompt)]
                        if images:
                            for img in images:
                                mime = "image/png"
                                data_str = img
                                if img.startswith("data:image"):
                                    try:
                                        header, data_str = img.split(",", 1)
                                        mime = header.split(";")[0].split(":", 1)[1] or "image/png"
                                    except Exception:
                                        mime = "image/png"
                                image_bytes = base64.b64decode(data_str)
                                parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
                        thinking_level = types.ThinkingLevel.HIGH if temperature >= 0.5 else types.ThinkingLevel.LOW
                        cfg = types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
                        )
                        resp = client.models.generate_content(
                            model=model_config.get('model'),
                            contents=parts,
                            config=cfg,
                        )
                        return resp.text or ""
                    
                    # Azure OpenAI vs OpenAI-compatible providers
                    if provider == 'azure_openai':
                        model_type = model_config.get('type', '').lower()
                        max_tokens = model_config.get('max_tokens', 16384)
                        if 'o1' in model_type or 'gpt-5' in model_type:
                            return azure_chat(
                                client,
                                model_config['deployment'],
                                self._build_openai_messages(prompt, images),
                                max_completion_tokens=max_tokens,
                                temperature=None,
                            )
                        return azure_chat(
                            client,
                            model_config['deployment'],
                            self._build_openai_messages(prompt, images),
                            max_completion_tokens=max_tokens,
                            temperature=temperature,
                        )
                    
                    # local / openai providers via generic OpenAI-compatible endpoint
                    model_identifier = model_config.get('model', model_config.get('deployment'))
                    max_tokens = model_config.get('max_tokens', 16384)
                    return openai_chat(
                        client,
                        model_identifier,
                        self._build_openai_messages(prompt, images),
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                
                loop = asyncio.get_event_loop()
                response_content = await loop.run_in_executor(None, _make_request)
                
                if is_local and response_content and len(response_content) < 10:
                    ts_print(f"🔍 {model_name} returned short response ({len(response_content)} chars): {repr(response_content)}")
                
                return response_content
                
            except Exception as e:
                attempt += 1
                
                if is_local:
                    error_msg = str(e)[:100]
                    retry_delay = min(2 + attempt * 0.5, 10)
                    import sys
                    ts_print(f"🔄 Local model {model_name} error (attempt {attempt}): {error_msg}... retrying in {retry_delay:.1f}s", file=sys.stderr)
                    sys.stderr.flush()
                    await asyncio.sleep(retry_delay)
                    continue
                
                if self._is_rate_limit_error(e) and attempt <= max_retries:
                    ts_print(f"⏸️ Rate limit (429), retrying in 2s (attempt {attempt}/{max_retries + 1})")
                    await asyncio.sleep(2)
                    continue
                
                raise e
    
    async def call_operator_model(self, prompt: str, screenshot: Optional[str] = None) -> str:
        """Call operator model using computer-use-preview API."""
        import asyncio
        
        client = self._get_client('operator')
        model_config = self.config['models']['operator']
        
        # Build input in OpenAI computer-use format
        content = [{"type": "input_text", "text": prompt}]
        
        if screenshot:
            if screenshot.startswith("data:image"):
                base64_image = screenshot
            elif (("/" in screenshot or "\\" in screenshot) and 
                  not screenshot.startswith(("iVBOR", "/9j", "UklG")) and
                  len(screenshot) < 1000):
                # Screenshot path – load and encode
                with open(screenshot, "rb") as f:
                    base64_data = base64.b64encode(f.read()).decode()
                base64_image = f"data:image/png;base64,{base64_data}"
            else:
                # Raw base64-encoded string
                base64_image = f"data:image/png;base64,{screenshot}"
            
            content.append({
                "type": "input_image",
                "image_url": base64_image
            })
        
        input_data = [{"role": "user", "content": content}]
        
        # Retry mechanism – bounded retries for OpenAI
        max_retries = 5
        attempt = 0
        
        while True:
            try:
                def _make_request():
                    # Use deployment for Azure OpenAI, model for regular OpenAI
                    model_param = model_config.get('deployment', model_config.get('type', model_config.get('model')))
                    
                    response = client.responses.create(
                        model=model_param,
                        tools=[{
                            "type": "computer_use_preview",
                            "display_width": model_config.get('display_width', 1920),
                            "display_height": model_config.get('display_height', 1080),
                            "environment": model_config.get('environment', 'browser')
                        }],
                        input=input_data,
                        truncation="auto"
                    )
                    return response
                
                import asyncio
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, _make_request)
                
                # Return raw OpenAI response to be handled by OperatorCUAPolicy
                return response
                
            except Exception as e:
                attempt += 1
                error_msg = str(e)[:100]
                
                # Bounded retry loop
                if attempt > max_retries:
                    ts_print(f"❌ {model_config.get('deployment', 'operator')} model failed after {max_retries} attempts: {error_msg}")
                    raise e
                
                # Retry with fixed delay
                retry_delay = 2
                ts_print(f"🔄 {model_config.get('deployment', 'operator')} model error (attempt {attempt}/{max_retries}): {error_msg}... retrying in {retry_delay}s")
                await asyncio.sleep(retry_delay)
                continue

    async def call_operator_initial(self, prompt: str, screenshot: Optional[str] = None,
                                    *, display_width: int = 1280, display_height: int = 720,
                                    environment: str = 'browser'):
        """Operator initial call using Responses API with computer_use_preview tool (truncation=auto)"""
        import asyncio

        client = self._get_client('operator')
        model_config = self.config['models']['operator']

        # Build input content
        content = [{"type": "input_text", "text": prompt}]
        if screenshot:
            if screenshot.startswith("data:image"):
                base64_image = screenshot
            elif (("/" in screenshot or "\\" in screenshot) and 
                  not screenshot.startswith(("iVBOR", "/9j", "UklG")) and
                  len(screenshot) < 1000):
                with open(screenshot, "rb") as f:
                    base64_data = base64.b64encode(f.read()).decode()
                base64_image = f"data:image/png;base64,{base64_data}"
            else:
                base64_image = f"data:image/png;base64,{screenshot}"
            content.append({"type": "input_image", "image_url": base64_image})

        input_data = [{"role": "user", "content": content}]

        def _make_request():
            model_param = model_config.get('deployment', model_config.get('type', model_config.get('model')))
            return client.responses.create(
                model=model_param,
                tools=[{
                    "type": "computer_use_preview",
                    "display_width": display_width,
                    "display_height": display_height,
                    "environment": environment
                }],
                input=input_data,
                reasoning={"summary": "concise"},
                truncation="auto"
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _make_request)

    async def call_operator_next(self, *, previous_response_id: str, call_id: str, screenshot: str,
                                 display_width: int = 1280, display_height: int = 720,
                                 environment: str = 'browser'):
        """Operator follow-up call with previous_response_id + computer_call_output"""
        import asyncio

        client = self._get_client('operator')
        model_config = self.config['models']['operator']

        # Prepare screenshot as data URL
        if screenshot.startswith("data:image"):
            base64_image = screenshot
        elif (("/" in screenshot or "\\" in screenshot) and 
              not screenshot.startswith(("iVBOR", "/9j", "UklG")) and
              len(screenshot) < 1000):
            with open(screenshot, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode()
            base64_image = f"data:image/png;base64,{base64_data}"
        else:
            base64_image = f"data:image/png;base64,{screenshot}"

        input_data = [{
            "call_id": call_id,
            "type": "computer_call_output",
            "output": {
                "type": "input_image",
                "image_url": base64_image
            }
        }]

        def _make_request():
            model_param = model_config.get('deployment', model_config.get('type', model_config.get('model')))
            return client.responses.create(
                model=model_param,
                previous_response_id=previous_response_id,
                tools=[{
                    "type": "computer_use_preview",
                    "display_width": display_width,
                    "display_height": display_height,
                    "environment": environment
                }],
                input=input_data,
                truncation="auto"
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _make_request)
    
    async def call_coder(self, model_name: str, prompt: str, *, verbosity: str = None, reasoning_effort: str = None, stream_callback: Optional[Callable[[str], None]] = None) -> str:
        """Call code-generation model.

        - Supports optional verbosity and reasoning_effort (for GPT-5 series).
        """
        if model_name in ('gpt5', 'gpt5.1'):
            v = verbosity if verbosity else "low"
            r = reasoning_effort if reasoning_effort else "low"
            return await self.call_model_with_gpt5_params(
                model_name, prompt, temperature=0.7, verbosity=v, reasoning_effort=r,
                stream_callback=stream_callback
            )
        return await self.call_model(model_name, prompt, temperature=0.7)
    
    async def call_judge(self, prompt: str, images: Optional[List[str]] = None) -> str:
        """Call judge model (GPT-5.1)."""
        return await self.call_model('gpt5.1', prompt, images, temperature=0.3)
    
    async def call_task_generator(self, prompt: str) -> str:
        """Call task-generation model (GPT-5.1)."""
        return await self.call_model('gpt5.1', prompt, temperature=0.3)
    
    async def call_commenter(self, model_name: str, prompt: str, images: List[str]) -> str:
        """Call commenter model for version selection (optimized for short analyses)."""
        # GPT-5 commenter: low reasoning_effort; GPT-5.1 commenter: none
        if model_name == 'gpt5':
            return await self.call_model_with_gpt5_params(
                model_name, prompt, images,
                temperature=0.3, verbosity="low", reasoning_effort="low"
            )
        if model_name == 'gpt5.1':
            return await self.call_model_with_gpt5_params(
                model_name, prompt, images,
                temperature=0.3, verbosity="low", reasoning_effort="none"
            )
        return await self.call_model(model_name, prompt, images, temperature=0.3)
    
    async def call_cua_model(self, model_name: str, prompt: str, images: Optional[List[str]] = None) -> str:
        """Call CUA model (UI-TARS or operator)."""
        if model_name == 'operator':
            # Operator model uses a dedicated API
            screenshot = images[0] if images else None
            return await self.call_operator_model(prompt, screenshot)
        else:
            # UI-TARS and other CUA models use generic call_model
            return await self.call_model(model_name, prompt, images, temperature=0.3)
