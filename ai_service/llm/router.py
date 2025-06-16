from typing import Optional
import httpx
import json
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel
import traceback

class LLMResponse(BaseModel):
    text: str
    model: str
    tokens_used: int

class LLMRouter:
    def __init__(self, ollama_host: str = "ollama"):
        """Initialize LLM router with Ollama host.
        
        Args:
            ollama_host: Hostname for Ollama service (default: "ollama" for Docker service name)
        """
        self.client = httpx.AsyncClient(base_url=f"http://{ollama_host}:11434", timeout=240.0)
        self.default_model = "mistral"
        
    async def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Generate response using Ollama.
        
        Args:
            prompt: Input prompt
            model: Model to use (defaults to mistral)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        model = model or self.default_model
        
        try:

            try:
                tags_response = await self.client.get("/api/tags")
                tags_response.raise_for_status()
                available_models = tags_response.json().get("models", [])

                # Allow prefix match: 'mistral' matches 'mistral:latest'
                model_found = any(
                    m.get("name", "").startswith(model) for m in available_models
                )
                if not model_found:
                    raise Exception(f"Model {model} is not available in Ollama")
            except Exception as e:

                traceback.print_exc()
                raise Exception(f"Ollama service not ready or model not available: {str(e)}")
            
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": True,          # tell Ollama to stream
            }

            async with self.client.stream(
                    "POST",              # HTTP verb
                    "/api/generate",     # path
                    json=payload
            ) as resp:
                resp.raise_for_status()

                text_chunks = []
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("{"):
                        continue
                    data = json.loads(line)
                    if "response" in data:
                        text_chunks.append(data["response"])
                    if data.get("done"):
                        break            # end of stream

            final_text = "".join(text_chunks).strip()
            return final_text
        except httpx.TimeoutException:
            print("[ERROR] Request timed out - model may still be loading")
            traceback.print_exc()
            raise Exception("Request timed out - model may still be loading")
        except Exception as e:
            print(f"[ERROR] Exception in generate_response: {e}")
            traceback.print_exc()
            raise Exception(f"Error generating response: {str(e)}")
    
    async def close(self):
        await self.client.aclose()
