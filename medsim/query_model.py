import os
import time
import re
import json
import random
from tqdm import tqdm

import openai
import replicate
import anthropic

from groq import Groq

import sys
import gzip
from collections import Counter
from importlib import import_module

# URLs for replicate models
LLAMA2_URL = "meta/llama-2-70b-chat"
LLAMA3_URL = "meta/meta-llama-3-70b-instruct"
MIXTRAL_URL = "mistralai/mixtral-8x7b-instruct-v0.1"

import time
import requests
import logging
from transformers import pipeline, AutoConfig, AutoModel, AutoTokenizer
import json

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BAgent:
    def __init__(
        self,
        model_name="meta-llama/Llama-3.2-3B-Instruct",  # rarely used when Ollama/remote is enabled
        server_url: str | None = None,
        server_token: str | None = None,
        ollama_url: str | None = None,
        # tên model phải trùng với `ollama list`, ví dụ: llama3.1:8b
        ollama_model="llama3.1:8b",
    ):
        """
        Initializes the BAgent:
        - Optionally uses a remote vLLM/chat-completions server.
        - Otherwise prefers Ollama if available (local or remote).
        - Falls back to loading a Hugging Face model locally.

        Args:
            model_name: Model name for vLLM/transformers (ignored if an LLM server is used)
            server_url: vLLM/chat-completions endpoint. If not set, will read
                from $SERVER_URL or default to http://localhost:8012/v1/chat/completions.
            server_token: Bearer token to add to the remote server requests. Also
                read from $SERVER_TOKEN if not passed explicitly.
            ollama_url: Ollama server URL (default: $OLLAMA_HOST or http://localhost:11434)
            ollama_model: Ollama model name (default: llama3.1)
        """
        # allow overrides via environment variables so users don't have to touch every
        # call site in the codebase
        self.server_url = server_url or os.environ.get(
            "SERVER_URL", "http://localhost:8012/v1/chat/completions"
        )
        self.server_token = server_token or os.environ.get("SERVER_TOKEN")
        self.ollama_url = ollama_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = ollama_model
        # backend cho vLLM / local HF (không liên quan Ollama)
        self.model_name = model_name

        # Nếu người dùng truyền alias dạng "ollama:xxx" hoặc "ollama",
        # thì tự map sang tên model Ollama tương ứng.
        if isinstance(model_name, str):
            if model_name.startswith("ollama:"):
                # Ví dụ: "ollama:llama3.1:8b" → "llama3.1:8b"
                self.ollama_model = model_name.split(":", 1)[1]
            elif model_name.strip().lower() == "ollama":
                # Alias "ollama" mặc định dùng llama3.1:8b
                self.ollama_model = "llama3.1:8b"

        # Check available backends in order of preference
        self.use_server = self._check_vllm_server()
        self.use_ollama = False

        if self.use_server:
            print(f"Using vLLM server: {self.server_url}")
            logger.info(
                f"Using vLLM server at {self.server_url}, skipping local model loading."
            )
        else:
            self.use_ollama = self._check_ollama_server()
            if self.use_ollama:
                print(
                    f"Using Ollama server: {self.ollama_url} with model {self.ollama_model}"
                )
                logger.info(
                    f"Using Ollama at {self.ollama_url} with model {self.ollama_model}"
                )
            else:
                print("No server available, loading model locally...")
                self._load_model()

    def _check_vllm_server(self):
        """Checks if the vLLM/chat-completions server is running."""
        try:
            headers = {}
            if self.server_token:
                headers["Authorization"] = f"Bearer {self.server_token}"
            response = requests.get(
                self.server_url.replace("/v1/chat/completions", "/health"),
                timeout=2,
                headers=headers,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _check_ollama_server(self):
        """Checks if the Ollama server is running."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _load_model(self):
        """Loads the model locally if no vLLM server is found."""
        logger.info("Loading model and tokenizer locally...")
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("Model loaded successfully.")
        except ValueError as e:
            if "Unknown quantization type" in str(e):
                logger.warning(
                    "Quantization not supported. Loading without quantization..."
                )
                try:
                    config = AutoConfig.from_pretrained(
                        self.model_name, trust_remote_code=True
                    )
                    if hasattr(config, "quantization_config"):
                        delattr(config, "quantization_config")

                    model = AutoModel.from_pretrained(
                        self.model_name,
                        config=config,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, trust_remote_code=True
                    )
                    self.pipeline = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    logger.info("Model loaded successfully without quantization.")
                except Exception as e:
                    logger.error(f"Failed to load model without quantization: {e}")
                    raise
            else:
                logger.error(f"Unexpected error during model loading: {e}")
                raise

    def query_model(
        self,
        prompt,
        system_prompt="You are a helpful assistant.",
        tries=5,
        timeout=120,
        image_requested=False,
        scene=None,
        max_prompt_len=2500,
        clip_prompt=False,
        thread_id=1,
    ):
        """Queries available backend: vLLM server > Ollama > local model."""
        if self.use_server:
            return self._query_server(prompt, system_prompt, tries, timeout)
        elif self.use_ollama:
            return self._query_ollama(prompt, system_prompt, tries, timeout)
        return self._query_local(
            prompt,
            system_prompt,
            image_requested,
            scene,
            max_prompt_len,
            clip_prompt,
            tries,
            timeout,
        )

    def _query_ollama(self, user_prompt, system_prompt, tries=5, timeout=120.0) -> str:
        """
        Queries the Ollama server with system and user prompts.

        Ghi chú tương thích phiên bản:
        - Một số bản Ollama chỉ hỗ trợ `/api/generate`.
        - Bản mới hơn hỗ trợ thêm `/api/chat` hoặc `/v1/chat/completions` (OpenAI-compatible).
        - Hàm này sẽ thử lần lượt: `/v1/chat/completions` → `/api/chat` → `/api/generate`.
        - Dùng (connect_timeout, read_timeout) để khi Ollama không chạy thì fail nhanh (~15s), tránh treo vô hạn.
        """
        headers = {"Content-Type": "application/json"}
        connect_timeout = 15  # Nếu Ollama không chạy, kết nối sẽ fail trong 15s
        read_timeout = min(120, max(30, int(timeout)))
        request_timeout = (connect_timeout, read_timeout)

        # Danh sách endpoint thử lần lượt: (chế_độ, URL)
        endpoints = [
            ("v1", f"{self.ollama_url}/v1/chat/completions"),
            ("chat", f"{self.ollama_url}/api/chat"),
            ("generate", f"{self.ollama_url}/api/generate"),
        ]

        last_error: Exception | None = None

        for mode, url in endpoints:
            for attempt in range(tries):
                try:
                    if mode in ("v1", "chat"):
                        # API kiểu chat: nhận messages
                        payload = {
                            "model": self.ollama_model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "stream": False,
                        }
                        # Tham số tên hơi khác nhau giữa v1/chat và api/chat nhưng đều bỏ qua nếu không dùng
                        if mode == "v1":
                            payload["temperature"] = 0.7
                            payload["max_tokens"] = 200
                        else:
                            payload["options"] = {
                                "temperature": 0.7,
                                "num_predict": 200,
                            }

                        response = requests.post(
                            url, headers=headers, json=payload, timeout=request_timeout
                        )
                        # Nếu endpoint không tồn tại, thử endpoint tiếp theo
                        if response.status_code == 404:
                            last_error = requests.HTTPError(
                                f"{response.status_code} {response.reason} for {url}"
                            )
                            logger.warning(
                                f"Ollama endpoint {url} not found (404), trying next endpoint..."
                            )
                            break

                        response.raise_for_status()
                        data = response.json()

                        # v1/chat/completions & api/chat đều trả choices[0].message.content
                        choices = data.get("choices")
                        if not choices:
                            raise ValueError(
                                f"Ollama {mode} response missing 'choices': {data}"
                            )
                        message = choices[0].get("message", {})
                        content = message.get("content", "").strip()
                        if not content:
                            raise ValueError(
                                f"Ollama {mode} response missing 'content': {data}"
                            )
                        return content

                    else:  # mode == "generate"
                        # API cũ /api/generate: nhận prompt duy nhất
                        prompt = f"{system_prompt}\n\n{user_prompt}"
                        payload = {
                            "model": self.ollama_model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.7,
                                "num_predict": 200,
                            },
                        }
                        response = requests.post(
                            url, headers=headers, json=payload, timeout=request_timeout
                        )
                        if response.status_code == 404:
                            last_error = requests.HTTPError(
                                f"{response.status_code} {response.reason} for {url}"
                            )
                            logger.warning(
                                f"Ollama endpoint {url} not found (404), no more endpoints to try."
                            )
                            break

                        response.raise_for_status()
                        data = response.json()
                        # /api/generate trả trường 'response'
                        text = data.get("response", "").strip()
                        if not text:
                            raise ValueError(
                                f"Ollama generate response missing 'response': {data}"
                            )
                        return text

                except (requests.exceptions.RequestException, ValueError) as e:
                    last_error = e
                    logger.warning(
                        f"Ollama query attempt {attempt + 1} with endpoint {url} failed: {e}"
                    )
                    time.sleep(min(timeout, 5.0))

            # Nếu thử nhiều lần mà không thành công với endpoint hiện tại, chuyển sang endpoint tiếp theo

        logger.error(
            f"Max retries exceeded across all Ollama endpoints. Last error: {last_error}"
        )
        return "Error: Failed to fetch response from Ollama."

    def _query_server(self, user_prompt, system_prompt, tries=10, timeout=20.0) -> str:
        """
        Queries the vLLM model endpoint with system and user prompts.
        Returns the generated text.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 200,  # Control response length
        }

        headers = {"Content-Type": "application/json"}
        if self.server_token:
            headers["Authorization"] = f"Bearer {self.server_token}"

        for attempt in range(tries):
            try:
                response = requests.post(
                    self.server_url, headers=headers, json=payload, timeout=timeout
                )
                response.raise_for_status()
                response_data = response.json()

                # Introduce a short delay to avoid rate limits
                time.sleep(2.0)

                # Return the generated response
                return response_data["choices"][0]["message"]["content"].strip()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Server query attempt {attempt + 1} failed: {e}")
                time.sleep(timeout)

        logger.error("Max retries exceeded: Unable to fetch response from server.")
        return "Error: Failed to fetch response from server."

    def _query_local(
        self,
        prompt,
        system_prompt,
        image_requested=False,
        scene=None,
        max_prompt_len=2500,
        clip_prompt=False,
        tries=3,
        timeout=5.0,
    ):
        """Uses the locally loaded model to generate responses."""
        for attempt in range(tries):
            try:
                if clip_prompt:
                    prompt = prompt[:max_prompt_len]

                if image_requested:
                    if scene is None or not hasattr(scene, "image_url"):
                        raise ValueError(
                            "Image requested but no scene or image_url provided."
                        )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": scene.image_url},
                                },
                            ],
                        },
                    ]
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]

                outputs = self.pipeline(
                    messages,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
                return outputs[0]["generated_text"][-1]["content"]
            except Exception as e:
                logger.warning(f"Local query attempt {attempt + 1} failed: {e}")
                time.sleep(timeout)

        logger.error(
            "Max retries exceeded: Unable to generate response from local model."
        )
        return "Error: Failed to generate response from local model."

    def _query_server_wot_system_prompt(
        self, user_prompt, tries=5, timeout=15.0
    ) -> str:
        """
        Queries the vLLM model endpoint with only the user prompt.
        Returns the generated text.
        """
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": 80,  # Control response length
        }

        headers = {"Content-Type": "application/json"}

        for attempt in range(tries):
            try:
                response = requests.post(
                    self.server_url, headers=headers, json=payload, timeout=timeout
                )
                response.raise_for_status()
                response_data = response.json()

                # Introduce a short delay to avoid rate limits
                time.sleep(2.0)

                # Return the generated response
                return response_data["choices"][0]["message"]["content"].strip()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Server query attempt {attempt + 1} failed: {e}")
                time.sleep(timeout)

        logger.error("Max retries exceeded: Unable to fetch response from server.")
        return "Error: Failed to fetch response from server."

    def query_model_with_ensembling(
        self,
        prompt,
        system_prompt,
        tries=3,
        timeout=5.0,
        image_requested=False,
        scene=None,
        max_prompt_len=2**14,
        clip_prompt=False,
        thread_id=1,
        shuffle_ensemble_count=3,  # Number of ensembles to create using choice shuffling
    ):
        for attempt in range(tries):
            if clip_prompt:
                prompt = prompt[:max_prompt_len]
            try:
                responses = []

                # Generate multiple responses using shuffled prompts
                for _ in range(shuffle_ensemble_count):
                    shuffled_prompt = self.shuffle_choices_in_prompt(prompt)
                    # messages = self.build_messages(system_prompt, shuffled_prompt, image_requested, scene)

                    # Use the pipeline to generate the response
                    # breakpoint()
                    outputs = self._query_server(
                        shuffled_prompt, system_prompt, tries, timeout
                    )  # self._query_server_wot_system_prompt(messages)
                    responses.append(outputs)

                # Aggregate responses (e.g., majority vote, longest consistent response, etc.)
                final_response = self.aggregate_responses(responses)
                return final_response

            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(timeout)
                continue
        raise Exception("Max retries exceeded: unable to generate response.")

    def shuffle_choices_in_prompt(self, prompt):
        # This function identifies choices (e.g., multiple-choice options) and shuffles them
        choices_pattern = r"\((a|b|c|d)\)\s+[^\n]+"
        choices = re.findall(choices_pattern, prompt)
        if choices:
            random.shuffle(choices)
            shuffled_prompt = re.sub(
                choices_pattern,
                lambda match: choices.pop(0),
                prompt,
                count=len(choices),
            )
            return shuffled_prompt
        return prompt

    def build_messages(self, system_prompt, prompt, image_requested, scene):
        if image_requested:
            if scene is None or not hasattr(scene, "image_url"):
                raise ValueError("Image requested but no scene or image_url provided.")
            return [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": scene.image_url}},
                    ],
                },
            ]
        else:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

    def aggregate_responses(self, responses):
        # Aggregate responses (e.g., by selecting the most common one)
        response_counts = {
            response: responses.count(response) for response in responses
        }
        return max(response_counts, key=response_counts.get)


fallback_agent = BAgent()


def query_model(
    model_str: str,
    prompt: str,
    system_prompt: str,
    tries: int = 1,
    timeout: float = 30.0,
    image_requested: bool = False,
    scene=None,
    max_prompt_len: int = 2**14,
    clip_prompt: bool = False,
):
    """
    Queries the specified language model with the given prompt and system prompt.
    Retries the query if an exception occurs.
    """
    # Initialize Groq client
    # client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    for _ in tqdm(range(tries), desc="Querying model"):
        # Optionally clip prompt length
        if clip_prompt:
            prompt = prompt[:max_prompt_len]

        try:
            answer = None

            # --- Handle image requests first ---
            if image_requested:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"{scene.image_url}"},
                            },
                        ],
                    },
                ]
                if model_str == "gpt4v":
                    response = openai.ChatCompletion.create(
                        model="gpt-4-vision-preview",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                elif model_str == "gpt-4o-mini":
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                elif model_str == "gpt4":
                    response = openai.ChatCompletion.create(
                        model="gpt-4-turbo",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                elif model_str == "gpt4o":
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                answer = response["choices"][0]["message"]["content"]

            # --- Handle text-only requests ---
            elif model_str in ["gpt4", "gpt4v", "gpt-4o-mini", "gpt4o", "gpt3.5"]:
                model_map = {
                    "gpt4": "gpt-4-turbo-preview",
                    "gpt4v": "gpt-4-vision-preview",
                    "gpt-4o-mini": "gpt-4o-mini",
                    "gpt4o": "gpt-4o",
                    "gpt3.5": "gpt-3.5-turbo",
                }
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                response = openai.ChatCompletion.create(
                    model=model_map[model_str],
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)

            elif model_str == "o1-preview":
                messages = [{"role": "user", "content": system_prompt + prompt}]
                response = openai.ChatCompletion.create(
                    model="o1-preview-2024-09-12",
                    messages=messages,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)

            elif model_str == "claude3.5sonnet":
                client_anthropic = anthropic.Anthropic(
                    api_key=os.environ["ANTHROPIC_API_KEY"]
                )
                message = client_anthropic.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=system_prompt,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = json.loads(message.to_json())["content"][0]["text"]

            elif model_str == "meta/llama-3.1-405b-instruct":
                from openai import OpenAI  # Assuming OpenAI integration for this model

                client_nvidia = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key="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",
                )
                response = client_nvidia.chat.completions.create(
                    model="meta/llama-3.1-405b-instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": "Write a limerick about the wonders of GPU computing.",
                        }
                    ],
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=1024,
                    stream=True,
                )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)

            elif model_str == "llama-2-70b-chat":
                output = replicate.run(
                    LLAMA2_URL,
                    input={
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "max_new_tokens": 200,
                    },
                )
                answer = "".join(output)
                answer = re.sub(r"\s+", " ", answer)

            elif model_str == "mixtral-8x7b":
                output = replicate.run(
                    MIXTRAL_URL,
                    input={
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "max_new_tokens": 75,
                    },
                )
                answer = "".join(output)
                answer = re.sub(r"\s+", " ", answer)

            elif model_str == "llama-3-70b-instruct":
                output = replicate.run(
                    LLAMA3_URL,
                    input={
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "max_new_tokens": 200,
                    },
                )
                answer = "".join(output)
                answer = re.sub(r"\s+", " ", answer)

            elif "GR_" in model_str:
                # For Groq-backed models, remove the prefix and use the Groq client.
                model = model_str.replace("GR_", "")
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=model,
                )
                answer = chat_completion.choices[0].message.content
                answer = re.sub(r"\s+", " ", answer)

            elif model_str.startswith("ollama:"):
                # For Ollama models, use the format "ollama:model_name"
                # e.g., "ollama:llama3.1", "ollama:mistral", "ollama:codellama"
                ollama_model = model_str.replace("ollama:", "")
                ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
                ollama_chat_url = f"{ollama_url}/api/chat"

                payload = {
                    "model": ollama_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"num_predict": 200, "temperature": 0.7},
                }
                headers = {"Content-Type": "application/json"}
                response = requests.post(
                    ollama_chat_url, headers=headers, json=payload, timeout=timeout
                )
                response.raise_for_status()
                response_data = response.json()
                answer = response_data["message"]["content"].strip()
                answer = re.sub(r"\s+", " ", answer)

            else:
                # Fallback to the baseline agent if none of the above match.
                answer = fallback_agent.query_model(prompt, system_prompt)

            return answer

        except Exception as e:
            time.sleep(timeout)
            continue


def import_generate():
    # Get the absolute path of the current script (agent.py)
    current_file = os.path.abspath(__file__)

    # Find the root directory (Medical_Project) dynamically
    root_dir = os.path.dirname(os.path.dirname(current_file))  # Move up two levels

    # Construct the path to MedPromptSimulate/src
    src_path = os.path.join(root_dir, "MedPromptSimulate", "src")

    # Add src to sys.path if not already present
    if src_path not in sys.path:
        sys.path.append(src_path)

    # Import generate.py as part of the package structure
    module = import_module("promptbase.mmlu.generate")

    # Return the function generate_single from the module
    return module.generate_single


def get_result_json():
    """
    Extracts and reads the JSON data from result.json.gz, then deletes the file.
    """
    # Get the absolute path of the current script (agent.py)
    current_file = os.path.abspath(__file__)

    # Find the root directory (Medical_Project) dynamically
    root_dir = os.path.dirname(os.path.dirname(current_file))  # Move up two levels

    # Construct the path to the result.json.gz file
    json_gz_path = os.path.join(
        root_dir,
        "MedPromptSimulate",
        "src",
        "promptbase",
        "generations",
        "expt",
        "train",
        "cot",
        "result.json.gz",
    )

    # Ensure the file exists
    if not os.path.exists(json_gz_path):
        raise FileNotFoundError(f"File not found: {json_gz_path}")

    with gzip.open(json_gz_path, "rt", encoding="utf-8") as gz_file:
        json_data = json.load(gz_file)  # Read JSON content

    return json_data


def get_diagnosis(backend, scenario_id):
    generate_single = import_generate()  # mmlu
    generate_single(backend, scenario_id, "temp_question.json")

    # Usage Example
    result_json = get_result_json()
    lst = [result_json[0]["expt"][key]["answer"] for key in result_json[0]["expt"]]

    counter = Counter(lst)
    most_common = counter.most_common(1)[0][0]  # Get the most common element

    dia = result_json[0]["answer_choices"][most_common]

    return dia


def extract_bracket_content(s: str):
    if "[" in s or "]" in s:
        return "[" + s.split("[", 1)[-1].split("]")[0] + "]"
    return s  # Return original string if brackets are not found


def clean_diagnosis(message):
    message = extract_bracket_content(message)

    message = message.replace("'", "")
    message = message.replace('"', "")
    message = message.replace("```python", "")
    message = message.replace("```", "")
    message = message.replace("\n", "")
    message = message.replace(", ", ",")
    formatted_str = message.strip()
    formatted_str = formatted_str.replace("[", '["')
    formatted_str = formatted_str.replace("]", '"]')
    if "., " in formatted_str or '.,"' in formatted_str:
        formatted_str = formatted_str.replace(".,", '.", "')
    else:
        formatted_str = formatted_str.replace(",", '", "')

    diagnosis_list = eval(formatted_str)
    diagnosis_list = [dia.strip() for dia in diagnosis_list]
    return diagnosis_list


def generate_possible_diagnoses(question: str, answer: str, backend: str):
    """
    Generates a Python list of 3 possible diagnoses based on the question while ensuring:
    - The generated diagnoses match the format of the correct diagnosis.
    - The correct diagnosis is excluded.
    - The diagnoses are unique.

    Parameters:
        question (str): The medical question.
        answer (str): The correct diagnosis.

    Returns:
        list: A list of 3 possible diagnoses.
    """
    prompt = (
        f"Given the following medical question, suggest three possible diagnoses in a format similar to the correct diagnosis. "
        f"Ensure that the diagnoses are unique, medically plausible, and formatted in a way that makes them indistinguishable from the correct answer. Do NOT suggest the correct diagnosis.\n\n"
        f"Question: {question}\n"
        f"Correct Diagnosis: {answer}\n\n"
        f"Provide the diagnoses in a Python list format."
    )

    system_prompt = "You are a highly knowledgeable and precise AI assistant specializing in medical reasoning. Your role is to analyze clinical scenarios and provide accurate, evidence-based differential diagnoses. You will assess the details of each case carefully and generate insightful responses that adhere to medical best practices."
    response = query_model(backend, prompt, system_prompt)

    # Extract the response as a list
    diagnoses = clean_diagnosis(response)  # Assuming response is in list format

    return diagnoses


def generate_possible_diagnoses_from_discussion(
    question: str, doctor_discussion: str, backend: str
):
    """
    Generates a Python list of 4 possible diagnoses based on the doctor discussion.
    This version does NOT use the ground truth answer - it extracts candidates purely
    from the doctors' opinions to avoid data leakage.

    Parameters:
        question (str): The medical question.
        doctor_discussion (str): The concatenated doctor discussion responses.
        backend (str): The LLM backend to use.

    Returns:
        list: A list of 4 possible diagnoses extracted from the discussion.
    """
    prompt = (
        f"Based on the following medical question and doctor discussion, extract the top 4 most likely diagnoses "
        f"that the doctors are considering. These should be the actual diagnoses mentioned or strongly implied "
        f"in the discussion. Ensure the diagnoses are unique and medically plausible.\n\n"
        f"Question: {question}\n\n"
        f"Doctor Discussion:\n{doctor_discussion}\n\n"
        f'Provide exactly 4 diagnoses in a Python list format, e.g., ["Diagnosis 1", "Diagnosis 2", "Diagnosis 3", "Diagnosis 4"].\n'
        f"Extract these from what the doctors are actually discussing - do not invent new diagnoses."
    )

    system_prompt = (
        "You are a highly knowledgeable and precise AI assistant specializing in medical reasoning. "
        "Your role is to extract and summarize the differential diagnoses being discussed by medical professionals. "
        "Focus on identifying the specific diagnoses mentioned in the discussion."
    )
    response = query_model(backend, prompt, system_prompt)

    # Extract the response as a list
    diagnoses = clean_diagnosis(response)

    # Ensure we have exactly 4 diagnoses
    if len(diagnoses) < 4:
        # If less than 4, pad with placeholder
        while len(diagnoses) < 4:
            diagnoses.append(f"Undetermined diagnosis {len(diagnoses) + 1}")
    elif len(diagnoses) > 4:
        diagnoses = diagnoses[:4]

    return diagnoses


def generate_answer_choices_from_candidates(candidate_diagnoses: list):
    """
    Creates answer choices from candidate diagnoses without knowing the correct answer.
    This is used to avoid data leakage during inference.

    Parameters:
        candidate_diagnoses (list): List of candidate diagnoses from doctor discussion.

    Returns:
        dict: Mapping of letter choices (A, B, C, D) to diagnoses.
    """
    letter_answers = ["A", "B", "C", "D"]

    # Ensure we have exactly 4 candidates
    candidates = (
        candidate_diagnoses[:4]
        if len(candidate_diagnoses) >= 4
        else candidate_diagnoses
    )
    while len(candidates) < 4:
        candidates.append(f"Other diagnosis {len(candidates) + 1}")

    random.shuffle(candidates)
    result = dict(zip(letter_answers, candidates))

    return result


def generate_answer_choices(correct_answer, answer_list):
    letter_answers = ["A", "B", "C", "D"]
    answer_list.append(correct_answer)
    random.shuffle(answer_list)  # Shuffle the list
    result = dict(zip(letter_answers, answer_list))  # Map keys to shuffled conditions
    answer_letter = letter_answers[answer_list.index(correct_answer)]
    return result, answer_letter


def generate_question_json(
    question: str,
    answer_choices: dict,
    correct_answer: str,
    filename: str = "temp_question.json",
):
    """
    Generates and saves a JSON file with a question, answer choices, and the correct answer.

    Parameters:
        question (str): The question text.
        answer_choices (dict): Dictionary of answer choices, e.g., {"A": "Answer1", "B": "Answer2"}.
        correct_answer (str): The correct answer key (e.g., "A").
        filename (str): The filename to save the JSON data. Default is "question.json".
    """
    data = [
        {
            "id": 1,
            "question": question,
            "answer_choices": answer_choices,
            "correct_answer": correct_answer,
        }
    ]

    # Get the absolute path of the current script (agent.py or test script in TOQ folder)
    current_file = os.path.abspath(__file__)

    # Find the root directory (Medical_Project) dynamically
    root_dir = os.path.dirname(os.path.dirname(current_file))  # Move up two levels

    # Construct the destination folder path
    save_folder = os.path.join(
        root_dir, "MedPromptSimulate", "src", "promptbase", "datasets", "mmlu", "train"
    )

    # Construct the full file path
    save_path = os.path.join(save_folder, filename)

    # Save to JSON file
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def generate_question_json_no_answer(
    question: str, answer_choices: dict, filename: str = "temp_question.json"
):
    """
    Generates and saves a JSON file with a question and answer choices WITHOUT the correct answer.
    This is used during inference to avoid data leakage - the correct answer is not known.

    Parameters:
        question (str): The question text.
        answer_choices (dict): Dictionary of answer choices, e.g., {"A": "Answer1", "B": "Answer2"}.
        filename (str): The filename to save the JSON data. Default is "temp_question.json".
    """
    data = [
        {
            "id": 1,
            "question": question,
            "answer_choices": answer_choices,
            "correct_answer": None,  # Unknown during inference
        }
    ]

    # Get the absolute path of the current script
    current_file = os.path.abspath(__file__)

    # Find the root directory dynamically
    root_dir = os.path.dirname(os.path.dirname(current_file))

    # Construct the destination folder path
    save_folder = os.path.join(
        root_dir, "MedPromptSimulate", "src", "promptbase", "datasets", "mmlu", "train"
    )

    # Construct the full file path
    save_path = os.path.join(save_folder, filename)

    # Save to JSON file
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def extract_question(patient_statement, agent_hist, parsed_responses, backend):
    # Prompt for the model
    prompt = (
        f"Given the following patient statement, additional context, and doctor discussion, "
        f"generate a structured diagnostic question that extracts key insights and relevant test results:\n\n"
        f"Patient Statement:\n{patient_statement}\n\n"
        f"Additional Context:\n{agent_hist}\n\n"
        f"Doctor Discussion:\n{parsed_responses}\n\n"
        f"Ensure the question is formatted in the following structured manner using as much information from the provided context above:\n"
        f"Provide the most likely final diagnosis for the following patient. A ___ year old [man/woman] "
        f"presents with [duration] of [symptom description], associated with [other symptoms]. "
        f"The symptoms worsened with [trigger], improved with [relief], but recurred. "
        f"Laboratory results show [lab findings]. Imaging revealed [imaging findings]. "
        f"What is the final diagnosis for this patient?"
    )

    system_prompt = "You are a highly knowledgeable and precise AI assistant specializing in medical reasoning. Your role is to analyze clinical scenarios and provide accurate, evidence-based differential diagnoses. You will assess the details of each case carefully and generate insightful responses that adhere to medical best practices."

    response = query_model(backend, prompt, system_prompt)  # maybe do some cleaning too

    return response

    raise Exception("Max retries exceeded: timeout")


# Example Usage
if __name__ == "__main__":
    agent = BAgent()
    response = agent.query_model("Hello! How are you?")
    print("Response:", response)
