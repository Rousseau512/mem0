import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.openai import OpenAIConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json
from mem0.utils.codex_oauth import load_codex_oauth_credentials, resolve_codex_base_url, should_use_codex_oauth


class OpenAILLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, OpenAIConfig, Dict]] = None):
        # Convert to OpenAIConfig if needed
        if config is None:
            config = OpenAIConfig()
        elif isinstance(config, dict):
            config = OpenAIConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, OpenAIConfig):
            # Convert BaseLlmConfig to OpenAIConfig
            config = OpenAIConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                reasoning_effort=getattr(config, 'reasoning_effort', None),
                http_client_proxies=config.http_client_proxies,
                is_reasoning_model=getattr(config, 'is_reasoning_model', None),
                use_codex_oauth=getattr(config, 'use_codex_oauth', None),
                codex_auth_file=getattr(config, 'codex_auth_file', None),
                codex_base_url=getattr(config, 'codex_base_url', None),
            )

        super().__init__(config)

        model_was_defaulted = not self.config.model
        if model_was_defaulted:
            self.config.model = "gpt-5-mini"

        self._use_codex_oauth = False

        if os.environ.get("OPENROUTER_API_KEY"):  # Use OpenRouter
            self.client = OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=self.config.openrouter_base_url
                or os.getenv("OPENROUTER_API_BASE")
                or "https://openrouter.ai/api/v1",
            )
        else:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            standard_base_url = (
                self.config.openai_base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            )
            base_url = standard_base_url
            default_headers = None
            if should_use_codex_oauth(
                api_key,
                standard_base_url,
                self.config.use_codex_oauth,
                self.config.codex_auth_file,
            ):
                api_key, default_headers = load_codex_oauth_credentials(self.config.codex_auth_file)
                base_url = resolve_codex_base_url(self.config.codex_base_url or os.getenv("OPENAI_CODEX_BASE_URL"))
                if model_was_defaulted:
                    self.config.model = "gpt-5.5"
                self._use_codex_oauth = True

            self.client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    @staticmethod
    def _get_response_field(item: Any, field: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(field, default)
        return getattr(item, field, default)

    def _split_responses_messages(self, messages: List[Dict[str, str]]) -> tuple[str, List[Dict[str, Any]]]:
        instructions = []
        input_messages = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            text = content if isinstance(content, str) else json.dumps(content)

            if role in {"system", "developer"}:
                instructions.append(text)
                continue

            responses_role = "assistant" if role == "assistant" else "user"
            content_type = "output_text" if responses_role == "assistant" else "input_text"
            input_messages.append({"role": responses_role, "content": [{"type": content_type, "text": text}]})

        return "\n\n".join(instructions) or "You are a helpful assistant.", input_messages

    def _convert_tools_for_responses(self, tools: Optional[List[Dict]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None

        responses_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                function = tool["function"]
                converted = {
                    "type": "function",
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                }
                if "strict" in function:
                    converted["strict"] = function["strict"]
                responses_tools.append(converted)
            else:
                responses_tools.append(tool)

        return responses_tools

    @staticmethod
    def _requires_json_object_response(response_format: Any) -> bool:
        return isinstance(response_format, dict) and response_format.get("type") == "json_object"

    def _ensure_responses_json_hint(self, input_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure JSON mode validation sees a JSON hint in Responses input.

        The ChatGPT/Codex Responses backend validates the ``json_object`` response format against the ``input``
        messages, not the separate ``instructions`` field. Memory extraction prompts put output guidance in system
        instructions, so add a tiny input hint when needed.
        """

        serialized_input = json.dumps(input_messages).lower()
        if "json" in serialized_input:
            return input_messages

        return [
            *input_messages,
            {"role": "user", "content": [{"type": "input_text", "text": "Respond with valid JSON."}]},
        ]

    def _collect_responses_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text is not None:
            return output_text

        texts = []
        for item in getattr(response, "output", []) or []:
            item_type = self._get_response_field(item, "type")
            if item_type == "message":
                for content in self._get_response_field(item, "content", []) or []:
                    content_type = self._get_response_field(content, "type")
                    if content_type in {"output_text", "text"}:
                        text = self._get_response_field(content, "text", "")
                        if text:
                            texts.append(text)
            elif item_type in {"output_text", "text"}:
                text = self._get_response_field(item, "text", "")
                if text:
                    texts.append(text)

        return "\n".join(texts)

    def _parse_responses_response(self, response: Any, tools: Optional[List[Dict]]) -> Union[str, Dict[str, Any]]:
        content = self._collect_responses_text(response)
        if not tools:
            return content

        processed_response = {"content": content, "tool_calls": []}
        for item in getattr(response, "output", []) or []:
            if self._get_response_field(item, "type") != "function_call":
                continue
            arguments = self._get_response_field(item, "arguments", "{}")
            parsed_arguments = json.loads(extract_json(arguments)) if isinstance(arguments, str) else arguments
            processed_response["tool_calls"].append(
                {
                    "name": self._get_response_field(item, "name"),
                    "arguments": parsed_arguments,
                }
            )
        return processed_response

    def _parse_responses_stream(
        self, stream: Any, tools: Optional[List[Dict]]
    ) -> tuple[Union[str, Dict[str, Any]], Any]:
        """Parse a streaming Responses API result.

        The ChatGPT/Codex subscription backend currently requires streaming requests. Prefer the final
        ``response.completed`` payload when the SDK exposes it, but also collect deltas so lightweight mocks and
        partial streams remain parseable.
        """

        text_parts: List[str] = []
        final_response = None
        tool_calls: Dict[str, Dict[str, Any]] = {}

        for event in stream:
            event_type = self._get_response_field(event, "type", "")

            if event_type == "response.completed":
                final_response = self._get_response_field(event, "response")
                continue

            if event_type == "response.output_text.delta":
                delta = self._get_response_field(event, "delta", "")
                if delta:
                    text_parts.append(delta)
                continue

            if event_type == "response.output_text.done" and not text_parts:
                text = self._get_response_field(event, "text", "")
                if text:
                    text_parts.append(text)
                continue

            if event_type in {"response.function_call_arguments.delta", "response.function_call_arguments.done"}:
                call_id = self._get_response_field(event, "call_id")
                item_id = self._get_response_field(event, "item_id")
                output_index = self._get_response_field(event, "output_index")
                key = str(call_id or item_id or output_index or len(tool_calls))
                call = tool_calls.setdefault(key, {"name": self._get_response_field(event, "name"), "arguments": ""})
                if self._get_response_field(event, "name"):
                    call["name"] = self._get_response_field(event, "name")
                if event_type.endswith(".delta"):
                    call["arguments"] += self._get_response_field(event, "delta", "") or ""
                else:
                    call["arguments"] = self._get_response_field(event, "arguments", call["arguments"] or "{}")
                continue

            if event_type == "response.output_item.done":
                item = self._get_response_field(event, "item")
                if self._get_response_field(item, "type") == "function_call":
                    key = str(
                        self._get_response_field(item, "call_id")
                        or self._get_response_field(item, "id")
                        or len(tool_calls)
                    )
                    tool_calls[key] = {
                        "name": self._get_response_field(item, "name"),
                        "arguments": self._get_response_field(item, "arguments", "{}"),
                    }

        if final_response is not None:
            parsed = self._parse_responses_response(final_response, tools)
            if tools or parsed:
                return parsed, final_response

        content = "".join(text_parts)
        if not tools:
            return content, final_response

        processed_response = {"content": content, "tool_calls": []}
        for call in tool_calls.values():
            arguments = call.get("arguments") or "{}"
            parsed_arguments = json.loads(extract_json(arguments)) if isinstance(arguments, str) else arguments
            processed_response["tool_calls"].append({"name": call.get("name"), "arguments": parsed_arguments})
        return processed_response, final_response

    def _generate_codex_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        if not hasattr(self.client, "responses"):
            raise RuntimeError("Codex OAuth requires an OpenAI Python SDK version with Responses API support")

        instructions, input_messages = self._split_responses_messages(messages)
        text_config = {"verbosity": "low"}
        if response_format:
            text_config["format"] = response_format
            if self._requires_json_object_response(response_format):
                input_messages = self._ensure_responses_json_hint(input_messages)

        params = {
            "model": self.config.model,
            "instructions": instructions,
            "input": input_messages,
            "store": False,
            "text": text_config,
            "parallel_tool_calls": True,
        }

        # The ChatGPT/Codex Responses backend currently rejects max_output_tokens even though the public Responses API
        # accepts it. Omit the cap and let the backend apply its model defaults.
        if self.config.reasoning_effort:
            params["reasoning"] = {"effort": self.config.reasoning_effort, "summary": "auto"}

        responses_tools = self._convert_tools_for_responses(tools)
        if responses_tools:
            params["tools"] = responses_tools
            params["tool_choice"] = tool_choice

        for key in ("temperature", "top_p", "service_tier"):
            if key in kwargs:
                params[key] = kwargs[key]

        params["stream"] = True
        response = self.client.responses.create(**params)
        parsed_response, final_response = self._parse_responses_stream(response, tools)
        callback_response = final_response if final_response is not None else response
        if self.config.response_callback:
            try:
                self.config.response_callback(self, callback_response, params)
            except Exception as e:
                logging.error(f"Error due to callback: {e}")
                pass
        return parsed_response

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a JSON response based on the given messages using OpenAI.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional OpenAI-specific parameters.

        Returns:
            json: The generated response.
        """
        if self._use_codex_oauth:
            return self._generate_codex_response(
                messages=messages,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )

        params = self._get_supported_params(messages=messages, **kwargs)
        
        params.update({
            "model": self.config.model,
            "messages": messages,
        })

        if os.getenv("OPENROUTER_API_KEY"):
            openrouter_params = {}
            if self.config.models:
                openrouter_params["models"] = self.config.models
                openrouter_params["route"] = self.config.route
                params.pop("model")

            if self.config.site_url and self.config.app_name:
                extra_headers = {
                    "HTTP-Referer": self.config.site_url,
                    "X-Title": self.config.app_name,
                }
                openrouter_params["extra_headers"] = extra_headers

            params.update(**openrouter_params)
        
        else:
            # Only send OpenAI-specific parameters when the user has explicitly
            # configured them. OpenAI-compatible backends (Gemini, Groq, vLLM, etc.)
            # reject unknown fields, so `store` must be opt-in, not opt-out.
            if self.config.store is not None:
                params["store"] = self.config.store

        if response_format:
            params["response_format"] = response_format
        if tools:  # TODO: Remove tools if no issues found with new memory addition logic
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        response = self.client.chat.completions.create(**params)
        parsed_response = self._parse_response(response, tools)
        if self.config.response_callback:
            try:
                self.config.response_callback(self, response, params)
            except Exception as e:
                # Log error but don't propagate
                logging.error(f"Error due to callback: {e}")
                pass
        return parsed_response
