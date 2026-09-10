import os
from typing import Any

from dotenv import load_dotenv
from langfuse import get_client
from openai import OpenAI

from insurance_rag.utils.config import load_config


load_dotenv()


class InsuranceGenerator:

    def __init__(self):

        self.config = load_config()

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.langfuse = get_client()

    def _get_system_prompt(self):

        prompt = self.langfuse.get_prompt(
            name=self.config.generation.prompt_name,
            label=self.config.generation.prompt_label,
        )

        return prompt.compile()

    def generate(
        self,
        question: str,
        contexts: list[str],
    ) -> dict[str, Any]:

        system_prompt = self._get_system_prompt()

        context_text = "\n\n".join(contexts)

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Policy Context:\n\n"
                    f"{context_text}\n\n"
                    f"Question:\n\n"
                    f"{question}"
                ),
            },
        ]

        response = self.client.chat.completions.create(
            model=self.config.generation.model,
            temperature=self.config.generation.temperature,
            messages=messages,
        )

        usage = response.usage

        input_tokens = (
            usage.prompt_tokens
            if usage
            else 0
        )

        output_tokens = (
            usage.completion_tokens
            if usage
            else 0
        )

        total_tokens = (
            usage.total_tokens
            if usage
            else 0
        )

        return {
            "answer":
                response.choices[0].message.content,

            "model":
                self.config.generation.model,

            "input_tokens":
                input_tokens,

            "output_tokens":
                output_tokens,

            "total_tokens":
                total_tokens,
        }
