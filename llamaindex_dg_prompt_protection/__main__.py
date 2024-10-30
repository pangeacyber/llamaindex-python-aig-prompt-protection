from __future__ import annotations

from typing import Any, override

import click
from llama_index.core.llms import MessageRole
from llama_index.llms.openai import OpenAI  # type: ignore[import-untyped]
from pangea import PangeaConfig
from pangea.services import AIGuard, PromptGuard
from pangea.services.prompt_guard import Message
from pydantic import SecretStr
from pydantic_core import to_json


class SecretStrParamType(click.ParamType):
    name = "secret"

    @override
    def convert(self, value: Any, param: click.Parameter | None = None, ctx: click.Context | None = None) -> SecretStr:
        if isinstance(value, SecretStr):
            return value

        return SecretStr(value)


SECRET_STR = SecretStrParamType()


@click.command()
@click.option(
    "--ai-guard-token",
    envvar="PANGEA_AI_GUARD_TOKEN",
    type=SECRET_STR,
    required=True,
    help="Pangea AI Guard API token. May also be set via the `PANGEA_AI_GUARD_TOKEN` environment variable.",
)
@click.option(
    "--prompt-guard-token",
    envvar="PANGEA_PROMPT_GUARD_TOKEN",
    type=SECRET_STR,
    required=True,
    help="Pangea Prompt Guard API token. May also be set via the `PANGEA_PROMPT_GUARD_TOKEN` environment variable.",
)
@click.option(
    "--pangea-domain",
    envvar="PANGEA_DOMAIN",
    default="aws.us.pangea.cloud",
    show_default=True,
    required=True,
    help="Pangea API domain. May also be set via the `PANGEA_DOMAIN` environment variable.",
)
@click.option(
    "--openai-api-key",
    envvar="OPENAI_API_KEY",
    type=SECRET_STR,
    required=True,
    help="OpenAI API key. May also be set via the `OPENAI_API_KEY` environment variable.",
)
@click.option("--model", default="gpt-4o-mini", show_default=True, required=True, help="OpenAI model.")
@click.argument("prompt")
def main(
    *,
    prompt: str,
    ai_guard_token: SecretStr,
    prompt_guard_token: SecretStr,
    pangea_domain: str,
    openai_api_key: SecretStr,
    model: str,
) -> None:
    llm = OpenAI(model=model, api_key=openai_api_key)

    # Initialize the ai guard and prompt guard
    ai_guard = AIGuard(token=ai_guard_token.get_secret_value(), config=PangeaConfig(domain=pangea_domain))
    prompt_guard = PromptGuard(token=prompt_guard_token.get_secret_value(), config=PangeaConfig(domain=pangea_domain))

    # Apply ai guard to the prompt
    ai_guard_response = ai_guard.guard_text(prompt)
    assert ai_guard_response.result

    # If the prompt was redacted, use the redacted prompt
    guarded_prompt = (
        ai_guard_response.result.redacted_prompt if ai_guard_response.result.redacted_prompt else prompt
    )

    # Construct chat messages from guarded prompt
    messages = [Message(role=MessageRole.USER, content=guarded_prompt)]

    # Apply prompt guard to the prompt
    prompt_guard_response = prompt_guard.guard(messages)
    assert prompt_guard_response.result

    # If injection was detected, raise an exception
    if prompt_guard_response.result.detected:
        raise Exception(to_json(prompt_guard_response.result).decode("utf-8"))
    else:
        click.echo(llm.chat(messages))


if __name__ == "__main__":
    main()
