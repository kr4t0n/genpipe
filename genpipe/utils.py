from retry import retry
from openai import OpenAI


@retry(tries=4)
def call(
    llm: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    thinking: bool = False,
    budget_tokens: int = 4 * 1000,
    max_tokens: int = 64 * 1000,
) -> tuple:
    if thinking:
        # we need to construct the extra body for the thinking feature
        if "qwen" in model:
            extra_body = {"enable_thinking": True}
            reasoning_response = "reasoning"
        elif "claude" in model:
            extra_body = {"thinking": {"type": "enabled", "budget_tokens": budget_tokens}}
            reasoning_response = "reasoning_content"
        else:
            raise ValueError(f"thinking feature is not supported for {model} model.")
    else:
        extra_body = {}
        reasoning_response = ""

    completion = llm.chat.completions.create(
        model=model,
        messages=messages,
        extra_body=extra_body,
        max_tokens=max_tokens,
    )

    response = completion.choices[0].message
    content = response.content
    reasoning = getattr(response, reasoning_response, None)

    return content, reasoning
