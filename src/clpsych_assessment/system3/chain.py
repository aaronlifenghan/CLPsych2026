from pathlib import Path
from typing import Literal, Optional, Type

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue


def get_model(
    provider: str = "ollama",
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    format: JsonSchemaValue | Literal["", "json"] = "",
    options: dict = {
        "num_ctx": 32768,
        "temperature": 0.3,
        "top_k": 40,
        "top_p": 0.9,
    },
):
    if provider == "ollama":
        return ChatOllama(
            base_url=base_url or "http://localhost:11434",
            model=model_name or "llama3.1",
            format=format,
            num_ctx=options["num_ctx"],
            temperature=options["temperature"],
            top_k=options["top_k"],
            top_p=options["top_p"],
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model_name or "gpt-4-turbo-preview"
        ).with_structured_output(schema=format, method="json_schema")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def create_chain(
    model,
    response_model: Type[BaseModel],
    prompt_name: str,
):
    """Create an LLM chain for any task with structured output."""
    parser = PydanticOutputParser(pydantic_object=response_model)

    template_path = (
        Path(__file__).parent / "prompts" / f"{prompt_name}.md"
    )

    if not template_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {template_path}"
        )

    template = template_path.read_text(encoding="utf-8")

    prompt = ChatPromptTemplate.from_template(template).partial(
        format_instructions=parser.get_format_instructions()
    )
    return prompt | model | parser
