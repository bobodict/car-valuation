import json
from typing import Callable

from schemas import AssistantResponse, PredictRequest, PredictResponse
from services.knowledge_service import retrieve_knowledge
from services.llm_client import (
    LLMClientError,
    LLMNotConfiguredError,
    OpenAICompatibleClient,
)
from services.model_service import ModelServiceError, call_model_api


ESTIMATE_VEHICLE_TOOL = {
    "type": "function",
    "function": {
        "name": "estimate_vehicle",
        "description": (
            "Call the used-car valuation model with structured vehicle parameters. "
            "Mileage is measured in km and the predicted price is measured in INR. "
            "Use source-compatible category values when known."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "brand": {"type": "string", "description": "vehicle brand"},
                "model": {"type": "string", "description": "vehicle model"},
                "city": {"type": "string", "description": "listing city"},
                "mileage": {"type": "number", "minimum": 0, "description": "driven distance in km"},
                "year": {"type": "integer", "description": "registration year"},
                "month": {"type": "integer", "minimum": 1, "maximum": 12},
                "gearbox": {"type": "string", "enum": ["Automatic", "Manual", "unknown"]},
                "emission": {"type": "string"},
                "fuel_type": {"type": "string"},
                "displacement": {"type": "number", "minimum": 0, "description": "engine liters"},
                "seats": {"type": "integer", "minimum": 1},
                "owner_count": {"type": "integer", "minimum": 1},
                "vehicle_type": {"type": "string"},
                "color": {"type": "string"},
                "accident_history": {"type": "string"},
            },
            "required": [
                "brand",
                "city",
                "mileage",
                "year",
                "month",
                "gearbox",
                "emission",
                "fuel_type",
                "displacement",
                "seats",
                "owner_count",
                "vehicle_type",
                "color",
                "accident_history",
            ],
        },
    },
}


SYSTEM_PROMPT = (
    "You are a used-car valuation research assistant. Answer knowledge questions only from "
    "the supplied sources. For price requests, you must call estimate_vehicle and must not "
    "invent a price. Explain that the model is experimental when the quality gate fails, "
    "cite source ids in square brackets, and never claim an uncalibrated confidence interval."
)


def _knowledge_context(records: list[dict]) -> str:
    return "\n\n".join(
        f"[{record['source_id']}] {record['title']}\n{record['content']}"
        for record in records
    )


def answer_user_message(
    message: str,
    client: OpenAICompatibleClient | None = None,
    predictor: Callable[[PredictRequest], dict] | None = None,
) -> AssistantResponse:
    client = client or OpenAICompatibleClient()
    predictor = predictor or call_model_api
    records = retrieve_knowledge(message, limit=3)
    citations = [{"source_id": record["source_id"], "title": record["title"]} for record in records]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Available sources:\n{_knowledge_context(records)}"},
        {"role": "user", "content": message},
    ]

    first_message = client.chat(messages, tools=[ESTIMATE_VEHICLE_TOOL], tool_choice="auto")
    estimate = None
    tool_calls = first_message.get("tool_calls") or []
    if tool_calls:
        tool_call = tool_calls[0]
        function = tool_call.get("function") or {}
        if function.get("name") != "estimate_vehicle":
            raise LLMClientError("model requested an unsupported tool")
        try:
            request = PredictRequest.model_validate(json.loads(function["arguments"]))
            tool_result = predictor(request)
            estimate = PredictResponse.model_validate(tool_result)
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMClientError("model vehicle parameters failed backend validation") from exc
        except ModelServiceError as exc:
            raise LLMClientError("valuation tool is temporarily unavailable") from exc

        messages.append(
            {
                "role": "assistant",
                "content": first_message.get("content"),
                "tool_calls": [tool_call],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.get("id", "estimate_vehicle"),
                "name": "estimate_vehicle",
                "content": json.dumps(tool_result, ensure_ascii=False),
            }
        )
        final_message = client.chat(messages)
    else:
        final_message = first_message

    answer = (final_message.get("content") or "").strip()
    if not answer:
        raise LLMClientError("model returned no displayable answer")

    return AssistantResponse(
        answer=answer,
        citations=citations,
        estimate=estimate,
        llm_status="configured",
    )
