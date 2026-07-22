import json
import unittest
from unittest.mock import patch

from services.assistant_service import LLMNotConfiguredError, answer_user_message
from services.knowledge_service import retrieve_knowledge
from services.llm_client import OpenAICompatibleClient


class FakeClient:
    configured = True

    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None, tool_choice=None):
        self.calls.append((messages, tools, tool_choice))
        if tools:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "estimate_vehicle",
                            "arguments": json.dumps(
                                {
                                    "brand": "大众",
                                    "model": "帕萨特",
                                    "city": "广州",
                                    "mileage": 6.5,
                                    "year": 2018,
                                    "month": 6,
                                    "gearbox": "自动",
                                    "emission": "国六",
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "根据估值工具和资料，这是一个实验性参考结果。"}


class KnowledgeTests(unittest.TestCase):
    def test_retrieval_returns_auditable_source_for_matching_terms(self):
        records = retrieve_knowledge("国六 排放标准", limit=2)

        self.assertTrue(records)
        self.assertTrue(records[0]["source_id"])
        self.assertIn("国六", records[0]["content"])


class ClientTests(unittest.TestCase):
    def test_unconfigured_client_is_explicitly_disabled(self):
        client = OpenAICompatibleClient(base_url="", api_key="", model="")

        self.assertFalse(client.configured)
        with self.assertRaises(LLMNotConfiguredError):
            client.chat([])


class AssistantTests(unittest.TestCase):
    @patch("services.assistant_service.call_model_api")
    def test_tool_call_executes_estimator_and_returns_citations(self, predict_mock):
        predict_mock.return_value = {
            "price": 5.57,
            "range": {"low": 5.12, "high": 6.02},
            "confidence": None,
            "model_status": "experimental",
            "metrics": {
                "mse": 177.17,
                "rmse": 13.31,
                "mae": 11.5,
                "r2": -0.015,
                "acc_10": 0.117,
            },
            "comment": "实验模型结果。",
        }
        client = FakeClient()

        result = answer_user_message("帮我估算一辆2018年广州帕萨特", client=client)

        self.assertEqual(result.llm_status, "configured")
        self.assertIsNotNone(result.estimate)
        self.assertTrue(result.citations)
        predict_mock.assert_called_once()
        self.assertGreaterEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
