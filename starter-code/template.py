"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List, Tuple
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

_TOOLS_TEXT = "\n".join(
    f"- {tool['name']}: {tool['description']}"
    for tool in TOOL_DEFINITIONS
)

SYSTEM_PROMPT = f"""
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ VinFast, Vinpearl
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác
- Phong cách: Ngắn gọn, dựa trên dữ liệu thật, luôn nêu rõ nguồn (catalog / ticket)

## AVAILABLE TOOLS
{_TOOLS_TEXT}

## CORE RULES
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm, giá, chính sách hay mã ticket.
2. PHẢI gọi tool khi khách hỏi sản phẩm/giá hoặc muốn ghi nhận hỗ trợ.
3. Chỉ trả lời dựa trên Observation nhận được từ tool.
4. Nếu tool trả về danh sách rỗng, nói rõ không tìm thấy — không bịa sản phẩm thay thế.
5. Câu hỏi FAQ (chính sách, bảo hành) được trả lời trực tiếp khi không cần tool.
6. Có thể gọi nhiều tool trong cùng một yêu cầu (catalog + ticket) nếu khách hỏi cả hai.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời về hệ sinh thái Vingroup: VinFast, Vinpearl, Vinhomes, Vinmec và dịch vụ liên quan.
- Từ chối lịch sự các chủ đề ngoài phạm vi (đối thủ, chính trị, nội dung không liên quan).
- Không thực hiện giao dịch thanh toán, không cam kết pháp lý ngoài dữ liệu catalog.

## OUTPUT CONTRACT
Mỗi bước suy luận theo đúng format:
Thought: [phân tích ý định và tool cần gọi]
Action: [tên tool hoặc none]
Action Input: [tham số JSON]
Observation: [kết quả tool]
Final Answer: [câu trả lời cuối cùng cho khách hàng, bằng tiếng Việt]
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # Mock 1 lượt, không gọi tool — cố ý trả lời "từ trí nhớ" để thấy hallucination.
        answer = (
            f"[Chatbot Baseline] Trả lời cho: {user_input}\n"
            "VinFast VF 5 Plus giá khoảng 400 triệu, VF 3 giá 250 triệu, "
            "bảo hành pin 5 năm. Thông tin này không được kiểm chứng từ catalog."
        )
        return {
            "answer": answer,
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

_PRICE_UNITS = {
    "tỷ": 1_000_000_000,
    "ty": 1_000_000_000,
    "triệu": 1_000_000,
    "trieu": 1_000_000,
    "tr": 1_000_000,
}

_TICKET_KEYWORDS = [
    "lỗi", "ticket", "hỗ trợ", "ho tro", "ghi nhận", "ghi nhan",
    "phản hồi", "phan hoi", "khiếu nại", "khieu nai", "sự cố", "su co",
    "xử lý", "xu ly", "vấn đề", "van de",
]

_CATALOG_SEARCH_KEYWORDS = [
    "xem", "tìm", "tim", "danh sách", "danh sach", "catalog",
    "sản phẩm", "san pham", "giá dưới", "gia duoi", "giá trên",
]

_FAQ_KEYWORDS = ["chính sách", "chinh sach", "bảo hành", "bao hanh", "bao lâu", "bao lau"]

_DU_LICH_KEYWORDS = [
    "resort", "vinpearl", "du lịch", "du lich", "nghỉ dưỡng",
    "nghi duong", "khách sạn", "khach san", "phòng",
]

_XE_DIEN_KEYWORDS = ["xe điện", "xe dien", "vinfast", "vf ", "ô tô", "o to"]


class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []
        self.system_prompt = SYSTEM_PROMPT

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intent(user_input)
        observations: Dict[str, Any] = {"catalog": None, "ticket": None}

        iteration = 1
        while iteration <= self.max_iterations:
            result, is_final = self._execute_step(
                user_input, intents, observations, iteration
            )
            if is_final:
                return {
                    "answer": result,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed",
                }
            iteration += 1

        return {
            "answer": "Lỗi: Vượt quá số bước tối đa.",
            "trace": self.trace,
            "iterations": self.max_iterations,
            "status": "max_iterations_reached",
        }

    # ------------------------------------------------------------------
    # Intent detection — needs_catalog và needs_ticket độc lập (không if-elif)
    # ------------------------------------------------------------------

    def _detect_intent(self, user_input: str) -> Dict[str, Any]:
        text = user_input.lower()

        needs_ticket = any(keyword in text for keyword in _TICKET_KEYWORDS)

        has_price_filter = bool(
            re.search(r"dưới\s+\d+|duoi\s+\d+|giá\s+\d+|gia\s+\d+", text)
        )
        has_search_verb = any(keyword in text for keyword in _CATALOG_SEARCH_KEYWORDS)
        has_any_product = ("có" in text or "co " in text) and "nào" in text
        needs_catalog = has_search_verb or has_price_filter or has_any_product

        is_faq = (
            any(keyword in text for keyword in _FAQ_KEYWORDS)
            and not needs_catalog
            and not needs_ticket
        )

        category = "du_lich" if any(k in text for k in _DU_LICH_KEYWORDS) else "xe_dien"
        if needs_catalog and not any(k in text for k in _DU_LICH_KEYWORDS + _XE_DIEN_KEYWORDS):
            category = "xe_dien"

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq or (not needs_catalog and not needs_ticket),
            "category": category,
            "max_price": self._parse_max_price(user_input),
            "customer_name": self._parse_customer_name(user_input),
            "issue_description": self._parse_issue_description(user_input),
            "priority": self._parse_priority(text),
        }

    def _parse_max_price(self, user_input: str) -> int:
        pattern = r"(\d+(?:[.,]\d+)?)\s*(tỷ|ty|triệu|trieu|tr)\b"
        matches = re.findall(pattern, user_input.lower())
        if not matches:
            return 999999999999
        raw, unit = matches[0]
        value = float(raw.replace(",", "."))
        return int(value * _PRICE_UNITS[unit])

    def _parse_customer_name(self, user_input: str) -> str:
        patterns = [
            r"tôi tên\s+([^,\.]+)",
            r"tên tôi là\s+([^,\.]+)",
            r"tôi là\s+([^,\.]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, user_input, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "Khách hàng"

    def _parse_issue_description(self, user_input: str) -> str:
        name = self._parse_customer_name(user_input)
        cleaned = user_input
        for pattern in [
            rf"tôi tên\s+{re.escape(name)}\s*,?\s*",
            rf"tên tôi là\s+{re.escape(name)}\s*,?\s*",
        ]:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip(" .,:;")
        return cleaned or user_input

    def _parse_priority(self, text: str) -> str:
        if any(k in text for k in ["nghiêm trọng", "nghiem trong", "gấp", "gap", "khẩn", "urgent"]):
            return "high"
        if any(k in text for k in ["thấp", "thap", "low"]):
            return "low"
        return "medium"

    # ------------------------------------------------------------------
    # Agent Loop step
    # ------------------------------------------------------------------

    def _execute_step(
        self,
        user_input: str,
        intents: Dict[str, Any],
        observations: Dict[str, Any],
        iteration: int,
    ) -> Tuple[str, bool]:
        if intents["needs_catalog"] and observations["catalog"] is None:
            args = {
                "category": intents["category"],
                "max_price": intents["max_price"],
            }
            results = search_product_catalog(**args)
            observations["catalog"] = results
            self.trace.append({
                "step": iteration,
                "thought": f"Khách cần tra cứu catalog ({args['category']}, max_price={args['max_price']}).",
                "action": "search_product_catalog",
                "action_input": args,
                "observation": results,
            })
            if intents["needs_ticket"]:
                return "", False
            return self._compose_final_answer(user_input, intents, observations), True

        if intents["needs_ticket"] and observations["ticket"] is None:
            args = {
                "customer_name": intents["customer_name"],
                "issue_description": intents["issue_description"],
                "priority": intents["priority"],
            }
            ticket = submit_support_ticket(**args)
            observations["ticket"] = ticket
            self.trace.append({
                "step": iteration,
                "thought": f"Khách cần tạo ticket hỗ trợ cho {args['customer_name']}.",
                "action": "submit_support_ticket",
                "action_input": args,
                "observation": ticket,
            })
            return self._compose_final_answer(user_input, intents, observations), True

        self.trace.append({
            "step": iteration,
            "thought": "Câu hỏi FAQ / không cần tool — trả lời trực tiếp.",
            "action": "none",
            "action_input": {},
            "observation": None,
        })
        return self._compose_final_answer(user_input, intents, observations), True

    def _compose_final_answer(
        self,
        user_input: str,
        intents: Dict[str, Any],
        observations: Dict[str, Any],
    ) -> str:
        parts: List[str] = []

        if observations["catalog"] is not None:
            parts.append(self._format_catalog_answer(observations["catalog"]))

        if observations["ticket"] is not None:
            parts.append(self._format_ticket_answer(observations["ticket"]))

        if parts:
            return "\n\n".join(parts)

        return self._format_faq_answer(user_input)

    def _format_catalog_answer(self, results: Any) -> str:
        if not results or not isinstance(results, list) or (
            len(results) == 0
        ) or (
            isinstance(results[0], dict) and results[0].get("error")
        ):
            return "Rất tiếc, không tìm thấy sản phẩm phù hợp."

        lines = ["Các sản phẩm phù hợp trong catalog Vingroup:"]
        for product in results:
            price = f"{product['price_vnd']:,} VNĐ".replace(",", ".")
            lines.append(f"- {product['name']}: {price}. {product.get('description', '')}")
        return "\n".join(lines)

    def _format_ticket_answer(self, ticket: Dict[str, Any]) -> str:
        return (
            f"Đã ghi nhận yêu cầu hỗ trợ của {ticket['customer_name']}. "
            f"Mã ticket: {ticket['ticket_id']}. "
            f"Trạng thái: {ticket['status']}, mức ưu tiên: {ticket['priority']}. "
            f"{ticket.get('message', '')}"
        )

    def _format_faq_answer(self, user_input: str) -> str:
        text = user_input.lower()
        if any(k in text for k in ["bảo hành", "bao hanh", "pin"]):
            return (
                "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm "
                "hoặc theo điều kiện bảo hành chính thức của VinFast tại thời điểm mua xe."
            )
        return (
            "VinAssistant chỉ hỗ trợ thông tin hệ sinh thái Vingroup "
            "(VinFast, Vinpearl). Bạn có thể hỏi về sản phẩm, giá, hoặc gửi yêu cầu hỗ trợ."
        )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
