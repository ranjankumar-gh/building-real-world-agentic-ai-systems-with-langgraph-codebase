"""Chapter 1: atlas/naive.py - the fragile first draft, not yet broken on purpose."""

from langchain_core.messages import AIMessage

from atlas import naive


def test_issue_refund_marks_ticket_refunded_and_records_the_call():
    naive._REFUNDS["T-1001"] = {"status": "pending", "amount": 49.0}
    naive.refund_calls.clear()

    result = naive.issue_refund.invoke({"ticket_id": "T-1001"})

    assert result == "Refund of $49.00 issued for T-1001."
    assert naive._REFUNDS["T-1001"]["status"] == "refunded"
    assert naive.refund_calls == ["T-1001"]


def test_text_of_handles_plain_string_content():
    message = AIMessage(content="hello")

    assert naive.text_of(message) == "hello"


def test_text_of_handles_content_block_list():
    message = AIMessage(
        content=[{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]
    )

    assert naive.text_of(message) == "hello world"
