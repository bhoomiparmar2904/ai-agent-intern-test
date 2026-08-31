"""
Order lookup tool.

The model never sees the raw orders.json. This function is called by the agent loop
as a tool, and returns ONLY an allowlisted, customer-safe subset of fields -- never
customer.name/email/shipping_address, and never anything inside "internal" (risk_score,
warehouse_note, support_tags). This is enforced here in code, not by prompting, so it
can't be bypassed by a prompt injection in the data.
"""
from __future__ import annotations

import json
import re
from typing import Optional


ORDER_ID_RE = re.compile(r"^ORD-\d{4}$")


def _normalize_order_id(raw: str) -> str:
    return raw.strip().upper()


def load_orders(path: str) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {o["order_id"]: o for o in data["orders"]}


class OrderLookupTool:
    name = "order_lookup"
    description = (
        "Look up the current status of a customer order by order ID. Returns only "
        "customer-safe fields. Use this whenever a question requires order-specific "
        "information -- never guess or answer from memory."
    )
    schema = {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "Order ID as given by the customer, e.g. 'ORD-1007'.",
            }
        },
        "required": ["order_id"],
    }

    def __init__(self, orders_path: str):
        self._orders = load_orders(orders_path)

    def call(self, order_id: str) -> dict:
        """Returns a sanitized, status-aware result dict. Never raises on bad input --
        always returns a structured result the agent can react to safely."""
        if not order_id or not order_id.strip():
            return {"found": False, "error": "missing_order_id"}

        normalized = _normalize_order_id(order_id)
        if not ORDER_ID_RE.match(normalized):
            return {"found": False, "error": "malformed_order_id", "supplied": order_id}

        order = self._orders.get(normalized)
        if order is None:
            return {"found": False, "error": "not_found", "supplied": order_id}

        status = order["status"]

        # Status-precedence logic: decide whether shipping/delivery fields are
        # trustworthy to surface, per the data dictionary rules. This is computed
        # here in code -- the model is not trusted to apply this judgment itself.
        show_delivery_fields = status not in ("cancelled", "returned")
        eta_available = show_delivery_fields and bool(order.get("estimated_delivery"))

        result = {
            "found": True,
            "order_id": order["order_id"],
            "membership_tier": order["membership_tier"],
            "items": [
                {"name": i["name"], "quantity": i["quantity"], "final_sale": i["final_sale"]}
                for i in order["items"]
            ],
            "placed_at": order["placed_at"],
            "status": status,
            "status_updated_at": order["status_updated_at"],
            "customer_safe_message": order["customer_safe_message"],
            "carrier": order["carrier"] if show_delivery_fields else None,
            "tracking_number": order["tracking_number"] if show_delivery_fields else None,
            "estimated_delivery": order["estimated_delivery"] if eta_available else None,
            "eta_available": eta_available,
            "requires_human_review": status in ("exception",),
        }
        return result
