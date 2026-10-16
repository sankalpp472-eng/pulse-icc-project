"""
Mock tool implementations for the airline agent.
Each function returns fake but realistic data.
Later these will be swapped for real OpenFaaS HTTP calls.
"""

import json
from datetime import datetime, timedelta
import random

# ---------------------------------------------------------------------------
# Tool registry — maps tool_name -> (function, description, arg_schema)
# ---------------------------------------------------------------------------

def get_user_details(customer_id: str) -> dict:
    """Return mock customer profile."""
    return {
        "customer_id": customer_id,
        "name": "Arjun Sharma",
        "email": "arjun.sharma@example.com",
        "loyalty_tier": "Gold",
        "loyalty_points": 12400,
        "phone": "+91-9876543210",
    }


def get_booking(booking_id: str) -> dict:
    """Return mock booking details."""
    return {
        "booking_id": booking_id,
        "customer_id": "C001",
        "flight_number": "AI-202",
        "origin": "BLR",
        "destination": "DEL",
        "departure": "2026-10-15T08:30:00",
        "arrival": "2026-10-15T11:00:00",
        "seat": "14A",
        "class": "Economy",
        "status": "Confirmed",
        "baggage_allowance_kg": 25,
        "base_fare": 4500,
        "currency": "INR",
    }


def get_all_customer_flight_bookings(customer_id: str) -> dict:
    """Return all bookings for a customer."""
    return {
        "customer_id": customer_id,
        "bookings": [
            {"booking_id": "BK101", "flight_number": "AI-202", "route": "BLR→DEL", "date": "2026-10-15", "status": "Confirmed"},
            {"booking_id": "BK085", "flight_number": "6E-411", "route": "DEL→BOM", "date": "2026-09-20", "status": "Completed"},
        ],
    }


def get_flight(flight_number: str) -> dict:
    """Return flight info."""
    return {
        "flight_number": flight_number,
        "airline": "Air India",
        "origin": "BLR",
        "destination": "DEL",
        "departure": "2026-10-15T08:30:00",
        "arrival": "2026-10-15T11:00:00",
        "aircraft": "Airbus A320",
        "status": "On Time",
        "available_seats": 12,
    }


def get_available_flights(origin: str, destination: str, date: str) -> dict:
    """Search available flights between two cities on a date."""
    return {
        "origin": origin,
        "destination": destination,
        "date": date,
        "flights": [
            {"flight_number": "AI-202", "departure": "08:30", "arrival": "11:00", "seats_left": 12, "price": 4500},
            {"flight_number": "6E-301", "departure": "13:15", "arrival": "15:45", "seats_left": 5,  "price": 3800},
            {"flight_number": "UK-810", "departure": "19:00", "arrival": "21:30", "seats_left": 20, "price": 5100},
        ],
    }


def get_seat_map(flight_number: str) -> dict:
    """Return seat availability for a flight."""
    return {
        "flight_number": flight_number,
        "seat_map": {
            "12A": "available", "12B": "occupied", "12C": "available",
            "13A": "occupied",  "13B": "available", "13C": "occupied",
            "14A": "occupied",  "14B": "available", "14C": "available",
        },
    }


def calculate_flight_fee(booking_id: str, new_flight_number: str) -> dict:
    """Estimate fee for changing a booking to a different flight."""
    return {
        "booking_id": booking_id,
        "current_flight": "AI-202",
        "new_flight": new_flight_number,
        "change_fee": 1200,
        "fare_difference": 300,
        "total_charge": 1500,
        "currency": "INR",
    }


def update_reservation_flights(booking_id: str, new_flight_number: str) -> dict:
    """Change the flight on an existing booking."""
    return {
        "booking_id": booking_id,
        "status": "Updated",
        "new_flight_number": new_flight_number,
        "message": f"Booking {booking_id} successfully changed to flight {new_flight_number}.",
    }


def update_reservation_baggages(booking_id: str, baggage_kg: int) -> dict:
    """Update baggage allowance on a booking."""
    return {
        "booking_id": booking_id,
        "status": "Updated",
        "new_baggage_allowance_kg": baggage_kg,
        "additional_charge": max(0, (baggage_kg - 25)) * 400,
        "currency": "INR",
    }


def update_reservation_passengers(booking_id: str, passenger_name: str, passport_number: str) -> dict:
    """Update passenger details on a booking."""
    return {
        "booking_id": booking_id,
        "status": "Updated",
        "passenger_name": passenger_name,
        "passport_number": passport_number,
        "message": "Passenger details updated successfully.",
    }


def cancel_reservation(booking_id: str) -> dict:
    """Cancel a booking and return refund info."""
    return {
        "booking_id": booking_id,
        "status": "Cancelled",
        "refund_amount": 3600,
        "refund_timeline": "5–7 business days",
        "currency": "INR",
    }


def send_certificate(customer_id: str, amount: int, reason: str) -> dict:
    """Issue a travel voucher/certificate to a customer."""
    cert_id = f"CERT-{random.randint(10000, 99999)}"
    return {
        "certificate_id": cert_id,
        "customer_id": customer_id,
        "amount": amount,
        "currency": "INR",
        "reason": reason,
        "valid_until": (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d"),
        "status": "Issued",
    }


def think(thought: str) -> dict:
    """Internal reasoning step — no-op, just echoes the thought."""
    return {"thought_recorded": thought}


def transfer_to_human_agents(reason: str) -> dict:
    """Escalate to a human agent."""
    return {
        "status": "Transferred",
        "ticket_id": f"HT-{random.randint(1000, 9999)}",
        "reason": reason,
        "message": "A human agent will contact the customer within 2 hours.",
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = {
    "get_user_details": {
        "fn": get_user_details,
        "description": "Get customer profile and loyalty info.",
        "args": {"customer_id": "string — the customer ID"},
    },
    "get_booking": {
        "fn": get_booking,
        "description": "Get full details of a single booking by booking ID.",
        "args": {"booking_id": "string — the booking ID"},
    },
    "get_all_customer_flight_bookings": {
        "fn": get_all_customer_flight_bookings,
        "description": "Get all flight bookings for a customer.",
        "args": {"customer_id": "string — the customer ID"},
    },
    "get_flight": {
        "fn": get_flight,
        "description": "Get details for a specific flight by flight number.",
        "args": {"flight_number": "string — e.g. 'AI-202'"},
    },
    "get_available_flights": {
        "fn": get_available_flights,
        "description": "Search available flights between two airports on a date.",
        "args": {
            "origin": "string — IATA airport code",
            "destination": "string — IATA airport code",
            "date": "string — YYYY-MM-DD",
        },
    },
    "get_seat_map": {
        "fn": get_seat_map,
        "description": "Get seat availability map for a flight.",
        "args": {"flight_number": "string — e.g. 'AI-202'"},
    },
    "calculate_flight_fee": {
        "fn": calculate_flight_fee,
        "description": "Calculate the fee for changing a booking to a different flight.",
        "args": {
            "booking_id": "string — existing booking ID",
            "new_flight_number": "string — target flight number",
        },
    },
    "update_reservation_flights": {
        "fn": update_reservation_flights,
        "description": "Change the flight on an existing booking.",
        "args": {
            "booking_id": "string — existing booking ID",
            "new_flight_number": "string — new flight number",
        },
    },
    "update_reservation_baggages": {
        "fn": update_reservation_baggages,
        "description": "Update baggage allowance on a booking.",
        "args": {
            "booking_id": "string — booking ID",
            "baggage_kg": "integer — total baggage in kg",
        },
    },
    "update_reservation_passengers": {
        "fn": update_reservation_passengers,
        "description": "Update passenger name or passport number on a booking.",
        "args": {
            "booking_id": "string — booking ID",
            "passenger_name": "string — full name",
            "passport_number": "string — passport number",
        },
    },
    "cancel_reservation": {
        "fn": cancel_reservation,
        "description": "Cancel a booking and get refund details.",
        "args": {"booking_id": "string — booking ID"},
    },
    "send_certificate": {
        "fn": send_certificate,
        "description": "Issue a travel voucher or certificate to a customer.",
        "args": {
            "customer_id": "string — customer ID",
            "amount": "integer — voucher amount in INR",
            "reason": "string — reason for issuing the certificate",
        },
    },
    "think": {
        "fn": think,
        "description": "Record an internal reasoning step. Use this to plan before acting.",
        "args": {"thought": "string — your internal reasoning"},
    },
    "transfer_to_human_agents": {
        "fn": transfer_to_human_agents,
        "description": "Escalate to a human agent when you cannot resolve the issue.",
        "args": {"reason": "string — reason for escalation"},
    },
}


def execute_tool(tool_name: str, args: dict) -> str:
    """Execute a tool by name with given args. Returns result as JSON string."""
    if tool_name not in TOOLS:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = TOOLS[tool_name]["fn"](**args)
        return json.dumps(result, indent=2)
    except TypeError as e:
        return json.dumps({"error": f"Invalid arguments for {tool_name}: {e}"})


def get_tool_descriptions() -> str:
    """Return a formatted string of all tool names and descriptions."""
    lines = []
    for name, meta in TOOLS.items():
        args_str = ", ".join(f"{k}: {v}" for k, v in meta["args"].items())
        lines.append(f"- {name}({args_str})\n  {meta['description']}")
    return "\n".join(lines)
