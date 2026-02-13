# src/schemas.py
INSIGHTS_JSON_SCHEMA = {
    "name": "safety_insights_response",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "executive_summary": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 8,
            },
            "key_findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "what_it_means": {"type": "string"},
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "metric": {"type": "string"},
                                    "value": {"type": ["number", "string"]},
                                    "slice": {"type": "string"},
                                },
                                "required": ["metric", "value", "slice"],
                            },
                            "minItems": 1,
                            "maxItems": 6,
                        },
                    },
                    "required": ["title", "what_it_means", "evidence"],
                },
                "minItems": 2,
                "maxItems": 10,
            },
            "recommended_actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "priority": {"type": "string", "enum": ["P0", "P1", "P2"]},
                        "action": {"type": "string"},
                        "who": {"type": "string"},
                        "expected_outcome": {"type": "string"},
                        "why": {"type": "string"},
                    },
                    "required": ["priority", "action", "who", "expected_outcome", "why"],
                },
                "minItems": 2,
                "maxItems": 10,
            },
            "charts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "chart_id": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": ["bar", "line", "scatter", "pareto", "histogram", "stacked_bar"],
                        },
                        "title": {"type": "string"},
                        "x": {"type": "string"},
                        "y": {"type": "string"},
                        "series": {"type": ["string", "null"]},
                        "notes": {"type": "string"},
                    },
                    "required": ["chart_id", "type", "title", "x", "y", "series", "notes"],
                },
                "minItems": 2,
                "maxItems": 6,
            },
            "follow_up_questions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 10,
            },
            "data_quality_notes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "issue": {"type": "string"},
                        "impact": {"type": "string"},
                        "suggested_fix": {"type": "string"},
                    },
                    "required": ["issue", "impact", "suggested_fix"],
                },
                "minItems": 0,
                "maxItems": 10,
            },
        },
        "required": [
            "executive_summary",
            "key_findings",
            "recommended_actions",
            "charts",
            "follow_up_questions",
            "data_quality_notes",
        ],
    },
}
