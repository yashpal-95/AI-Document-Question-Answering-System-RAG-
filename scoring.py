# ============================================================
#  scoring.py  –  Human-readable score labels
# ============================================================


def get_accuracy_label(score: float) -> str:
    if score >= 85:
        return "✅ Excellent"
    elif score >= 65:
        return "🟡 Good"
    elif score >= 40:
        return "🟠 Partial"
    elif score >= 15:
        return "🔴 Weak"
    else:
        return "🚨 None"


def get_hallucination_label(score: float) -> str:
    if score <= 10:
        return "✅ Minimal"
    elif score <= 35:
        return "🟡 Low"
    elif score <= 60:
        return "🟠 Moderate"
    elif score <= 85:
        return "🔴 High"
    else:
        return "🚨 Severe"