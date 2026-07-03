"""Exercise all four views outside Streamlit to catch runtime errors."""
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=240)
at.run()
assert not at.exception, f"Radar Insights exception: {at.exception}"
print("Radar Insights: OK")
for v in ["Causality Lab", "Athlete Focus", "Injury Focus"]:
    at.sidebar.radio[0].set_value(v).run()
    assert not at.exception, f"{v} exception: {at.exception}"
    print(f"{v}: OK")
