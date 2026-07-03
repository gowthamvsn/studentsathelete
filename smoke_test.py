"""Exercise all three views outside Streamlit to catch runtime errors."""
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=240)
at.run()
assert not at.exception, f"Radar Insights exception: {at.exception}"
print("Radar Insights view: OK -", len(at.metric), "metrics")
at.sidebar.radio[0].set_value("Athlete Focus").run()
assert not at.exception, f"Athlete view exception: {at.exception}"
print("Athlete Focus view:  OK -", len(at.metric), "metrics")
at.sidebar.radio[0].set_value("Injury Focus").run()
assert not at.exception, f"Injury view exception: {at.exception}"
print("Injury Focus view:   OK -", len(at.metric), "metrics")
