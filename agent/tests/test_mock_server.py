from __future__ import annotations

from agent.mock_server import MockState, SCENARIOS


def test_random_delay_scenario_generates_a_new_delay_for_each_call(monkeypatch) -> None:
    generated = iter((5.25, 9.75))
    monkeypatch.setattr(
        "agent.mock_server.random.uniform",
        lambda minimum, maximum: next(generated),
    )

    state = MockState("random-delay")

    assert "random-delay" in SCENARIOS
    assert state.delay("navigation") == 5.25
    assert state.delay("navigation") == 9.75
