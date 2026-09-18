"""Tests for GET /participants/compare: score comparison merged into intervals."""

import pytest
from fastapi.testclient import TestClient

import app.db as db_module
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient backed by an isolated SQLite file."""
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "compare-test.db"))
    db_module.init_db()
    return TestClient(app)


def _post(client, event_id, participant_id, points, correction_of=None):
    payload = {
        "event_id": event_id,
        "participant_id": participant_id,
        "points": points,
    }
    if correction_of is not None:
        payload["correction_of"] = correction_of
    r = client.post("/events", json=payload)
    assert r.status_code == 201
    return r.json()


def _compare(client, **params):
    r = client.get("/participants/compare", params=params)
    assert r.status_code == 200, params
    return r.json()


BASE = dict(left_id="alice", right_id="bob")


def test_from_zero_merges_consecutive_equal_snapshots(client):
    # rev1 alice +10; rev2 bob +10 (tie starts); rev3 unrelated carol event
    # (tie persists); rev4 alice +5 (alice leads).
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 10)
    _post(client, "e3", "carol", 3)
    _post(client, "e4", "alice", 5)

    body = _compare(client, **BASE, from_revision=0, to_revision=4)
    assert body["from_revision"] == 0
    assert body["to_revision"] == 4
    assert body["intervals"] == [
        {
            "start_revision": 0,
            "end_revision": 0,
            "left_score": 0,
            "right_score": 0,
            "leader": "tie",
        },
        {
            "start_revision": 1,
            "end_revision": 1,
            "left_score": 10,
            "right_score": 0,
            "leader": "left",
        },
        {
            "start_revision": 2,
            "end_revision": 3,
            "left_score": 10,
            "right_score": 10,
            "leader": "tie",
        },
        {
            "start_revision": 4,
            "end_revision": 4,
            "left_score": 15,
            "right_score": 10,
            "leader": "left",
        },
    ]


def test_intervals_cover_query_range_contiguously(client):
    for i in range(1, 6):
        _post(client, f"e{i}", "carol", i)
    body = _compare(client, **BASE, from_revision=0, to_revision=5)
    assert body["intervals"] == [
        {
            "start_revision": 0,
            "end_revision": 5,
            "left_score": 0,
            "right_score": 0,
            "leader": "tie",
        }
    ]

    # A sub-range not starting at 0 is still fully covered.
    body = _compare(client, **BASE, from_revision=2, to_revision=4)
    assert body["intervals"][0]["start_revision"] == 2
    assert body["intervals"][-1]["end_revision"] == 4
    covered = [
        rev
        for iv in body["intervals"]
        for rev in range(iv["start_revision"], iv["end_revision"] + 1)
    ]
    assert covered == list(range(2, 5))


def test_equal_revisions_is_single_point_interval(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 5)
    body = _compare(client, **BASE, from_revision=2, to_revision=2)
    assert body["intervals"] == [
        {
            "start_revision": 2,
            "end_revision": 2,
            "left_score": 10,
            "right_score": 5,
            "leader": "left",
        }
    ]


def test_unknown_participants_do_not_error_and_tie_covers_range(client):
    _post(client, "e1", "carol", 10)
    body = _compare(
        client, left_id="ghost-a", right_id="ghost-b",
        from_revision=0, to_revision=1,
    )
    assert body["intervals"] == [
        {
            "start_revision": 0,
            "end_revision": 1,
            "left_score": 0,
            "right_score": 0,
            "leader": "tie",
        }
    ]

    # One known, one unknown: the known side leads on its scoring revisions.
    body = _compare(
        client, left_id="carol", right_id="ghost",
        from_revision=0, to_revision=1,
    )
    assert [(iv["leader"], iv["start_revision"]) for iv in body["intervals"]] == [
        ("tie", 0),
        ("left", 1),
    ]


def test_empty_ledger_zero_to_zero(client):
    body = _compare(client, **BASE, from_revision=0, to_revision=0)
    assert body["intervals"] == [
        {
            "start_revision": 0,
            "end_revision": 0,
            "left_score": 0,
            "right_score": 0,
            "leader": "tie",
        }
    ]


def test_reattribution_debits_and_credits_in_same_revision(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 4)
    # rev3: alice's 10-point event is re-attributed to bob via a correction.
    _post(client, "e3", "bob", 10, correction_of="e1")
    body = _compare(client, **BASE, from_revision=0, to_revision=3)
    assert body["intervals"] == [
        {"start_revision": 0, "end_revision": 0, "left_score": 0,
         "right_score": 0, "leader": "tie"},
        {"start_revision": 1, "end_revision": 1, "left_score": 10,
         "right_score": 0, "leader": "left"},
        {"start_revision": 2, "end_revision": 2, "left_score": 10,
         "right_score": 4, "leader": "left"},
        # At rev3 alice loses all 10 (-> 0) and bob gains the 10 (4 -> 14)
        # within the same snapshot.
        {"start_revision": 3, "end_revision": 3, "left_score": 0,
         "right_score": 14, "leader": "right"},
    ]

    # Cross-check against /leaderboard totals at the correction revision.
    lb = client.get("/leaderboard", params={"revision": 3}).json()["entries"]
    totals = {e["participant_id"]: e["total_points"] for e in lb}
    assert totals.get("alice", 0) == 0
    assert totals["bob"] == 14
    assert "alice" not in totals


def test_reattribution_passes_through_tie(client):
    _post(client, "e1", "alice", 10)
    # bob gets the same 10 by re-attribution: snapshot at rev2 is a tie.
    _post(client, "e2", "bob", 10, correction_of="e1")
    _post(client, "e3", "bob", 3)
    body = _compare(client, **BASE, from_revision=1, to_revision=3)
    assert body["intervals"] == [
        {"start_revision": 1, "end_revision": 1, "left_score": 10,
         "right_score": 0, "leader": "left"},
        {"start_revision": 2, "end_revision": 2, "left_score": 0,
         "right_score": 10, "leader": "right"},
        {"start_revision": 3, "end_revision": 3, "left_score": 0,
         "right_score": 13, "leader": "right"},
    ]


def test_score_correction_chain_tip(client):
    # e1 (alice 10) -> e2 (alice 12) -> e3 (alice 15); bob scores 15 at rev4.
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "alice", 12, correction_of="e1")
    _post(client, "e3", "alice", 15, correction_of="e2")
    _post(client, "e4", "bob", 15)
    body = _compare(client, **BASE, from_revision=0, to_revision=4)
    assert [
        (iv["start_revision"], iv["end_revision"], iv["left_score"],
         iv["right_score"], iv["leader"])
        for iv in body["intervals"]
    ] == [
        (0, 0, 0, 0, "tie"),
        (1, 1, 10, 0, "left"),
        (2, 2, 12, 0, "left"),
        (3, 3, 15, 0, "left"),
        (4, 4, 15, 15, "tie"),
    ]


def test_forked_chain_both_tips_count(client):
    # e1 -> e2 moves 10 points alice -> bob; e3 corrects the superseded e1
    # again, forking the chain: both tips stay effective.
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 10, correction_of="e1")
    _post(client, "e3", "carol", 7, correction_of="e1")
    body = _compare(client, **BASE, from_revision=3, to_revision=3)
    assert body["intervals"] == [
        {"start_revision": 3, "end_revision": 3, "left_score": 0,
         "right_score": 10, "leader": "right"},
    ]
    lb = client.get("/leaderboard", params={"revision": 3}).json()["entries"]
    totals = {e["participant_id"]: e["total_points"] for e in lb}
    assert totals == {"bob": 10, "carol": 7}


def test_same_id_is_always_tie(client):
    _post(client, "e1", "alice", 10)
    body = _compare(
        client, left_id="alice", right_id="alice",
        from_revision=0, to_revision=1,
    )
    assert all(iv["leader"] == "tie" for iv in body["intervals"])
    assert body["intervals"] == [
        {"start_revision": 0, "end_revision": 0, "left_score": 0,
         "right_score": 0, "leader": "tie"},
        {"start_revision": 1, "end_revision": 1, "left_score": 10,
         "right_score": 10, "leader": "tie"},
    ]


def test_negative_points_reflected_in_scores(client):
    _post(client, "e1", "alice", 5)
    _post(client, "e2", "bob", -3)
    body = _compare(client, **BASE, from_revision=2, to_revision=2)
    iv = body["intervals"][0]
    assert iv["left_score"] == 5
    assert iv["right_score"] == -3
    assert iv["leader"] == "left"


def test_echoes_ids_and_query_params(client):
    _post(client, "e1", "alice", 1)
    body = _compare(
        client, left_id="alice", right_id="bob",
        from_revision=0, to_revision=1,
    )
    assert body["left_id"] == "alice"
    assert body["right_id"] == "bob"


def test_missing_params_422(client):
    assert client.get("/participants/compare").status_code == 422
    assert client.get(
        "/participants/compare", params={"right_id": "bob",
                                         "from_revision": 0, "to_revision": 1}
    ).status_code == 422
    assert client.get(
        "/participants/compare",
        params={"left_id": "alice", "right_id": "bob", "to_revision": 1},
    ).status_code == 422
    assert client.get(
        "/participants/compare",
        params={"left_id": "alice", "right_id": "bob", "from_revision": 0},
    ).status_code == 422


def test_non_integer_and_negative_params_422(client):
    base = {"left_id": "alice", "right_id": "bob"}
    assert client.get(
        "/participants/compare", params={**base, "from_revision": "x",
                                         "to_revision": 1}
    ).status_code == 422
    assert client.get(
        "/participants/compare", params={**base, "from_revision": -1,
                                         "to_revision": 1}
    ).status_code == 422


def test_from_greater_than_to_422(client):
    _post(client, "e1", "alice", 10)
    assert client.get(
        "/participants/compare",
        params={**BASE, "from_revision": 2, "to_revision": 1},
    ).status_code == 422


def test_ordering_checked_before_range(client):
    # both out of range and wrongly ordered: ordering wins -> 422
    assert client.get(
        "/participants/compare",
        params={**BASE, "from_revision": 9, "to_revision": 5},
    ).status_code == 422


def test_to_revision_beyond_current_404(client):
    _post(client, "e1", "alice", 10)
    assert client.get(
        "/participants/compare",
        params={**BASE, "from_revision": 0, "to_revision": 2},
    ).status_code == 404
    assert client.get(
        "/participants/compare",
        params={**BASE, "from_revision": 5, "to_revision": 5},
    ).status_code == 404


def test_from_beyond_current_but_equal_to_also_404(client):
    _post(client, "e1", "alice", 10)
    assert client.get(
        "/participants/compare",
        params={**BASE, "from_revision": 2, "to_revision": 2},
    ).status_code == 404


def test_empty_ledger_nonzero_range_404(client):
    assert client.get(
        "/participants/compare",
        params={**BASE, "from_revision": 0, "to_revision": 1},
    ).status_code == 404


def test_route_not_shadowed_by_score_breakdown(client):
    # "compare" must be routed to the compare endpoint, not captured as a
    # participant_id by /participants/{participant_id}/score-breakdown.
    r = client.get(
        "/participants/compare",
        params={**BASE, "from_revision": 0, "to_revision": 0},
    )
    assert r.status_code == 200
    assert "intervals" in r.json()
