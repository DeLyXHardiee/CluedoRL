import numpy as np

from cluedo import contract as C
from cluedo.core.rules import Game
from cluedo.rl.encoding import SeatEncoder

# 3 players. Envelope (5, 11, 20). Learner = Seat 0.
HANDS = [{0, 1, 6, 7, 12, 13}, {2, 3, 8, 9, 14, 15}, {4, 10, 16, 17, 18, 19}]


def scripted():
    return Game.from_deal(HANDS, (5, 11, 20), start_seat=0)


def test_initial_observation():
    g = scripted()
    enc = SeatEncoder(g, 0)
    obs = enc.observation()
    assert obs.shape == (C.OBS_DIM,) and obs.dtype == np.float32
    k = C.view(obs, "knowledge")
    assert k[0, 0, C.HAS] == 1 and k[0, 2, C.NOT] == 1 and k[1, 0, C.NOT] == 1
    assert k[1, 2, C.UNKNOWN] == 1 and k[6, 5, C.UNKNOWN] == 1
    assert k[3:6].sum() == 0  # padded Seats are all-zero
    assert C.view(obs, "n_players").tolist() == [1, 0, 0, 0]
    assert np.allclose(C.view(obs, "hand_sizes"), [1, 1, 1, 0, 0, 0])
    assert C.obs_phase(obs) == C.SUGGEST
    assert np.isclose(C.view(obs, "cand_total")[0], 4 * 4 * 7 / 324)
    assert np.allclose(C.view(obs, "cand_per_cat"), [4 / 6, 4 / 6, 7 / 9])
    assert C.view(obs, "turn")[0] == 0
    assert C.view(obs, "eliminated").sum() == 0
    assert C.mask_phase(enc.mask()) == C.SUGGEST and enc.mask().sum() == 324


def test_learner_relative_seats():
    g = scripted()
    obs = SeatEncoder(g, 2).observation()  # Seat 2's own hand is owner 0
    k = C.view(obs, "knowledge")
    assert k[0, 4, C.HAS] == 1 and k[1, 0, C.UNKNOWN] == 1  # rel 1 = absolute Seat 0


def play_script(g):
    g.apply(("suggest", 2, 8, 16))  # Seat 1 holds 2 and 8 -> Reveal decision
    g.apply(("reveal", 2))
    g.apply(("pass",))  # Learner ends Turn
    g.apply(("suggest", 1, 7, 12))  # Seat 1 (no match from Seat 2; Learner holds all three)
    return g


def test_public_history_and_reveal_mask():
    g = play_script(scripted())
    assert g.decision.seat == 0 and g.decision.phase == "reveal"
    enc = SeatEncoder(g, 0)
    obs = enc.observation()
    assert C.obs_phase(obs) == C.REVEAL
    m = enc.mask()
    assert C.mask_phase(m) == C.REVEAL
    assert set(np.flatnonzero(m)) == {C.encode_reveal(c) for c in (1, 7, 12)}
    # last Suggestion is the earlier one (the pending one is not answered yet)
    assert set(np.flatnonzero(C.view(obs, "last_cards"))) == {2, 8, 16}
    assert C.view(obs, "last_asker").tolist() == [1, 0, 0, 0, 0, 0]
    assert C.view(obs, "last_answerer").tolist() == [0, 1, 0, 0, 0, 0, 0]
    nf = C.view(obs, "name_freq")
    assert nf[0, 2] > 0 and nf[0, 8] > 0 and nf[0, 16] > 0 and nf[1].sum() == 0
    assert C.view(obs, "knowledge")[1, 2, C.HAS] == 1  # saw the shown card
    g.apply(("reveal", 1))
    obs = SeatEncoder(g, 0).observation()
    st = C.view(obs, "shown_to")
    assert st[1, 1] == 1 and st.sum() == 1  # Learner showed card 1 to rel Seat 1
    assert C.view(obs, "name_freq")[1, 1] > 0
    assert C.view(obs, "last_asker").tolist() == [0, 1, 0, 0, 0, 0]
    assert C.view(obs, "last_answerer")[0] == 1  # Learner answered


def test_unanswered_suggestion_marks_nobody():
    g = Game.from_deal(HANDS, (5, 11, 20), start_seat=0)
    g.apply(("suggest", 5, 11, 20))  # the Envelope: nobody answers
    obs = SeatEncoder(g, 0).observation()
    assert C.view(obs, "last_answerer")[6] == 1


def test_blind_zeroes_public_history_only():
    g = play_script(scripted())
    open_obs = SeatEncoder(g, 0).observation()
    blind_obs = SeatEncoder(g, 0, blind=True).observation()
    for name in C.PUBLIC_HISTORY:
        assert not C.view(blind_obs, name).any()
    assert open_obs[C.SLICES["name_freq"].sl].any()
    assert np.array_equal(blind_obs, C.blind(open_obs))


def test_accuse_mask_is_candidates_and_contains_envelope():
    g = scripted()
    g.apply(("suggest", 5, 11, 20))
    enc = SeatEncoder(g, 0)
    enc.observation()
    m = enc.mask()
    assert C.mask_phase(m) == C.ACCUSE
    assert m[C.encode_accuse(5, 11, 20)] and m[C.PASS_ACTION]
    assert m.sum() == enc.ded.candidate_count() + 1
