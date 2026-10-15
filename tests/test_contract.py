import numpy as np
import pytest

from cluedo import contract as C


def test_action_layout():
    assert C.N_ACTIONS == 670
    assert (C.SUGGEST_START, C.ACCUSE_START, C.PASS_ACTION, C.REVEAL_START) == (0, 324, 648, 649)
    assert C.OBS_DIM == sum(s.size for s in C.SLICES.values())


def test_suggest_accuse_roundtrip_all_combos():
    seen = set()
    for s in range(6):
        for w in range(6, 12):
            for r in range(12, 21):
                a = C.encode_suggest(s, w, r)
                seen.add(a)
                assert 0 <= a < 324
                assert C.decode_action(a) == C.Action("suggest", (s, w, r))
                b = C.encode_accuse(s, w, r)
                assert 324 <= b < 648
                assert C.decode_action(b) == C.Action("accuse", (s, w, r))
    assert len(seen) == 324


def test_pass_and_reveal():
    assert C.decode_action(648) == C.Action("pass", ())
    for c in range(21):
        a = C.encode_reveal(c)
        assert 649 <= a < 670
        assert C.decode_action(a) == C.Action("reveal", (c,))
    with pytest.raises(ValueError):
        C.decode_action(670)


def test_slices_contiguous_and_disjoint():
    pos = 0
    for name, s in C.SLICES.items():
        assert s.start == pos, name
        pos = s.stop
    assert pos == C.OBS_DIM
    assert set(C.PUBLIC_HISTORY) == {"name_freq", "shown_to", "last_cards", "last_asker", "last_answerer"}


def test_blind_zeroes_only_public_history():
    obs = np.ones(C.OBS_DIM, dtype=np.float32)
    b = C.blind(obs)
    for name in C.SLICES:
        v = C.view(b, name)
        assert (v == 0).all() == (name in C.PUBLIC_HISTORY), name
    assert (obs == 1).all()


def test_masks_single_phase():
    assert C.mask_phase(C.suggest_mask()) == C.SUGGEST
    m = C.accuse_mask([(0, 6, 12)])
    assert C.mask_phase(m) == C.ACCUSE and m[C.PASS_ACTION] and m[C.encode_accuse(0, 6, 12)]
    assert C.mask_phase(C.accuse_mask([])) == C.ACCUSE
    assert C.mask_phase(C.reveal_mask([3, 7])) == C.REVEAL
    with pytest.raises(ValueError):
        C.mask_phase(C.suggest_mask() | C.reveal_mask([1]))


def test_relative_seats():
    for n in range(3, 7):
        for learner in range(n):
            assert C.relative_seat(learner, learner, n) == 0
            for rel in range(n):
                assert C.relative_seat(C.absolute_seat(rel, learner, n), learner, n) == rel
