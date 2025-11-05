import pytest
from vocabulous import Vocabulous


class TestAlgebraicConsistency:
    def _datasets(self):
        A = {
            "en": ["hello world", "good day", "this is a test"],
            "fr": ["bonjour le monde", "bon jour", "ceci est un test"],
        }
        B = {
            "en": ["how are you", "hello there"],
            "fr": ["comment ca va", "bonjour"],
        }
        return A, B

    def test_merge_equivalence_with_joint_training(self):
        A, B = self._datasets()
        va, _ = Vocabulous().train(A)
        vb, _ = Vocabulous().train(B)
        vm = va.merge(vb)
        joint = {k: (A.get(k, []) + B.get(k, [])) for k in set(A) | set(B)}
        vj, _ = Vocabulous().train(joint)
        assert vm.languages == vj.languages
        assert vm.word_lang_freq == vj.word_lang_freq

    def test_partial_training_equivalence(self):
        A, B = self._datasets()
        v0, _ = Vocabulous().train(A)
        v_inc = v0.partial_train(B)
        joint = {k: (A.get(k, []) + B.get(k, [])) for k in set(A) | set(B)}
        vj, _ = Vocabulous().train(joint)
        assert v_inc.languages == vj.languages
        assert v_inc.word_lang_freq == vj.word_lang_freq

    def test_get_language_word_probs_normalization(self):
        data = {"en": ["a b c", "a a"], "fr": ["a b", "b b"]}
        v, _ = Vocabulous().train(data)
        probs = v.get_language_word_probs(alpha=0.0)
        for l, m in probs.items():
            s = sum(m.values())
            assert pytest.approx(s, rel=1e-9) == 1.0
        probs_s = v.get_language_word_probs(alpha=1.0)
        for l, m in probs_s.items():
            s = sum(m.values())
            assert pytest.approx(s, rel=1e-9) == 1.0

    def test_language_overlap_properties(self):
        data = {
            "en": ["a b c", "a a d"],
            "fr": ["a b", "b b e"],
            "de": ["x y z"],
        }
        v, _ = Vocabulous().train(data)
        mat_cos = v.language_overlap(metric="cosine")
        mat_jac = v.language_overlap(metric="jaccard")
        import numpy as np
        for mat in (mat_cos, mat_jac):
            assert list(mat.index) == list(mat.columns)
            assert (mat.values.T == mat.values).all()
            diag = mat.values.diagonal()
            assert np.allclose(diag, 1.0)
