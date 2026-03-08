import pytest
import torch
from libtcrlm import schema
from libtcrlm.tokeniser import *
from libtcrlm.tokeniser.token_indices import AminoAcidTokenIndex


@pytest.mark.parametrize(
    argnames=("key", "expected"),
    argvalues=(
        ("A", 3),
        ("C", 4),
        ("*", 0),
    ),
)
def test_custom_indexer(key, expected):
    val = AminoAcidTokenIndex.from_aa_char(key)
    assert val == expected


class TestBetaCdr3Tokeniser:
    tokeniser = BetaCdr3Tokeniser()

    def test_tokenise(self, mock_tcr):
        tokenised_tcr = self.tokeniser.tokenise(mock_tcr)
        expected = torch.tensor(
            [
                [2, 0, 0],
                [4, 1, 6],
                [3, 2, 6],
                [18, 3, 6],
                [16, 4, 6],
                [22, 5, 6],
                [7, 6, 6],
            ]
        )

        assert torch.equal(tokenised_tcr, expected)

    def test_tokenise_tcr_with_empty_beta_junction(self):
        tcr_with_empty_beta = schema.make_tcr_from_components(
            "TRAV1-1*01", "CASQYF", "TRBV2*01", None
        )

        with pytest.raises(RuntimeError):
            self.tokeniser.tokenise(tcr_with_empty_beta)


class TestCdrTokeniser:
    tokeniser = CdrTokeniser()

    def test_tokenise(self, mock_tcr):
        tokenised_tcr = self.tokeniser.tokenise(mock_tcr)
        expected = torch.tensor(
            [
                [2, 0, 0, 0],
                [19, 1, 6, 1],
                [18, 2, 6, 1],
                [8, 3, 6, 1],
                [7, 4, 6, 1],
                [22, 5, 6, 1],
                [8, 6, 6, 1],
                [14, 1, 6, 2],
                [3, 2, 6, 2],
                [12, 3, 6, 2],
                [5, 4, 6, 2],
                [8, 5, 6, 2],
                [12, 6, 6, 2],
                [4, 1, 6, 3],
                [3, 2, 6, 3],
                [19, 3, 6, 3],
                [16, 4, 6, 3],
                [22, 5, 6, 3],
                [7, 6, 6, 3],
                [18, 1, 5, 4],
                [14, 2, 5, 4],
                [9, 3, 5, 4],
                [12, 4, 5, 4],
                [22, 5, 5, 4],
                [7, 1, 6, 5],
                [22, 2, 6, 5],
                [14, 3, 6, 5],
                [14, 4, 6, 5],
                [6, 5, 6, 5],
                [10, 6, 6, 5],
                [4, 1, 6, 6],
                [3, 2, 6, 6],
                [18, 3, 6, 6],
                [16, 4, 6, 6],
                [22, 5, 6, 6],
                [7, 6, 6, 6],
            ]
        )

        assert torch.equal(tokenised_tcr, expected)

    def test_tokenise_tcr_with_unknown_residue(self):
        tcr_with_unknown_residue = schema.make_tcr_from_components(
            "TRAV23/DV6*02", "CATQYF", "TRBV2*01", "CASQYF"
        )

        tokenised_tcr = self.tokeniser.tokenise(tcr_with_unknown_residue)
        expected = torch.tensor(
            [
                [2, 0, 0, 0],
                [14, 1, 6, 1],
                [19, 2, 6, 1],
                [3, 3, 6, 1],
                [7, 4, 6, 1],
                [5, 5, 6, 1],
                [22, 6, 6, 1],
                [16, 1, 4, 2],
                [13, 2, 4, 2],
                [20, 4, 4, 2],
                [4, 1, 6, 3],
                [3, 2, 6, 3],
                [19, 3, 6, 3],
                [16, 4, 6, 3],
                [22, 5, 6, 3],
                [7, 6, 6, 3],
                [18, 1, 5, 4],
                [14, 2, 5, 4],
                [9, 3, 5, 4],
                [12, 4, 5, 4],
                [22, 5, 5, 4],
                [7, 1, 6, 5],
                [22, 2, 6, 5],
                [14, 3, 6, 5],
                [14, 4, 6, 5],
                [6, 5, 6, 5],
                [10, 6, 6, 5],
                [4, 1, 6, 6],
                [3, 2, 6, 6],
                [18, 3, 6, 6],
                [16, 4, 6, 6],
                [22, 5, 6, 6],
                [7, 6, 6, 6],
            ]
        )

        assert torch.equal(tokenised_tcr, expected)


class TestBetaCdrTokeniser:
    tokeniser = BetaCdrTokeniser()

    def test_tokenise(self, mock_tcr):
        tokenised_tcr = self.tokeniser.tokenise(mock_tcr)
        expected = torch.tensor(
            [
                [2, 0, 0, 0],
                [18, 1, 5, 1],
                [14, 2, 5, 1],
                [9, 3, 5, 1],
                [12, 4, 5, 1],
                [22, 5, 5, 1],
                [7, 1, 6, 2],
                [22, 2, 6, 2],
                [14, 3, 6, 2],
                [14, 4, 6, 2],
                [6, 5, 6, 2],
                [10, 6, 6, 2],
                [4, 1, 6, 3],
                [3, 2, 6, 3],
                [18, 3, 6, 3],
                [16, 4, 6, 3],
                [22, 5, 6, 3],
                [7, 6, 6, 3],
            ]
        )

        assert torch.equal(tokenised_tcr, expected)

    def test_tokenise_tcr_with_empty_beta(self):
        tcr_with_empty_beta = schema.make_tcr_from_components(
            "TRAV1-1*01", "CASQYF", None, None
        )

        with pytest.raises(RuntimeError):
            self.tokeniser.tokenise(tcr_with_empty_beta)


class TestAlphaTokeniser:
    tokeniser = AlphaCdrTokeniser()

    def test_tokenise(self, mock_tcr):
        tokenised_tcr = self.tokeniser.tokenise(mock_tcr)
        expected = torch.tensor(
            [
                [2, 0, 0, 0],
                [19, 1, 6, 1],
                [18, 2, 6, 1],
                [8, 3, 6, 1],
                [7, 4, 6, 1],
                [22, 5, 6, 1],
                [8, 6, 6, 1],
                [14, 1, 6, 2],
                [3, 2, 6, 2],
                [12, 3, 6, 2],
                [5, 4, 6, 2],
                [8, 5, 6, 2],
                [12, 6, 6, 2],
                [4, 1, 6, 3],
                [3, 2, 6, 3],
                [19, 3, 6, 3],
                [16, 4, 6, 3],
                [22, 5, 6, 3],
                [7, 6, 6, 3],
            ]
        )

        assert torch.equal(tokenised_tcr, expected)

    def test_tokenise_tcr_with_empty_alpha(self):
        tcr_with_empty_alpha = schema.make_tcr_from_components(
            None, None, "TRBV2*01", "CASQYF"
        )

        with pytest.raises(RuntimeError):
            self.tokeniser.tokenise(tcr_with_empty_alpha)

    def test_tokenise_tcr_with_unknown_residue(self):
        tcr_with_unknown_residue = schema.make_tcr_from_components(
            "TRAV23/DV6*02", "CATQYF", None, None
        )

        tokenised_tcr = self.tokeniser.tokenise(tcr_with_unknown_residue)
        expected = torch.tensor(
            [
                [2, 0, 0, 0],
                [14, 1, 6, 1],
                [19, 2, 6, 1],
                [3, 3, 6, 1],
                [7, 4, 6, 1],
                [5, 5, 6, 1],
                [22, 6, 6, 1],
                [16, 1, 4, 2],
                [13, 2, 4, 2],
                [20, 4, 4, 2],
                [4, 1, 6, 3],
                [3, 2, 6, 3],
                [19, 3, 6, 3],
                [16, 4, 6, 3],
                [22, 5, 6, 3],
                [7, 6, 6, 3],
            ]
        )

        assert torch.equal(tokenised_tcr, expected)
