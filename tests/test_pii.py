"""PII scrubbing, including regressions for bugs found against real TWCS text."""
import pytest

from hiver.pii import extract_agent, normalise, scrub

BRAND = frozenset({"SpotifyCares", "115888"})


@pytest.mark.parametrize("raw,expect", [
    ("email me at bob.smith@gmail.com", "<EMAIL>"),
    ("call +1 (555) 123-4567 please", "<PHONE>"),
    ("card 4111 1111 1111 1111 declined", "<CARD>"),
    ("see https://t.co/abc123 now", "<URL>"),
    ("my order #A1B2-C3D4 is late", "<ORDER>"),
])
def test_masks_identifiers(raw, expect):
    assert expect in scrub(raw).text


@pytest.mark.parametrize("raw", [
    "I have been waiting since 2017",
    "give me my 100 songs back",
    "premium is 9.99 a month",
    "it fails 3 times out of 10",
])
def test_does_not_mask_ordinary_numbers(raw):
    """Over-scrubbing destroys the signal the classifier needs."""
    assert scrub(raw).text == raw
    assert not scrub(raw).had_pii


def test_handle_wins_over_phone():
    """Regression: the phone rule used to eat @115712 before the handle rule ran."""
    assert scrub("Hi @115712 there").text == "Hi @<USER> there"


def test_card_does_not_swallow_following_space():
    """Regression: the card pattern consumed its trailing separator."""
    assert scrub("card 4111 1111 1111 1111 declined").text == "card <CARD> declined"


def test_brand_alias_is_not_masked_as_a_customer():
    """Regression: @115888 is Spotify itself, in 15k customer tweets."""
    out = scrub("@115888 your app sucks, ask @999123", BRAND).text
    assert out == "@BRAND your app sucks, ask @<USER>"


def test_brand_mention_is_not_counted_as_pii():
    """Regression: adding a `brand` counter made had_pii fire on 92% of messages."""
    assert not scrub("@115888 hello", BRAND).had_pii
    assert scrub("@115888 mail me at a@b.com", BRAND).had_pii


@pytest.mark.parametrize("raw,body,agent", [
    ("We can help /QI", "We can help", "QI"),
    ("Check it out /L", "Check it out", "L"),          # 1-char sigs: 950 in corpus
    ("Sent you a DM ^ABC", "Sent you a DM", "ABC"),    # 3-char sigs exist too
    ("DM us your email? /J <URL>", "DM us your email?", "J"),  # sig before trailing link
    ("you can use A and/or", "you can use A and/or", None),    # not a signature
    ("no signature here", "no signature here", None),
])
def test_agent_signature_extraction(raw, body, agent):
    assert extract_agent(raw) == (body, agent)


def test_normalise_unescapes_entities_and_collapses_space():
    assert normalise("a &amp;  b &quot;c&quot;\n d") == 'a & b "c" d'
