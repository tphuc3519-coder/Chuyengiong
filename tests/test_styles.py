"""The beat styles: the catalogue, and the two things a name resolves to.

Pure data and pure functions, so all of it is decided here rather than in a
container. The tests fall into three groups and each is guarding a different
kind of mistake:

* **The catalogue is well formed.** Duplicate ids, a mix profile out of range, a
  style whose prompt would tell the generator to sing. These are edits, not
  code changes, which is exactly why nothing else would catch them.
* **A name always resolves.** Every path in and out of `find`/`clean_style`
  lands on a real style, because a job that has already cost a separation pass
  is not something to fail over a typo in a dropdown.
* **The prompt precedence.** Which of "what the user typed", "what the style
  says" and "what was measured" wins, in every combination.
"""

import pytest

from modal_app import styles

# --- the catalogue --------------------------------------------------------


def test_every_style_has_a_distinct_id():
    ids = [style.id for style in styles.STYLES]
    assert len(ids) == len(set(ids))
    assert styles.STYLE_IDS == tuple(ids)


def test_every_style_is_labelled_for_the_person_reading_it():
    """Vietnamese label, Vietnamese hint, on every entry. A style with no hint
    is a word in a list nobody can choose between."""
    for style in styles.STYLES:
        assert style.label.strip()
        assert style.hint.strip()


def test_the_default_style_exists_and_describes_nothing():
    """`auto` is the fallback for every unusable name, so it has to be the
    entry that adds nothing: no prompt, and the mix profile the app had before
    styles existed."""
    auto = styles.find(styles.DEFAULT_STYLE)
    assert auto.id == styles.DEFAULT_STYLE
    assert not auto.describes_sound()
    assert auto.mix == styles.NEUTRAL


def test_every_other_style_actually_describes_music():
    described = [s for s in styles.STYLES if s.id != styles.DEFAULT_STYLE]
    assert described, "a catalogue of one is not a catalogue"
    for style in described:
        assert style.describes_sound()


def test_no_style_asks_the_generator_for_a_singer():
    """The single worst failure this branch has available to it is a generated
    vocal underneath a converted one. `beatgen.PROMPT_SUFFIX` says "no vocals"
    on the way out; this is the other end of the same guarantee."""
    for style in styles.STYLES:
        text = style.prompt.lower()
        for word in ("vocal", "singer", "singing", "choir", "lyrics", "rapper"):
            assert word not in text, f"{style.id} asks for {word}"


def test_every_mix_profile_is_already_inside_its_own_ceilings():
    """`clamped()` exists so an edited table cannot reach somewhere
    unrecoverable — but a table that needs clamping is a table with a mistake in
    it, and this is where that is noticed."""
    for style in styles.STYLES:
        assert style.mix == style.mix.clamped(), style.id


def test_a_pocket_is_a_cut_and_never_a_boost():
    """A style asking to be *louder* where the voice lives has misunderstood
    what the control is; `clamped` enforces it and the table should not need it
    enforced."""
    for style in styles.STYLES:
        assert style.mix.pocket_db <= 0, style.id


def test_the_styles_disagree_about_the_mix_and_not_only_about_the_prompt():
    """The reason `Mixdown` is on the same record as the prompt. If every style
    shipped the same profile this would be a list of prompts with a mixing
    module attached for decoration."""
    profiles = {style.mix for style in styles.STYLES}
    assert len(profiles) > 4
    ducks = {style.mix.duck_db for style in styles.STYLES}
    assert max(ducks) - min(ducks) >= 3.0


def test_the_genres_that_bury_a_voice_duck_harder_than_the_ones_that_do_not():
    """Not a tuning assertion — a direction one. A wall of distorted guitar and
    a four-on-the-floor club mix have to get further out of the way than a
    bolero arrangement does, whatever the numbers end up being."""
    duck = {style.id: style.mix.duck_db for style in styles.STYLES}
    assert duck["rock"] > duck["bolero"]
    assert duck["trap"] > duck["ballad"]
    assert duck["edm"] > duck["acoustic"]


def test_the_style_built_on_an_808_keeps_its_low_end():
    """`beats.balance` levels a generated bed by RMS, which is the right rule
    and is also the one that would quietly turn a trap beat into a pop beat."""
    assert styles.find("trap").mix.low_shelf_db > 0


# --- resolving a name -----------------------------------------------------


@pytest.mark.parametrize("name", ["", None, "   ", "nonsense", "Trap\n", 7, {}])
def test_an_unusable_name_costs_a_style_and_never_a_job(name):
    """Clamped rather than refused, the same as every other named setting with
    a safe answer. By this point the job may already have paid for a separation
    pass."""
    assert styles.find(name).id in styles.STYLE_IDS
    assert styles.clean_style(name) in styles.STYLE_IDS


def test_casing_and_stray_whitespace_are_normalised_away():
    """A form field carries what a browser put in it. Case and padding are not
    the user's choices and should not cost them their style."""
    assert styles.clean_style("TRAP") == "trap"
    assert styles.clean_style(" LoFi ") == "lofi"
    assert styles.clean_style("Boombap\n") == "boombap"


def test_a_near_miss_is_not_guessed_at():
    """Normalising is not correcting. "lo-fi" is a different string from the id
    and the fallback is `auto`, which is honest — inventing a match would mean
    a style nobody picked deciding the mix."""
    assert styles.clean_style("lo-fi") == styles.DEFAULT_STYLE
    assert styles.clean_style("trapp") == styles.DEFAULT_STYLE


def test_every_id_in_the_catalogue_resolves_to_itself():
    for style in styles.STYLES:
        assert styles.find(style.id) is style
        assert styles.clean_style(style.id) == style.id


def test_the_catalogue_serialises_to_what_a_browser_needs():
    rows = styles.catalogue()
    assert [row["id"] for row in rows] == list(styles.STYLE_IDS)
    for row in rows:
        assert set(row) == {"id", "label", "hint"}
        assert all(isinstance(value, str) for value in row.values())


# --- the mix profile ------------------------------------------------------


def test_a_profile_out_of_range_is_pulled_back_rather_than_trusted():
    wild = styles.Mixdown(
        bed_below_voice_db=99, duck_db=99, pocket_db=99, pocket_hz=99999, low_shelf_db=-99
    )
    tame = wild.clamped()
    assert tame.bed_below_voice_db == styles.MAX_BED_LEVEL_DB
    assert tame.duck_db == styles.MAX_DUCK_DB
    # A boost asked for in the pocket becomes no pocket at all, never a boost.
    assert tame.pocket_db == 0.0
    assert tame.pocket_hz == 5000.0
    assert tame.low_shelf_db == -styles.MAX_SHELF_DB


def test_a_profile_that_cannot_be_read_as_numbers_does_not_raise():
    """`clamped` is the last thing between a table and a filter graph, and an
    exception there fails a job at the mix step, after every GPU pass has been
    paid for."""
    assert styles.Mixdown(duck_db="loud").clamped().duck_db == 0.0


def test_mix_for_is_find_plus_clamp():
    assert styles.mix_for("trap") == styles.find("trap").mix.clamped()
    assert styles.mix_for("nonsense") == styles.NEUTRAL.clamped()


# --- the prompt -----------------------------------------------------------


def test_what_the_user_typed_beats_the_style():
    """Somebody who wrote it has said something more specific than any preset,
    and a style appended to it would be the app arguing with them."""
    assert styles.prompt_for("trap", "guitar méo, trống thật") == "guitar méo, trống thật"


def test_the_style_speaks_when_the_box_is_empty():
    prompt = styles.prompt_for("lofi", "")
    assert "lo-fi" in prompt
    assert styles.ARRANGEMENT_SUFFIX in prompt


def test_the_measurement_speaks_when_neither_does():
    """The `auto` path, and the behaviour that existed before this module."""
    measured = "96 BPM, key of Am, drums and bass"
    assert styles.prompt_for("auto", "", measured) == measured


def test_a_prompt_is_never_three_descriptions_stacked_together():
    """A generator asked for a genre that does not exist returns music nobody
    wanted, and the job succeeded, which is the expensive way to find out."""
    measured = "96 BPM, key of Am, drums and bass"
    assert styles.prompt_for("trap", "acoustic guitar", measured) == "acoustic guitar"
    assert measured not in styles.prompt_for("trap", "", measured)


def test_whitespace_is_not_a_description():
    measured = "120 BPM, key of C, drums and bass"
    assert styles.prompt_for("auto", "   \n ", measured) == measured


def test_the_balance_says_which_genres_put_the_bed_in_front():
    """`bed_below_voice_db` is an absolute placement now, not a trim on whatever
    arrived — so the table can be read as a claim about the music, and this is
    that claim. A ballad and a bolero are about the words; trap and club music
    are about the bed and the voice rides on it."""
    level = {style.id: style.mix.bed_below_voice_db for style in styles.STYLES}
    assert level["ballad"] > 0 and level["bolero"] > 0 and level["acoustic"] > 0
    assert level["trap"] < 0 and level["edm"] < 0
    assert level["ballad"] > level["auto"] > level["trap"]


def test_the_styles_disagree_about_the_balance_too():
    levels = {style.mix.bed_below_voice_db for style in styles.STYLES}
    assert max(levels) - min(levels) >= 4.0
