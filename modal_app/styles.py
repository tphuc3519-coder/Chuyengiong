"""The beat styles: what a "mode" is, in one place.

    "Trap" ──┬─► prompt cho máy sinh nhạc  ──► beatgen
             └─► cách đặt beat dưới giọng ──► mixing.mix

Every other module in the beat path takes audio and gives audio back. This one
takes a **name** and gives back the two settings that name implies, which is
the whole content of the word "mode" as the AI song apps use it: pick *Lo-fi*
and you are not picking a filter, you are picking a set of instruments and a
way of sitting them under a voice.

**Why the two halves are in the same record.** A style is not a prompt. Trap
and a piano ballad want different instruments *and* different mixes — a trap
bed is mostly sub and has to move out of the way of every syllable, a ballad
bed is mostly midrange and must not be seen to duck at all. Shipping the prompt
without the mix produces the failure this module exists to prevent: the right
instruments, mixed the one way the app knows, so every style sounds like the
same song with different sounds on it.

**What a style deliberately does not control.** Tempo and key. Those are
measured off the song by `analysis` and applied by `beats.fit`, and a style
that overrode them would be a style that can put a beat in the wrong key. The
one place tempo *feel* appears here is in the prompt text — "half-time",
"double-time hi-hats" — because asking the generator for a groove costs
nothing, while stretching a finished loop to twice its length costs the loop.
`beats.fold_tempo` is the reason: it treats a factor of two as free, so a
half-time bed fits at 1.0x and a stretched one would not.

**Where the numbers come from, honestly.** The prompts are written; the mix
profiles are *reasoned*, not measured. Each one is a small departure from
`NEUTRAL`, in the direction the genre is known to want, and bounded by
`mixing.MAX_*` so no style can be dialled somewhere unrecoverable. They need an
ear on a real job exactly like `beats.LOW_SHARE_TARGET` and `beatgen`'s two
strengths do, and they are constants in one table so that ear has one file to
change.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- how a bed sits under a voice ----------------------------------------

# Ceilings, so a style is a choice inside a range rather than a free hand on
# the mix. Each is placed where the effect stops being an arrangement decision
# and starts being damage:
#
#  * 9 dB of ducking is already a pumping effect rather than space-making;
#  * a bell deeper than 6 dB in the vocal band leaves a hole audible when
#    nobody is singing;
#  * ±6 dB of shelf is the range in which a bed still sounds like the record
#    somebody made.
MAX_DUCK_DB = 9.0
MAX_POCKET_DB = 6.0
MAX_SHELF_DB = 6.0
MAX_BED_LEVEL_DB = 6.0


@dataclass(frozen=True)
class Mixdown:
    """How much room the bed gives the voice, and where.

    Four numbers, and each answers a question a mixing engineer answers by
    hand:

    * `bed_below_voice_db` — how far the bed's own loudness sits below the
      voice's, in dB. The blunt one, the one a listener notices first, and the
      one that used to be impossible to set meaningfully: it was a gain applied
      to whatever level happened to arrive, so the same style landed somewhere
      different for a quiet upload and a loud one. `mixing.mix` now measures
      both sides and places the bed, so this number is a **balance** rather
      than a trim — the same style produces the same mix whatever it is handed.
      Positive means the bed is quieter than the voice; negative means the bed
      is the point and the voice rides on it, which is the honest description
      of trap and of club music.
    * `duck_db` — how far the voice pushes the bed down *while it is singing*.
      This is the setting that makes a modern record sound modern: the bed is
      loud in the gaps and gets out of the way for the words, so nothing has to
      be quiet all the way through to stay intelligible. 0 switches the
      sidechain off entirely rather than applying a null one.
    * `pocket_db` / `pocket_hz` — a bell cut in the bed where consonants live.
      Static, unlike the duck: it carves the *place* the voice occupies rather
      than the moments it occupies it, and the two together are what "sits in
      the mix" means. Negative is a cut, which is the only direction that makes
      sense here; a positive value would be the bed competing on purpose.
    * `low_shelf_db` — the bed's own bottom end. Up for anything built on an
      808, down for anything whose weight would otherwise fight a low voice.

    Only the level is absolute. The three filters are relative to the bed as it
    arrived, because there is nothing to compare a shelf against. A style never
    touches the voice — `enhance` owns that chain and the user owns its one
    slider.
    """

    bed_below_voice_db: float = 1.0
    duck_db: float = 3.0
    pocket_db: float = -2.0
    pocket_hz: float = 2400.0
    low_shelf_db: float = 0.0

    def clamped(self) -> Mixdown:
        """The same profile with every number inside its ceiling.

        Called by `mixing` rather than trusted from the table, for the same
        reason `pipeline.clean_params` clamps a slider it also bounds in the
        UI: the table is data, and data that has been edited is not checked by
        anything else.
        """
        return Mixdown(
            bed_below_voice_db=_clamp(self.bed_below_voice_db, -MAX_BED_LEVEL_DB, MAX_BED_LEVEL_DB),
            duck_db=_clamp(self.duck_db, 0.0, MAX_DUCK_DB),
            # Cuts only. A style asking to be *louder* where the voice is has
            # misunderstood what this control is.
            pocket_db=_clamp(self.pocket_db, -MAX_POCKET_DB, 0.0),
            # Where a voice's intelligibility lives. Outside this the bell is
            # not a pocket, it is a tone control with the wrong name.
            pocket_hz=_clamp(self.pocket_hz, 800.0, 5000.0),
            low_shelf_db=_clamp(self.low_shelf_db, -MAX_SHELF_DB, MAX_SHELF_DB),
        )


def _clamp(value: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return low


# The profile every style is a departure from, and the one `song` mode would
# get if it asked: a bed placed 1 dB under the voice, 3 dB out of the way while
# somebody sings, and a shallow bell where the consonants are. Nothing else.
NEUTRAL = Mixdown()


@dataclass(frozen=True)
class Style:
    """One entry in the list the user picks from.

    `prompt` is English because that is what the generator was trained on;
    `label` and `hint` are Vietnamese because that is who reads them. Keeping
    both on the same record is what stops the list in the UI and the list the
    generator sees from drifting apart — there is one list.

    `prompt` is empty on exactly one style (`auto`), and that emptiness is the
    feature: it means "say nothing, let `beatgen.describe` write the prompt from
    what was measured off the song". Every other style names instruments.
    """

    id: str
    label: str
    hint: str
    prompt: str
    mix: Mixdown = NEUTRAL

    def describes_sound(self) -> bool:
        """Whether this style has anything to tell the generator.

        False only for `auto`, and the caller that asks is `prompt_for`.
        """
        return bool(self.prompt)


# Appended to whatever the style or the user asked for, on every generated
# beat. `beatgen.PROMPT_SUFFIX` says the same thing and says it last; this one
# is about arrangement rather than about vocals — a bed for somebody else's
# voice wants space in the middle, and asking for it costs one clause.
ARRANGEMENT_SUFFIX = "clean mix, space for a lead vocal"


# The catalogue.
#
# Ordered the way it is read rather than alphabetically: `auto` first because
# it is the answer for somebody who does not want to choose, then the four
# styles this app is most often pointed at (Vietnamese pop and rap), then the
# rest. A list and not a dict, because the order is part of the data.
STYLES: tuple[Style, ...] = (
    Style(
        id="auto",
        label="Theo bài",
        hint="Máy tự chọn theo tông và tốc độ đo được",
        # Deliberately empty — see `Style.prompt`.
        prompt="",
        mix=NEUTRAL,
    ),
    Style(
        id="ballad",
        label="Ballad piano",
        hint="Piano, đệm dây, trống chổi — hợp bài tình ca",
        prompt=(
            "emotional pop ballad, grand piano, warm string pad, brushed drums, "
            "soft fretless bass, half-time feel"
        ),
        # A ballad bed is midrange and it is the arrangement, not a backdrop:
        # duck it hard and the piano audibly breathes with every line. Low
        # shelf down a touch because a piano's own bottom octave and a low
        # voice occupy the same place.
        mix=Mixdown(
            bed_below_voice_db=3.0, duck_db=1.5, pocket_db=-1.5, pocket_hz=2200.0, low_shelf_db=-1.0
        ),
    ),
    Style(
        id="lofi",
        label="Lo-fi chill",
        hint="Trống mềm, piano điện, tiếng băng cũ",
        prompt=(
            "lo-fi hip hop, soft tape-saturated drums, mellow electric piano, "
            "round sub bass, vinyl crackle, relaxed swing"
        ),
        # The genre is a texture and the voice is the event: a lo-fi bed can
        # sit loud and duck a long way without anybody hearing it move, because
        # nothing in it has a sharp transient to give the movement away.
        mix=Mixdown(bed_below_voice_db=0.0, duck_db=4.0, pocket_db=-2.5, pocket_hz=2600.0),
    ),
    Style(
        id="trap",
        label="Trap",
        hint="808 nặng, hi-hat rải nhanh, phím tối",
        prompt=(
            "trap, booming 808 sub bass, crisp rolling hi-hats, sparse dark keys, "
            "hard snare on the backbeat, double-time hi-hats"
        ),
        # The one style where the bed is genuinely bigger than the voice, and
        # therefore the one that needs the most ducking to stay intelligible.
        # The shelf goes *up*: an 808 that has been levelled by RMS is an 808
        # that is no longer the point.
        mix=Mixdown(
            bed_below_voice_db=-2.0, duck_db=6.0, pocket_db=-3.5, pocket_hz=2800.0, low_shelf_db=2.5
        ),
    ),
    Style(
        id="boombap",
        label="Boom bap",
        hint="Trống mộc bụi bặm, bass gỗ, piano jazz",
        prompt=(
            "boom bap hip hop, dusty acoustic drum break, upright bass, "
            "jazzy piano chops, vinyl warmth, head-nodding groove"
        ),
        mix=Mixdown(
            bed_below_voice_db=-0.5, duck_db=4.0, pocket_db=-2.5, pocket_hz=2500.0, low_shelf_db=1.0
        ),
    ),
    Style(
        id="bolero",
        label="Bolero",
        hint="Guitar thùng, organ nhẹ, nhịp rumba chậm",
        prompt=(
            "slow bolero, nylon string guitar, gentle electric organ, "
            "brushed rumba percussion, upright bass, warm and nostalgic"
        ),
        # An arrangement people listen *to*, on records where the singer is
        # already forward. Almost no duck, and no shelving: leave it as the
        # arrangement it is.
        mix=Mixdown(bed_below_voice_db=2.5, duck_db=1.5, pocket_db=-1.5, pocket_hz=2200.0),
    ),
    Style(
        id="rnb",
        label="R&B",
        hint="Trống chậm, Rhodes ấm, bass tròn",
        prompt=(
            "contemporary r&b, laid-back drums, warm rhodes chords, round bass, "
            "subtle clean guitar, airy background texture"
        ),
        mix=Mixdown(
            bed_below_voice_db=0.5, duck_db=4.0, pocket_db=-3.0, pocket_hz=2600.0, low_shelf_db=1.0
        ),
    ),
    Style(
        id="acoustic",
        label="Mộc",
        hint="Guitar gảy, cajon, bass gỗ — không synth",
        prompt=(
            "acoustic arrangement, fingerpicked steel string guitar, cajon, "
            "light double bass, room ambience, no synths"
        ),
        # Nothing here is loud and everything here is midrange, which is the
        # hard case: the pocket does the work and the duck stays small so the
        # guitar does not sound like it is being switched on and off.
        mix=Mixdown(
            bed_below_voice_db=2.5, duck_db=2.0, pocket_db=-2.5, pocket_hz=2200.0, low_shelf_db=-1.0
        ),
    ),
    Style(
        id="rock",
        label="Rock band",
        hint="Guitar méo, trống thật, bass chạy",
        prompt=(
            "rock band, distorted electric guitars, live drum kit, driving bass guitar, "
            "energetic and loud"
        ),
        # Distorted guitar is a wall in exactly the band a voice needs, so this
        # gets the deepest pocket in the table and a wide one by implication —
        # it is the genre where "the vocal is buried" is the default outcome.
        mix=Mixdown(bed_below_voice_db=-1.0, duck_db=5.0, pocket_db=-4.0, pocket_hz=2400.0),
    ),
    Style(
        id="edm",
        label="EDM / House",
        hint="Kick đều bốn nhịp, bass ấm, synth sáng",
        prompt=(
            "house, four-on-the-floor kick, warm analog bass, bright plucked synths, "
            "airy pads, club mix"
        ),
        # The genre that invented sidechain ducking, and the only one where a
        # listener hears the pumping as part of the music rather than as a
        # mistake.
        mix=Mixdown(
            bed_below_voice_db=-2.0, duck_db=6.0, pocket_db=-3.0, pocket_hz=2800.0, low_shelf_db=1.5
        ),
    ),
    Style(
        id="citypop",
        label="City pop",
        hint="Bass slap, guitar chorus, phím điện sáng",
        prompt=(
            "city pop funk, slap bass, clean chorus electric guitar, "
            "bright electric piano, tight drums, 80s production"
        ),
        mix=Mixdown(bed_below_voice_db=0.0, duck_db=3.5, pocket_db=-3.0, pocket_hz=2600.0),
    ),
    Style(
        id="orchestral",
        label="Bán cổ điển",
        hint="Dàn dây, piano, kèn — kiểu nhạc phim",
        prompt=(
            "cinematic orchestral, sustained string section, soft piano, "
            "french horns, timpani, wide and slow"
        ),
        # Sustained strings never stop, so there is no gap for the voice to
        # appear in and the duck has to make one. The pocket is shallow because
        # a bell in a string section is audible as a hole.
        mix=Mixdown(
            bed_below_voice_db=1.5, duck_db=5.0, pocket_db=-2.0, pocket_hz=2400.0, low_shelf_db=-1.0
        ),
    ),
)

DEFAULT_STYLE = "auto"

_BY_ID = {style.id: style for style in STYLES}

# Every id, for the places that only need to check membership.
STYLE_IDS: tuple[str, ...] = tuple(style.id for style in STYLES)


def find(style_id: str | None) -> Style:
    """The style for `style_id`, falling back to `auto` rather than raising.

    Clamped and not refused, the same way `pipeline.clean_params` treats every
    other named setting with a safe answer: an old client, or a typo, costs the
    user the style they asked for and not the job they paid a GPU for. And the
    fallback is the one that adds nothing — `auto` writes no prompt and mixes
    neutrally, so a name nobody recognises produces the behaviour this app had
    before styles existed.
    """
    return _BY_ID.get(clean_style(style_id), _BY_ID[DEFAULT_STYLE])


def clean_style(style_id: str | None) -> str:
    """A style id from whatever arrived, or `auto`."""
    name = str(style_id or "").strip().lower()
    return name if name in _BY_ID else DEFAULT_STYLE


def prompt_for(style_id: str | None, typed: str = "", described: str = "") -> str:
    """What the generator is actually asked for. The one place the three meet.

    Precedence, and each step of it is a decision:

    1. **What the user typed wins.** Somebody who wrote "guitar méo, trống
       thật" has said something more specific than any preset, and a style
       silently appended to it would be the app arguing with them.
    2. **Otherwise the style speaks**, because that is what picking one means.
    3. **Otherwise whatever was measured** — `beatgen.describe(track)`, which
       names the tempo and the key and stops there. This is the `auto` path and
       it is the behaviour that existed before this module.

    Never a concatenation of all three. A prompt is a description of one piece
    of music, and three descriptions of it stacked into one string is how a
    generator is asked for a genre that does not exist.
    """
    typed = (typed or "").strip()
    if typed:
        return typed
    style = find(style_id)
    if style.describes_sound():
        return f"{style.prompt}, {ARRANGEMENT_SUFFIX}"
    return (described or "").strip()


def mix_for(style_id: str | None) -> Mixdown:
    """The mix profile for a style, already clamped. What `mixing.mix` takes."""
    return find(style_id).mix.clamped()


def catalogue() -> list[dict]:
    """The list as JSON, for `/health` to hand the browser.

    The UI mirrors this table in `web/lib/params.ts` the way it mirrors every
    other list of options — but a mirror is a copy, and a copy of a list of
    twelve entries with Vietnamese labels is a copy that will be edited on one
    side only. Serving it means a deployment that has grown a style can say so,
    and a browser that has not been redeployed still renders the right names.
    """
    return [{"id": style.id, "label": style.label, "hint": style.hint} for style in STYLES]
