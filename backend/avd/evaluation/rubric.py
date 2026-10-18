"""Anchored rubric definitions shared by LLM prompts, human annotation guidelines and the UI.

Every dimension has a criterion-specific description for each of the five integer levels. Scores
are ordinal: a 4 means "matches the level-4 description", not "80 %". Displays that show 0-100 use
the documented transformation ``(score - 1) / 4 * 100`` and must not imply finer precision.

Mapping to commonly requested criteria (no separate dimensions, to keep the contract stable):
* completeness  -> ``depth`` ("depth and completeness relative to the stated scope")
* logical coherence -> ``structure`` ("logical structure and coherence of the argument")
* editing quality -> ``editing``; visual quality -> ``visual_quality``.
"""

from __future__ import annotations

from dataclasses import dataclass

from avd.evaluation.schema import RUBRIC_VERSION, Dimension, Modality


@dataclass(frozen=True)
class RubricItem:
    dimension: Dimension
    question: str
    anchors: dict[int, str]
    # Modalities that must be available (and authorized) to judge this dimension at all.
    requires: frozenset[Modality]
    # What does NOT count against this dimension (guards against cross-dimension leakage).
    exclusions: str = ""

    @property
    def anchor_1(self) -> str:
        return self.anchors[1]

    @property
    def anchor_3(self) -> str:
        return self.anchors[3]

    @property
    def anchor_5(self) -> str:
        return self.anchors[5]

    def anchor_for(self, score: float | None) -> str | None:
        if score is None:
            return None
        return self.anchors[max(1, min(5, round(score)))]


T = frozenset({Modality.TRANSCRIPT})
A = frozenset({Modality.AUDIO})
V = frozenset({Modality.VIDEO})


def _item(d: Dimension, q: str, a: list[str], req: frozenset[Modality], excl: str = "") -> RubricItem:
    assert len(a) == 5
    return RubricItem(d, q, {i + 1: text for i, text in enumerate(a)}, req, excl)


RUBRIC: dict[Dimension, RubricItem] = {
    item.dimension: item
    for item in [
        _item(
            Dimension.INFORMATION_DENSITY,
            "How much substantive, non-redundant information or meaningful content is delivered per minute?",
            [
                "Mostly filler, repetition, hype or self-promotion; little substance.",
                "Substance appears but is diluted by long tangents, padding or repetition.",
                "Some substance mixed with noticeable padding or repetition.",
                "Consistently substantive with only brief digressions.",
                "Dense and substantive throughout; nothing important could be cut.",
            ],
            T,
            "Do not penalise deliberate pauses or recaps that aid understanding.",
        ),
        _item(
            Dimension.DEPTH,
            "How deeply and completely does the video treat its subject, relative to its stated scope "
            "(mechanisms, reasons, caveats, coverage of essential sub-topics)?",
            [
                "Superficial; essential how/why and major parts of the stated scope are missing.",
                "Touches the topic but leaves important parts unexplained or omitted.",
                "Covers the basics with some explanation of mechanisms; some gaps in completeness.",
                "Thorough treatment of most of the scope with mechanisms and some caveats.",
                "Thorough and complete within scope: mechanisms, caveats, edge cases and context.",
            ],
            T,
            "Judge against the video's own stated scope; a short introductory video can score high.",
        ),
        _item(
            Dimension.FACTUAL_ACCURACY,
            "Are specific, checkable factual assertions correct? Judge only claims that can actually "
            "be assessed; ignore opinions, jokes, predictions, fiction and personal experiences.",
            [
                "Central claims are clearly wrong or misleading in consequential ways.",
                "Several notable factual errors or misleading simplifications.",
                "Mostly correct, with minor errors or imprecision.",
                "Accurate, with at most trivial imprecision.",
                "All assessable claims are accurate and appropriately qualified (uncertainty stated).",
            ],
            T,
            "Satire, fiction, opinion and clearly marked speculation are not factual errors.",
        ),
        _item(
            Dimension.ORIGINALITY,
            "Does the video offer an original perspective, idea, synthesis or creative execution?",
            [
                "Generic and derivative; interchangeable with countless others.",
                "Mostly conventional with a few personal touches.",
                "A competent personal take or some original elements.",
                "A distinctive angle, synthesis or execution in several places.",
                "Highly distinctive ideas, framing or creative execution throughout.",
            ],
            T,
        ),
        _item(
            Dimension.INFORMATIONAL_VALUE,
            "How much would a viewer in the target audience learn or durably gain from watching?",
            [
                "Little to nothing of lasting value.",
                "One or two minor takeaways.",
                "Several useful takeaways.",
                "Substantial, practical or conceptual value for the target audience.",
                "Exceptional value: changes how the viewer understands or does something.",
            ],
            T,
            "For entertainment profiles this dimension has low weight; do not penalise fun content.",
        ),
        _item(
            Dimension.CLARITY,
            "How clearly are ideas expressed (terms defined, examples, unambiguous language)?",
            [
                "Confusing, ambiguous or jargon-heavy without explanation.",
                "Often hard to follow; key terms undefined.",
                "Generally understandable with some unclear passages.",
                "Clear, with terms defined and helpful examples.",
                "Exceptionally clear; complex ideas made accessible without distortion.",
            ],
            T,
            "Accents, audio quality and visuals are not clarity issues unless they obscure meaning in the transcript.",
        ),
        _item(
            Dimension.STRUCTURE,
            "Is there a logical, coherent structure (setup, progression, signposting, conclusion) and "
            "an internally consistent line of argument?",
            [
                "Disorganised or self-contradictory; jumps between topics.",
                "Weak organisation; the thread is frequently lost.",
                "Recognisable structure with some meandering.",
                "Well organised and coherent with minor detours.",
                "Exemplary progression and coherence with clear signposting.",
            ],
            T,
            "Conversational formats (interviews, podcasts) are judged on coherence, not on formal outlines.",
        ),
        _item(
            Dimension.PACING,
            "Is the pacing appropriate for the content and audience (neither rushed nor dragging)?",
            [
                "Severely rushed or dragging; long stretches without progress.",
                "Frequently too fast or too slow.",
                "Acceptable with uneven sections.",
                "Well paced with brief uneven moments.",
                "Well-judged pacing throughout.",
            ],
            T,
        ),
        _item(
            Dimension.STORYTELLING,
            "How effectively does the video use narrative, tension, motivation or engagement "
            "techniques in service of its content?",
            [
                "No narrative thread; or manipulative hooks without payoff.",
                "Weak framing; engagement devices feel forced or unrelated.",
                "Some narrative framing or engaging moments.",
                "Engaging narrative that mostly serves the content.",
                "Compelling narrative that serves the content throughout.",
            ],
            T,
        ),
        _item(
            Dimension.TITLE_CONSISTENCY,
            "Does the content deliver what the title and description promise?",
            [
                "Clickbait: the promised content is missing or misrepresented.",
                "Substantially overpromises; delivers only part of the promise.",
                "Partially delivers; some overpromising.",
                "Delivers the promise with minor exaggeration.",
                "Fully and accurately delivers what the title and description promise.",
            ],
            T,
        ),
        _item(
            Dimension.AUDIO_QUALITY,
            "Is the audio clean and intelligible (noise, clipping, levels, dropouts)?",
            [
                "Hard to understand: heavy noise, clipping, dropouts or very low level.",
                "Frequently impaired; listening requires effort.",
                "Intelligible with noticeable issues.",
                "Clean with minor issues.",
                "Clean, consistent, well-levelled audio.",
            ],
            A,
            "Signal metrics measure level/noise/distortion only, not voice or music quality.",
        ),
        _item(
            Dimension.VISUAL_QUALITY,
            "Image quality: resolution, focus, exposure, stability.",
            [
                "Visually poor to the point of distraction (blur, severe exposure problems).",
                "Frequently soft, dark/blown-out or unstable.",
                "Adequate visuals.",
                "Good visual quality with minor issues.",
                "High visual quality throughout.",
            ],
            V,
            "Never inferred from the title, description or popularity.",
        ),
        _item(
            Dimension.EDITING,
            "Editing quality: cuts, transitions, removal of dead air, supporting graphics.",
            [
                "Unedited or distracting edits.",
                "Rough editing that often gets in the way.",
                "Competent editing.",
                "Clean editing that supports the content.",
                "Excellent editing that clearly serves the content.",
            ],
            V,
        ),
        _item(
            Dimension.VISUAL_READABILITY,
            "Are on-screen text, diagrams, code or slides legible and well designed?",
            [
                "Illegible or cluttered visual information.",
                "Often hard to read (size, contrast, time on screen).",
                "Mostly legible.",
                "Legible and reasonably well designed.",
                "Clear, well-designed visual information throughout.",
            ],
            V,
        ),
    ]
}

assert set(RUBRIC) == set(Dimension), "every dimension needs a rubric item"


def rubric_text(dimensions: list[Dimension]) -> str:
    lines = []
    for d in dimensions:
        r = RUBRIC[d]
        anchors = "\n".join(f"    {k} = {v}" for k, v in sorted(r.anchors.items()))
        excl = f"\n    Note: {r.exclusions}" if r.exclusions else ""
        lines.append(f"- {d.value}: {r.question}\n{anchors}{excl}")
    return "\n".join(lines)


def rubric_document() -> dict[str, object]:
    """Serializable rubric (served by /v1/meta and used by the annotation UI)."""
    return {
        "version": RUBRIC_VERSION,
        "scale": "ordinal 1-5; display transform 0-100 = (score - 1) / 4 * 100",
        "dimensions": [
            {
                "dimension": d.value,
                "group": d.group.value,
                "question": r.question,
                "anchors": {str(k): v for k, v in sorted(r.anchors.items())},
                "exclusions": r.exclusions,
                "requires": sorted(m.value for m in r.requires),
            }
            for d, r in RUBRIC.items()
        ],
    }
