"""Anchored rubric definitions shared by LLM prompts, human annotation guidelines and the UI."""

from __future__ import annotations

from dataclasses import dataclass

from avd.evaluation.schema import Dimension, Modality


@dataclass(frozen=True)
class RubricItem:
    dimension: Dimension
    question: str
    anchor_1: str
    anchor_3: str
    anchor_5: str
    # Modalities that must be available (and authorized) to judge this dimension at all.
    requires: frozenset[Modality]


T = frozenset({Modality.TRANSCRIPT})

RUBRIC: dict[Dimension, RubricItem] = {
    item.dimension: item
    for item in [
        RubricItem(
            Dimension.INFORMATION_DENSITY,
            "How much substantive, non-redundant information or meaningful content is delivered per minute?",
            "Mostly filler, repetition, hype or self-promotion.",
            "Some substance mixed with noticeable padding.",
            "Consistently substantive; little to no padding.",
            T,
        ),
        RubricItem(
            Dimension.DEPTH,
            "How deeply and completely does the video treat its subject, given its stated scope?",
            "Superficial; skips the essential how/why.",
            "Covers the basics with some explanation of mechanisms or reasons.",
            "Thorough within scope: mechanisms, caveats, edge cases, context.",
            T,
        ),
        RubricItem(
            Dimension.FACTUAL_ACCURACY,
            "Are checkable factual claims correct? Only judge claims you can actually assess.",
            "Contains clear, consequential factual errors.",
            "Mostly correct with minor errors or imprecision.",
            "All assessable claims are accurate and appropriately qualified.",
            T,
        ),
        RubricItem(
            Dimension.ORIGINALITY,
            "Does the video offer an original perspective, idea, creative execution or synthesis?",
            "Generic, derivative, interchangeable with countless others.",
            "Some original elements or a competent personal take.",
            "Distinctive ideas, framing or creative execution.",
            T,
        ),
        RubricItem(
            Dimension.INFORMATIONAL_VALUE,
            "How much would a viewer in the target audience learn or gain from watching?",
            "Little to nothing of lasting value.",
            "Some useful takeaways.",
            "Substantial, durable value for the target audience.",
            T,
        ),
        RubricItem(
            Dimension.CLARITY,
            "How clearly are ideas expressed (terminology defined, examples, unambiguous language)?",
            "Confusing, ambiguous or jargon-heavy without explanation.",
            "Generally understandable with some unclear passages.",
            "Exceptionally clear; complex ideas made accessible without distortion.",
            T,
        ),
        RubricItem(
            Dimension.STRUCTURE,
            "Is there a logical structure (setup, progression, signposting, conclusion)?",
            "Disorganised; jumps between topics.",
            "Recognisable structure with some meandering.",
            "Well-organised progression with clear signposting.",
            T,
        ),
        RubricItem(
            Dimension.PACING,
            "Is the pacing appropriate for the content and audience (neither rushed nor dragging)?",
            "Severely rushed or dragging; long stretches without progress.",
            "Acceptable with uneven sections.",
            "Well-judged pacing throughout.",
            T,
        ),
        RubricItem(
            Dimension.STORYTELLING,
            "How effectively does the video use narrative, tension, motivation or engagement "
            "techniques in service of its content?",
            "No narrative thread or engagement; or manipulative hooks without payoff.",
            "Some narrative framing or engaging moments.",
            "Compelling narrative that serves the content.",
            T,
        ),
        RubricItem(
            Dimension.TITLE_CONSISTENCY,
            "Does the content deliver what the title and description promise?",
            "Clickbait: the promised content is missing or misrepresented.",
            "Partially delivers; some overpromising.",
            "Fully delivers what the title and description promise.",
            T,
        ),
        RubricItem(
            Dimension.AUDIO_QUALITY,
            "Is the audio clean and intelligible (noise, clipping, levels)?",
            "Hard to understand: heavy noise, clipping or very low level.",
            "Intelligible with noticeable issues.",
            "Clean, well-levelled audio.",
            frozenset({Modality.AUDIO}),
        ),
        RubricItem(
            Dimension.VISUAL_QUALITY,
            "Image quality: resolution, focus, exposure, stability.",
            "Visually poor to the point of distraction.",
            "Adequate visuals.",
            "High visual quality.",
            frozenset({Modality.VIDEO}),
        ),
        RubricItem(
            Dimension.EDITING,
            "Editing quality: cuts, transitions, removal of dead air, supporting graphics.",
            "Unedited or distracting edits.",
            "Competent editing.",
            "Excellent editing that serves the content.",
            frozenset({Modality.VIDEO}),
        ),
        RubricItem(
            Dimension.VISUAL_READABILITY,
            "Are on-screen text, diagrams, code or slides legible and well designed?",
            "Illegible or cluttered visual information.",
            "Mostly legible.",
            "Clear, well-designed visual information.",
            frozenset({Modality.VIDEO}),
        ),
    ]
}

assert set(RUBRIC) == set(Dimension), "every dimension needs a rubric item"


def rubric_text(dimensions: list[Dimension]) -> str:
    lines = []
    for d in dimensions:
        r = RUBRIC[d]
        lines.append(
            f"- {d.value}: {r.question}\n    1 = {r.anchor_1}\n    3 = {r.anchor_3}\n    5 = {r.anchor_5}"
        )
    return "\n".join(lines)
