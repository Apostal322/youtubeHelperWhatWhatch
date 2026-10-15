"""Genre-specific, transparent aggregation profiles.

A profile assigns each dimension a weight *within its group* (0 = not applicable for the genre) and
assigns each group a weight in the overall summary. Profiles are plain data, versioned with
``FRAMEWORK_VERSION``, and returned verbatim by the API so users can see exactly how a summary was
formed.

Design principle: technical quality never leaks into content scores. A scientifically excellent
lecture with poor audio keeps its content score; only the technical group (and, with a small
weight, the overall summary) reflects production quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from avd.evaluation.schema import Dimension, DimensionGroup, Genre

D = Dimension
G = DimensionGroup


@dataclass(frozen=True)
class GenreProfile:
    genre: Genre
    description: str
    dimension_weights: dict[Dimension, float]
    group_weights: dict[DimensionGroup, float]
    # Guidance injected into LLM prompts so judgements respect genre conventions.
    guidance: str
    notes: list[str] = field(default_factory=list)

    def weight(self, d: Dimension) -> float:
        return self.dimension_weights.get(d, 0.0)

    def applicable(self, d: Dimension) -> bool:
        return self.weight(d) > 0.0

    def applicable_dimensions(self) -> list[Dimension]:
        return [d for d in Dimension if self.applicable(d)]

    def as_dict(self) -> dict[str, object]:
        return {
            "genre": self.genre.value,
            "description": self.description,
            "dimension_weights": {d.value: w for d, w in self.dimension_weights.items()},
            "group_weights": {g.value: w for g, w in self.group_weights.items()},
            "guidance": self.guidance,
        }


def _weights(**kw: float) -> dict[Dimension, float]:
    base = {d: 1.0 for d in Dimension}
    base.update({Dimension(k): v for k, v in kw.items()})
    return base


PROFILES: dict[Genre, GenreProfile] = {
    p.genre: p
    for p in [
        GenreProfile(
            Genre.EDUCATION,
            "Lectures, explainers and courses: explanatory depth and accuracy dominate.",
            _weights(
                depth=1.3,
                factual_accuracy=1.5,
                originality=0.5,
                informational_value=1.3,
                clarity=1.5,
                structure=1.2,
                pacing=0.8,
                storytelling=0.4,
                visual_quality=0.4,
                editing=0.3,
            ),
            {G.CONTENT: 0.6, G.PRESENTATION: 0.3, G.TECHNICAL: 0.1},
            "Prioritise correctness, explanatory depth and clarity. Do not penalise plain or "
            "low-budget production in content or presentation dimensions.",
        ),
        GenreProfile(
            Genre.SCIENCE,
            "Science communication and lectures: accuracy and depth dominate.",
            _weights(
                depth=1.5,
                factual_accuracy=2.0,
                originality=0.6,
                informational_value=1.3,
                clarity=1.4,
                structure=1.1,
                pacing=0.7,
                storytelling=0.5,
                visual_quality=0.4,
                editing=0.3,
            ),
            {G.CONTENT: 0.6, G.PRESENTATION: 0.3, G.TECHNICAL: 0.1},
            "Scientific accuracy and depth of explanation matter most. Flag overclaiming, missing "
            "caveats and confidently stated errors. Production quality is secondary.",
        ),
        GenreProfile(
            Genre.TECHNOLOGY,
            "Technology explainers, reviews and tutorials.",
            _weights(
                information_density=1.2,
                depth=1.2,
                factual_accuracy=1.3,
                originality=0.7,
                informational_value=1.3,
                clarity=1.3,
                storytelling=0.4,
                visual_readability=1.2,
            ),
            {G.CONTENT: 0.55, G.PRESENTATION: 0.3, G.TECHNICAL: 0.15},
            "Value technical correctness, concrete detail and practical usefulness over hype.",
        ),
        GenreProfile(
            Genre.HOWTO,
            "Tutorials and how-to guides: actionable, correct, easy to follow.",
            _weights(
                depth=0.8,
                originality=0.3,
                informational_value=1.5,
                clarity=1.5,
                structure=1.3,
                storytelling=0.2,
                visual_readability=1.2,
                editing=0.6,
            ),
            {G.CONTENT: 0.5, G.PRESENTATION: 0.35, G.TECHNICAL: 0.15},
            "Judge whether a viewer could actually follow the steps and whether the instructions "
            "are correct and complete.",
        ),
        GenreProfile(
            Genre.NEWS_COMMENTARY,
            "News analysis and commentary.",
            _weights(
                factual_accuracy=2.0,
                originality=0.8,
                storytelling=0.6,
                title_consistency=1.5,
                visual_quality=0.4,
                editing=0.4,
            ),
            {G.CONTENT: 0.6, G.PRESENTATION: 0.3, G.TECHNICAL: 0.1},
            "Separate reporting from opinion. Judge sourcing, accuracy of checkable claims and "
            "whether the title fairly represents the content. Do not score political stance.",
        ),
        GenreProfile(
            Genre.DOCUMENTARY,
            "Documentaries and long-form investigations.",
            _weights(
                information_density=0.8,
                depth=1.2,
                factual_accuracy=1.5,
                storytelling=1.5,
                visual_quality=1.2,
                visual_readability=0.5,
            ),
            {G.CONTENT: 0.45, G.PRESENTATION: 0.35, G.TECHNICAL: 0.2},
            "Balance accuracy and depth with narrative craft.",
        ),
        GenreProfile(
            Genre.ENTERTAINMENT,
            "Comedy, sketches, vlogs, challenges and other entertainment.",
            _weights(
                information_density=0.3,
                depth=0.3,
                factual_accuracy=0.0,
                originality=1.5,
                informational_value=0.3,
                clarity=0.7,
                structure=0.7,
                pacing=1.3,
                storytelling=1.5,
                editing=1.2,
                visual_readability=0.3,
            ),
            {G.CONTENT: 0.35, G.PRESENTATION: 0.4, G.TECHNICAL: 0.25},
            "Judge originality, execution, comedic or narrative craft and pacing. Do not penalise "
            "the absence of educational content.",
        ),
        GenreProfile(
            Genre.GAMING,
            "Let's plays, game analysis, speedruns and gaming commentary.",
            _weights(
                information_density=0.5,
                depth=0.7,
                factual_accuracy=0.3,
                originality=1.3,
                informational_value=0.7,
                clarity=0.7,
                structure=0.6,
                pacing=1.2,
                storytelling=1.2,
                visual_readability=0.5,
            ),
            {G.CONTENT: 0.35, G.PRESENTATION: 0.4, G.TECHNICAL: 0.25},
            "Judge commentary value, insight into the game, creativity and entertainment "
            "execution. Pure gameplay without commentary has limited originality.",
        ),
        GenreProfile(
            Genre.ART,
            "Visual art, craft, animation and creative process videos.",
            _weights(
                information_density=0.3,
                depth=0.8,
                factual_accuracy=0.0,
                originality=1.8,
                informational_value=0.6,
                clarity=0.6,
                structure=0.6,
                pacing=0.8,
                title_consistency=0.8,
                audio_quality=0.8,
                visual_quality=1.5,
                editing=0.8,
                visual_readability=0.3,
            ),
            {G.CONTENT: 0.45, G.PRESENTATION: 0.3, G.TECHNICAL: 0.25},
            "Prioritise creative originality and the insight offered into the craft.",
        ),
        GenreProfile(
            Genre.MUSIC,
            "Music performances, compositions and music analysis.",
            _weights(
                information_density=0.0,
                depth=0.6,
                factual_accuracy=0.0,
                originality=2.0,
                informational_value=0.4,
                clarity=0.3,
                structure=0.8,
                pacing=0.8,
                storytelling=0.8,
                title_consistency=0.6,
                audio_quality=2.0,
                visual_quality=0.6,
                editing=0.6,
                visual_readability=0.0,
            ),
            {G.CONTENT: 0.45, G.PRESENTATION: 0.2, G.TECHNICAL: 0.35},
            "A transcript captures lyrics and spoken parts only; abstain on what cannot be judged "
            "from text. Audio quality matters strongly when authorized audio is available.",
        ),
        GenreProfile(
            Genre.OTHER,
            "Fallback profile with uniform weights.",
            _weights(),
            {G.CONTENT: 0.5, G.PRESENTATION: 0.35, G.TECHNICAL: 0.15},
            "Apply the rubric literally.",
        ),
    ]
}

assert set(PROFILES) == set(Genre), "every genre needs a profile"


def get_profile(genre: Genre) -> GenreProfile:
    return PROFILES[genre]
