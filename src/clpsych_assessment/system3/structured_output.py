from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ── Task 1.1: ABCD Element & Subelement Classification ──────────


class ElementClassification(BaseModel):
    """Classification for a single ABCD element"""

    element: Literal["A", "B-O", "B-S", "C-O", "C-S", "D"] = Field(
        ...,
        description="The ABCD element being classified",
    )
    subelement: int = Field(
        ...,
        ge=0,
        description=(
            "Subelement label: 0 = absent, "
            "1-K = specific subelement type"
        ),
    )
    evidence: str = Field(
        ...,
        description="Text excerpt from the post supporting this classification",
    )


class SelfStateClassification(BaseModel):
    """ABCD classification for one self-state (adaptive or maladaptive)"""

    elements: List[ElementClassification] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Classifications for all 6 ABCD elements",
    )


class ABCDClassificationResponse(BaseModel):
    """Task 1.1 response: ABCD element & subelement classification"""

    post_id: str = Field(
        ..., description="Identifier of the post being assessed"
    )
    adaptive_state: Optional[SelfStateClassification] = Field(
        None,
        description="ABCD classification for the adaptive self-state",
    )
    maladaptive_state: Optional[SelfStateClassification] = Field(
        None,
        description="ABCD classification for the maladaptive self-state",
    )


# ── Task 1.2: Presence Rating ───────────────────────────────────


class SelfStatePresence(BaseModel):
    """Presence rating for a single self-state"""

    presence_rating: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "Psychological centrality rating (1-5) indicating "
            "how central this self-state is to the post"
        ),
    )
    justification: str = Field(
        ...,
        description="Brief explanation of why this presence rating was assigned",
    )


class PresenceRatingResponse(BaseModel):
    """Task 1.2 response: presence rating per post"""

    post_id: str = Field(
        ..., description="Identifier of the post being assessed"
    )
    adaptive_state: Optional[SelfStatePresence] = Field(
        None,
        description="Presence rating for the adaptive self-state",
    )
    maladaptive_state: Optional[SelfStatePresence] = Field(
        None,
        description="Presence rating for the maladaptive self-state",
    )


# ── Task 2: Moments of Change ───────────────────────────────────


class MomentsOfChangeResponse(BaseModel):
    """Task 2 response: switch and escalation detection per post"""

    post_id: str = Field(
        ..., description="Identifier of the post being assessed"
    )
    switch: bool = Field(
        ...,
        description=(
            "Whether this post represents a switch — "
            "a distinct shift in the user's mental health trajectory"
        ),
    )
    escalation: bool = Field(
        ...,
        description=(
            "Whether this post represents an escalation — "
            "an intensification of the current mental health state"
        ),
    )
    justification: str = Field(
        ...,
        description="Brief explanation for the switch/escalation labels",
    )
