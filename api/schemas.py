from typing import List
from pydantic import BaseModel, Field

class FrameKeypointsPayload(BaseModel):
    # List of 53 joints, each containing [x, y, z] coordinates
    landmarks: List[List[float]] = Field(..., description="Shape (53, 3) normalized joint coordinates")

class TranslationResponse(BaseModel):
    predicted_sign: str
    confidence: float
    latency_ms: float
    buffer_fill: int