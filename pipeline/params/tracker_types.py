from enum import Enum


# Definitions of all implemented trackers. This enum class is used to determine which tracker to use in pipeline
class TRACKERS(Enum):
    AVTrack = "avtrack",
    VitTracker = "vittrack",
    ORTrack = "ortrack",
    MixFormerV2 = "mixformer_v2",
