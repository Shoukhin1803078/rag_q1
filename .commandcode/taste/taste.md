# Taste

## Workflow

- Prioritizes fast turnaround over exhaustive runs: when a job takes too long, cut its scope (e.g. reduce the sample size) so results come back quickly rather than waiting for the full run to finish. Confidence: 0.6
- Wants a written progress/status document (markdown) covering what's done, a summary, results, and what's next/remaining — as a standing deliverable alongside the code and reports. Confidence: 0.55
- Wants new experiments to mirror the structure of prior ones — when starting a related task, reuse the proven project layout and shared code rather than designing from scratch (e.g. "do similar experiment for X" / "correspondingly similar like previous"). Repeats this pattern across successive proposals (CHAB-RAG → PACER → SDM-MAR → DECAF), issuing the same "read this proposal, then do a correspondingly similar experiment" instruction each time. Confidence: 0.75

## Communication

- Wants to see the actual underlying data/artifacts, not just a prose description of them ("i want to see those data also otherwise how i know") — expects raw data surfaced (e.g. inspect the corpus, dump readable samples/examples) so claims can be verified directly. Confidence: 0.6

## Figures / Visualization

- Cares about clean, readable figures: no overlapping or crowded axis tick labels or point annotations. Expects label collisions to be fixed (e.g. horizontal bars so category names read as y-ticks, a shared legend instead of inline point annotations). Confidence: 0.7
