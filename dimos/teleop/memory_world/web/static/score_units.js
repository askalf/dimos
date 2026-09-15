// What a score on the wire actually MEANS, and how to say it.
//
// `ClusterSummary.score` has already meant two incomparable things once: hyperspace
// scores a voxel in [0, 1] and the embedding engine scores a place by cosine similarity
// in [-1, 1], and bounds written for the first rejected every answer from the second.
// It now means three, because `engine="hyperspace"` covers BOTH an item answer -- whose
// number is the detector's calibrated box confidence -- and a heatmap or area answer,
// whose number is a cell's score on a different scale entirely.
//
// So `engine` alone does not say whether two numbers may be compared, and neither does
// `kind`. The PAIR does. Every function here takes both, which is the point: formatting
// or comparing a score without saying where it came from should be awkward to write,
// because "never compare across answers" is an invariant that lives nowhere and is
// enforced by nothing. It is true today only because `_publish_query_result` carries
// `action="replace"` -- a rendering decision, not a statement about scores. The first
// history pane or "best so far" badge breaks it without touching either file.

/** A stable name for the scale a number is on. Two answers may be compared only if these
 *  are equal; the string itself is never shown. */
export function scaleOf(engine, kind) {
    return `${engine || 'siglip'}:${kind || 'embedding'}`;
}

/** Whether two answers' scores sit on the same scale, and so may be compared at all. */
export function comparable(a, b) {
    return scaleOf(a.engine, a.kind) === scaleOf(b.engine, b.kind);
}

/** One score, written in the units whatever produced it actually measured.
 *
 *  A cosine is named because it is not a probability and a reader who assumes [0, 1]
 *  misreads every one of them. A cell's score is named because nothing detected
 *  anything -- it is the weight of accumulated evidence, not a confidence. A detector's
 *  box confidence is the only one that means what a reader already assumes, so it is the
 *  only one shown bare.
 */
export function scoreText(engine, kind, value) {
    if (!Number.isFinite(value)) return '';
    const scale = scaleOf(engine, kind);
    if (scale === 'hyperspace:item') return value.toFixed(2);
    if (scale === 'hyperspace:heatmap' || scale === 'hyperspace:area') {
        return `score ${value.toFixed(2)}`;
    }
    return `cosine ${value.toFixed(2)}`;
}
