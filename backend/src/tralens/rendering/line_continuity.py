"""Estimate line continuity from matching edge profiles on both sides of a repair."""
import cv2
import numpy as np


def bridges(crop, missing, commit):
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    unknown = cv2.dilate(missing.astype(np.uint8), np.ones((5, 5), np.uint8))
    edges = cv2.Canny(gray, 35, 90)
    edges[unknown != 0] = 0
    # A broad sound-effect box can hide more of a diagonal than a glyph mask.
    # Require twice the visible support on each bank for longer extrapolations.
    radius = cv2.distanceTransform(commit.astype(np.uint8), cv2.DIST_L2, 3).max()
    gap_limit = min(72, int(3 * radius + 8)) if radius >= 14 else 32
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 14, minLineLength=19, maxLineGap=gap_limit)
    if lines is None:
        return []

    evidence = []
    used = np.zeros_like(commit)
    # Prefer long segments and avoid counting several detections of one edge.
    longest = sorted(lines[:, 0], key=lambda line: -np.hypot(line[2]-line[0], line[3]-line[1]))
    for x, y, right, bottom in longest[:80]:
        length = np.hypot(right-x, bottom-y)
        dx, dy = (right-x)/length, (bottom-y)/length
        along = np.arange(int(length)+1)
        across = np.arange(-3, 4)[:, None]
        xx = np.rint(x + along*dx - across*dy).astype(int)
        yy = np.rint(y + along*dy + across*dx).astype(int)
        inside = (xx >= 0) & (xx < gray.shape[1]) & (yy >= 0) & (yy < gray.shape[0])
        xx = xx.clip(0, gray.shape[1]-1)
        yy = yy.clip(0, gray.shape[0]-1)
        hidden = unknown[yy, xx].any(axis=0)
        endpoints = np.flatnonzero(np.diff(np.r_[False, hidden, False]))
        profile = gray[yy, xx].astype(np.float32)
        profile -= profile.mean(axis=0)
        for start, end in endpoints.reshape(-1, 2):
            support = 10 if end-start > 42 else 5
            if end-start > max(42, gap_limit) or start < support or end+support > len(along):
                continue
            if not inside[:, start-support:end+support].all() or hidden[start-support:start].any() or hidden[end:end+support].any():
                continue
            gap = np.arange(start, end)
            gap = gap[commit[yy[3, gap], xx[3, gap]]]
            if len(gap) < 6 or used[yy[3, gap], xx[3, gap]].mean() > .5:
                continue
            # Contrast and polarity must agree across the gap. A shadow
            # ending on one side is insufficient evidence for a continuation.
            left = np.median(profile[:, start-support:start], axis=1)
            right_profile = np.median(profile[:, end:end+support], axis=1)
            agreement = left @ right_profile
            if (min(np.ptp(left), np.ptp(right_profile)) < 15
                    or agreement < .8 * np.linalg.norm(left) * np.linalg.norm(right_profile)):
                continue
            fraction = (gap-start+1)/(end-start+1)
            expected = left[:, None]*(1-fraction) + right_profile[:, None]*fraction
            used[yy[:, gap], xx[:, gap]] = True
            evidence.append((yy[:, gap], xx[:, gap], expected))
    return evidence


def error(candidate, evidence):
    """Mean absolute cross-line contrast error on the sampled continuations."""
    gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    errors = []
    for yy, xx, expected in evidence:
        observed = gray[yy, xx].astype(np.float32)
        observed -= observed.mean(axis=0)
        errors.append(np.abs(observed-expected).mean(axis=0))
    return float(np.concatenate(errors).mean())
