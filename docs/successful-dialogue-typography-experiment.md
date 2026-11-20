# Successful-dialogue enclosure typography

## Hypothesis frozen before tuning

At `c86fd39`, successful dialogue can still be a narrow stack of short lines.
Examples include `ubunchu01_06` unit 11 (eight lines, 18px) and `ubunchu06_0002`
unit 9 (eight lines, 24px). Overflow-only retries cannot improve these successes.
Use the existing independently checked original enclosure space to reletter such
dialogue without changing a single cleanup pixel or target character.

Freeze the same five representative pages before this comparison: ordinary
dialogue, fragments, furigana, mixed scripts, adjacent/shared balloons, dark text
and effects remain in view. Replay the 21 reused pages, without OCR/translation
inference. They are not held-out linguistic evidence. Keep old runs untouched.

Bound the experiment to four already-successful, multiline, multiword candidates
per page, prioritized by line count and source identity. Preserve every other
committed operation as a reservation. Try only existing closed-enclosure expansion
on validated light/flat native cleanup, with the existing neighboring-source,
unexplained-ink, connectivity, outline and clearance checks unchanged.

Accept a candidate only if exact characters and foreground/erase masks are
unchanged, the font is not smaller, line count falls by at least two, and the
existing line-quality cost improves. Keep the successful primary on overflow,
resource exhaustion, failed geometry, containment or candidate comparison. This
is a bounded presentation hypothesis, not a claim of optimal typesetting.

Inspect original/baseline/candidate full pages at reading size and relevant crops.
Report readability, omissions/remnants, border/art damage, source coverage, cost
and failures separately. Reject immaterial changes or regressions. Do not change
OCR/translation, shorten strings, alter font assets, acquire models/dependencies,
or weaken cleanup protections. Viewer integration follows only useful validation.

## Result: rejected as immaterial, not exposed

The five cases and original/font/layer/source bytes were frozen under
`test-results/successful-typography-freeze-v1/`. The full 21-page reused comparison
and exact experimental implementation are retained under
`test-results/successful-typography-corpus-v1/`; the smaller development pilot is
`test-results/successful-typography-pilot-v1/`. No models or translations ran.

All 182 successful replacements, original targets, foreground/erase masks and
clean pages remain identical. Only `ubunchu01_06` unit 38 changed text pixels:
seven lines at 17px became five at 19px. The same slash menu remains. Reading-size
full-page and original/baseline/candidate crop inspection found it less cramped,
but not a material improvement to this page overall. The other 20 composites
were unchanged. The motivating unit 11 fails original 4/8-core agreement; unit 9
on `ubunchu06_0002` fails neighboring-source protection. Neither was weakened.
Existing OCR/wording errors, omissions and effects remain; no accuracy claim.

There were 93 eligible successful layouts, 72 bounded candidate evaluations and
21 candidate-count budget exclusions. Most candidates rejected neighboring-source
protection (33) or 4/8 agreement (21); four rejected open regions, four had no
additional space, six failed the material-gain comparison, and three had a
resource/overflow validation rejection. The single accepted candidate is the one
described above. Additional typography work measured 4.53s total; full renderer
replay took 79.91s. These serial reused-artifact measurements are not an end-to-end
speed or linguistic benchmark. Inference calls: **zero**.

Seven synthetic typography tests passed in the development experiment, including
primary retention on overflow/resource/geometry/text-drift failures. That is not
evidence of usefulness. The experimental module and option were removed from
production rather than shipping a marginal route. Frozen code/evidence remain
local; they were not force-added. Viewer integration will expose only the
previously demonstrated enclosure-failure retry and demand-driven localization,
not successful-dialogue relettering or an unvalidated translation model switch.
