# Viewer: validated enclosure and localization controls

## Scope

The viewer previously could not select the enclosure-failure retry or the
crop-localized cleanup route. The latter now uses demand planning (`c86fd39`),
not eager predictions for already-rendered regions. Expose those two demonstrated
experimental routes without altering defaults, selecting a model, or expanding
pixel permissions. Keep OCR/translation/rendering coordinated by `src/main.py`.

Successful-dialogue enclosure relettering was evaluated and rejected as immaterial
in `successful-dialogue-typography-experiment.md`; it is **not** a viewer option.
Nearby translation context also remains unavailable here pending a controlled
same-model/profile comparison with identical single-rendition instructions. The
rejected full-generalist translation switch and slash-menu evaluation harness
are not silently enabled or promoted.

## Controls and persistence

- **Enclosure layout:** Source-local (unchanged default), or retry failed lettering
  within supported original enclosure space. It does not reletter successes or
  expand cleanup masks.
- **Demand-driven crop localization:** off by default. Native rendering produces
  bound requests before the existing local OCR reader performs crop-only inference.
  Model agreement is not a source correction, accuracy score or art-ownership proof.
- Either experimental choice explicitly requires **Final rendering**, **Best
  effort**, and **Native cleanup + OCR checks**. Invalid combinations reject before
  dispatch; no automatic model/policy switches or asset-path selection.

`options()` validates the choices. Job creation stores them in `viewer.json`;
`execute_job()` forwards them to the normal coordinator. Retry inherits and can
explicitly change them while retaining the old run/upload. History restoration
retains new metadata and can reconstruct executed choices from older pipeline
reports without rewriting them or running models. Existing jobs without the new
settings remain off/source-local.

The browser restores the controls on history selection, includes them in upload
and retry requests, and reports actual requested/call/consumption/budget counts
and optional-pass failures. Per-region inspection includes crop request reasons
and localized/enclosure attempt outcomes. Native baselines, derived previews,
intermediate PNGs and source-bound records remain accessible as job artifacts.
Original OCR and successful primary translations are not replaced by menu parsing.

## Tests

Portable tests cover invalid values/dependencies/path injection, defaults,
actual HTTP upload/retry through the real coordinator with injected readers,
no redundant translation/full-page OCR in the crop pass, result reporting,
artifact availability, unchanged previous files, restart restoration without
inference, and legacy executed-policy reconstruction. Existing viewer security
and token/origin/symlink protections remain unchanged.

An actual local-model viewer run is recorded separately below; injected tests
alone are not presented as live execution or linguistic validation.

## Actual viewer run

`test-results/viewer-localization-live-v2/` contains the frozen executed sources,
model identity, runner, snapshot, actual job artifacts, and within-run audit.
After that execution, only Python/HTML formatting and the browser's missing-count
notice were refined. Python ASTs and all coordinator/OCR/translation/rendering
sources are unchanged. The final notice was checked against the saved actual
snapshot without more inference under
`test-results/viewer-localization-final-presentation-audit-v1/`; failed optional
passes now say “not recorded” rather than inventing zero counts or staying pending.
Job `6b5259611d4c429fbf23a1d68aa7a654` used a real loopback HTTP upload of
`ubunchu01_06`, existing local Paddle v6 medium and explicitly configured native
`translategemma:27b`. Neither providers nor models were switched or mocked.

- 42 saved units, 41 translation calls, one blocked unit, no translation budget
  omissions. This **fresh OCR differs from the old frozen 40-unit/39-call page**;
  it is not an isolated quality A/B against that old page.
- Native output: 15 replacements. Final: 16. Two crop predictions were requested,
  called and supplied to cleanup; four eligible requests remained budget-limited.
- Synaptic rendered at 22px with its unchanged primary translation. Network
  rejected at `localized_gray_interior`. All 15 native foreground/erase/text PNGs
  remain byte-identical; original translation records and raw OCR are unchanged
  by the optional crop pass. Artifact bytes were fetched and checked through HTTP.
- Status **partial**, no fatal execution error, no crop-provider failure. At
  reading size, Japanese omissions, screen/effect text, awkward wording and slash
  menus remain. This reproduces a known narrow recovery; no new linguistic or
  generalization gain is claimed. The rejected typography route is absent.
- Measured elapsed 64.1s; native rendering 6.93s (including 0.62s planning), crop
  phase 3.86s including initialization/release, retry rendering 11.09s. Tests ran
  concurrently; load/cache state is not controlled. Not a throughput/speedup claim.
- Restart/history inspection restored both options and the reported counts
  without inference. Actual upload/retry execution and immutable old artifacts
  were separately tested through HTTP with synthetic readers/providers.

The first live-run harness attempt (`viewer-localization-live-v1`) failed because
its temporary script shadowed Python's `http` module. It failed before `/api/config`
or upload, so made no OCR/translation calls. The corrected attempt used a fresh
output directory; the failed artifacts were not overwritten or hidden.

The temporary loopback test server was shut down after inspection. To use the
controls, restart the normal viewer with your existing model/font configuration,
for example (choose an available port and job root):

```sh
.venv-paddle6/bin/python tools/dev_frontend.py --port 7860 \
  --model-root ~/.cache/tralens/paddle-v6-models --device cpu \
  --fonts test-results/render-parent-input-v1/fonts.json \
  --endpoint http://100.121.133.88:11434
```

This serves only `http://127.0.0.1:7860/`. No inference occurs merely from starting
the viewer; upload and enable the experimental controls explicitly. Existing
history stays inspection-only until the operator requests a fresh retry.

The final release repeat passed **926 tests in each existing environment**;
source-only passed with 39 optional-evidence skips. JavaScript syntax and pure
view-state/count/failure notices also pass. Three new portable viewer tests cover
the experimental controls and real coordinator/HTTP lifecycle; the typography
prototype's seven tests are not counted as shipped coverage. No dependencies,
models, fonts, licenses, providers, repository visibility, or user-owned files
were changed. Nothing was pushed.
