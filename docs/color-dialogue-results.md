# Color dialogue and form-row trials

Three Japanese Pepper&Carrot pages and two monochrome Ubunchu pages were processed locally. The original Japanese pixels, detector and reader outputs, translation responses, failed attempts, and clean and composite layers are retained under `test-results/production-best-effort-v8/`. Pepper&Carrot episode 6 art and scenario are by David Revoy, with Japanese translation by guruguru and contribution by Hồ Nhựt Châu. The downloaded source HTML and CC BY 4.0 provenance accompany the inputs.

## Visible changes

On Pepper&Carrot page 3, splitting centered lines produced both “I hereby declare the opening of the meeting!” and three disconnected pieces of the welcome. The revised grouping uses the original blank seams, centering, line height and existing image-boundary checks. It produces two connected announcements and keeps the English banner separate. An early trial merged “Carrot!” into “the Carrot! contest” on page 2. The final rule preserves the separate exclamation. The mayor still calls himself “Mayor Komona” where the Japanese means the mayor of Komona.

The light interior placement retry moved four color passages clear of nearby ink or balloon rims. The first version also moved a monochrome form title into a border. A long-rule check disables this retry there. Short runs of three detached dots adjacent to a horizontal source line can now be cleaned from a light background, including dots beside individual lines of a multiline passage. These pixels are recorded in the declared cleanup support. This restores a clean surface but cannot restore ellipses lost earlier in the translation.

On Ubunchu chapter 3, separating overlapping OCR rows across a sloping printed rule let the existing crop reader recover メッセージ and the translator display “Message.” independently. The following adversative in まったく意味不明だが now appears as “It doesn't make any sense at all, but...” alongside the next balloon. The original model response and the retry remain available. Broad incomplete-phrase retries sometimes lost the technical subject, added a bug, or changed a question. Only the narrower lost-contrast retry is enabled. Common neutral mathematical symbols can be drawn if the font contains them, which allowed the form report to appear. Its claim that the update “removed” a symbol is still unsupported. Sound effects obscure the continuation.

## Repair and language experiments

Using the pinned LaMa model, two predictions of the same native crop, viewed normally and reflected horizontally, were aligned and averaged before applying the original commit mask. The extra inference is confined to crops with substantial color in their known pixels. It uses the existing isolated CPU worker. On twelve painted crops with artificial lettering masks, mean absolute channel error inside the mask fell from 8.39 to 7.77 (0–255). On twelve different painted crops from the same pages with a different mask, it fell from 7.65 to 7.27, improving 11 of 12. Known pixels under artificial masks make this comparison possible. The original artwork under real lettering remains unknown. Monochrome controls had mixed per-crop results and retain single-view LaMa.

Qwen3.6 Dense and Qwen2.5-VL 7B each translated eight original contextual crops. Both recovered “Komona!” where the primary says “Como na!” The smaller model correctly used first person for falling asleep. The larger model added “good luck” to the contest dialogue. Both struggled with the clipped report. A wider crop caused the smaller reader to mix neighboring form rows into its response. These alternatives remain experimental.

## Verification and evidence

The final five saved pages have 72 bound target strings, 62 displayed replacements, and 10 source/target equalities. Some equalities are authentic artwork or punctuation; others can hide OCR errors. Every cleanplate was compared with its source outside the erase mask and every composite was reproduced from the saved layers. All 815 manifest-listed artifacts passed their hash checks. These are rendering checks. Remaining failures include incomplete readings, the wrong pronoun or referent, and damage to artwork around stylized text. There has been no independent bilingual assessment of these pages.

| Evidence within the experiment root | Contents |
| --- | --- |
| `color-improvement-01/` | First four-page baseline and the two vision model response logs |
| `color-improvement-02/` | First centered-row trial, including the Carrot regression |
| `color-improvement-03/` | Five-page run with the sentence-boundary fix; four final saved renderings in the individual `reviewed-render/` directories |
| `color-improvement-04/` | Separate form rows; report initially erased with no target because × was rejected |
| `color-improvement-05/` | Report displayed after glyph support fix |
| `color-improvement-06/` | Final Ubunchu chapter 3 run, comparison image, final viewer job and five-page pixel audit |
| `color-inpaint-controls/`, `color-inpaint-holdout/`, `color-lama-actual/` | Masked-pixel controls, predictions, error measurements and real repair comparisons |
| `color-fragment*.jsonl`, `color-improvement-06-tests.log` | Rejected language trials and 156 passing focused tests |
