# GPT-6 Luna manga translation trial

The feature branch keeps local translation as the viewer default. Selecting **GPT-6 Luna** uses the signed-in Codex CLI. My choice for Luna is a whole-page pass with numbered region crops, followed by a magnified reread of any skipped region while the call budget allows. Isolated text remains selectable. Page context helped with the meaning of hard dialogue and required fewer calls. A close view sometimes corrected a skipped region. Neither technique consistently read stylized effects.

## Matched comparison

I reused the same saved OCR readings and detector boxes from two color pages of *Pepper & Carrot* (episode 6, pages 3 and 6) and two monochrome pages of *Ubunchu!* (chapter 5 page 10 and chapter 8 page 4). The local column comes from saved TranslateGemma 27B runs. Luna received Japanese target inputs only. Visual conditions received one other Japanese page from the same series unless marked target only. The eight magnified targets included dialogue, already-English lettering, and difficult effects. All other conditions covered every region.

| Input to Luna | Target regions | Model calls | English returned | Sum of call durations |
|---|---:|---:|---:|---:|
| Isolated OCR, existing region boundaries | 41 | 41 | 39 | 223 s |
| Whole page and related page | 41 | 4 | 39 | 59 s |
| Whole page, numbered crops and related page | 41 | 4 | 39 | 51 s |
| Whole page and numbered crops, target only | 41 | 4 | 39 | 52 s |
| Magnified region, whole page and related page | 8 | 8 | 7 | 56 s |

The local baseline returned text for all 41 regions, including a sign already in English and OCR noise. Output count measures coverage. The comparison below uses source-image inspection and translation judgment. Timing includes CLI startup and inference, excludes OCR and rendering, and sums individual durations even when calls overlapped. Each condition used one generation per target or batch. These four pages cannot establish a general accuracy percentage.

## Where the choice mattered

| Source | Local | Isolated Luna | Visual Luna and assessment |
|---|---|---|---|
| *Pepper & Carrot* 6.3, コモナ市長の私 | “I, Mayor Komona” | “I, the mayor of Komona” | The visual variants also identify Komona as the place. Here replacing the translator alone fixed the meaning. |
| *Pepper & Carrot* 6.3, English contest banner | Redrew “Komona Potion Contest” | Kept the sign | All visual modes also kept it. |
| *Pepper & Carrot* 6.6, monster warning | Potion “for giant monsters” | Also “for giant monsters” | The page shows the transformation. Whole-page Luna rendered “turns you into a giant monster.” Both page modes also translated the roar and recognized Shichimi as a name. |
| *Ubunchu!* 8.4, 不要パッケージ | “unnecessary package” | “unwanted packaging” | Whole-page Luna preserved the software-package joke. The local result was already good. The crop-sheet rendition introduced an unnecessary plural. |
| *Ubunchu!* 5.10, 全然ワカってないお | “I don't understand at all.” | “You don't get it at all.” | Both visual modes also addressed the other speaker. |
| *Ubunchu!* 5.10, handwritten interruption | “I’m a network enthusiast.” | “So, since I'm a network—” | Whole-page Luna recovered “network install” but borrowed “But!” from the next balloon. With numbered crops and context it kept the interruption separate, losing “install.” |
| *Ubunchu!* 5.10, ピキッ | “Pxn” | Skipped | The art shows a disc cracking. Whole-page mode gave “THWACK!”, magnified gave “Thump!”, numbered crops with context gave “WHAM!”, and the target-only crop sheet gave “CRACK!” with incorrect kana. The final viewer run gave “SNAP!” This is still an unreliable class of lettering. |
| *Ubunchu!* 8.4, apparent 三 | “Three.” | “Three” | Both page modes left it alone. A magnified view suggested an ellipsis. Inspection shows border strokes and small marks at the edge. Rendering “Three” adds spurious dialogue. |

The extra same-series page had mixed effects. Removing it improved the cracking effect on 5.10 but brought the adjacent balloon's “But!” back into the interruption. This small trial supports offering related pages as an explicit choice. It does not establish that more pages always improve a translation.

## End-to-end behavior

Local detection and bounding boxes define what can receive replacement text. Luna supplies a Japanese reading, an English rendition or a skip, and a reason for each eligible region. The viewer retains its OCR reading alongside Luna's. Existing cleanup, compositing and artwork-protection settings still determine how the result is drawn. An undetected region cannot be recovered by this route.

On *Ubunchu!* 5.10, a live crop-sheet run displayed 12 of 14 regions. A new run through the isolated viewer worker, with Paddle, Hayai, and a closer review of skipped regions, displayed 14 of 14 in two Luna calls. The second call recovered the interrupted network line. “SNAP!” came from the first pass in that run, so the SFX gain cannot be credited to the retry. The final page still has stray ink around the handwritten line, and aggressive cleanup visibly softens the drawing around the cracking disc.

On the color wing page, a fresh viewer run with the crop reader displayed all 11 regions in one Luna call, including the roar. In this run the warning became “That potion makes giant monsters!” The transformation was clearer in the matched language trial. An earlier run with plain Paddle and whole-page Luna skipped the roar after reading its OCR as “Lkede.” Both runs illustrate how OCR configuration and generation variability can affect the final page.

I rendered the local and Luna previews a second time with identical LaMa settings, confidence-gated replacement, aggressive removal and always-show visibility. Comparing the resulting images, the color wing has a legible English roar in both variants. On the monochrome page, Luna letters the cracking disc, but part of the original effect and handwritten line survives beside the English. A translated region still needs pixel-level inspection.

The viewer accepts up to six explicitly selected saved runs as same-manga context. The production entry point also accepts repeated `--related-page PATH` arguments. Visual modes send those images to OpenAI. The isolated mode sends OCR text and coordinates. A zero call budget sends nothing. Model decisions bind to existing region IDs, and the normal renderer validates the translation against the current page and region before lettering.
