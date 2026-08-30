# RAG Evaluation Question Bank
### Corpus: `vell_harbour_story.pdf` — *The Salt Cartographers of Vell Harbour* (10 pages, fully fictional)

**How to use this:** the story is invented, so nothing here can be answered from an LLM's pretrained knowledge. Every correct answer must come from retrieval. If your system answers Q14–Q20 correctly, your chunking and retrieval are doing real work. If it answers Q19 or Q20 with anything other than "the document doesn't say," you have a hallucination problem, not a retrieval problem.

**Built-in traps.** The document deliberately contains:
- **Name collisions** — Marek Oduya (rival Deepmark) vs. Maren Oduya (his sister, the clerk).
- **Date collisions** — 14 Tessin 1871 (the sighting) vs. 14 Vessin 1871 (the audit).
- **A number cluster** — "nineteen" appears in five unrelated roles; "240" and "244" fathoms are four apart.
- **Facts split across distant pages** — a rule on p3 is only violated by a roster on p5; a birthdate on p2 only becomes meaningful on p9.
- **A plausible-sounding non-answer** — a guessed translation presented as a guess.

---

## Easy (single chunk, verbatim lookup)

**Q1.** Who founded Vell Harbour, and in what year?
> Odessa Pell, in the spring of 1682 of the Coastal Reckoning. *(p1)*

**Q2.** What vessel carried the Verity Expedition, and what were her length and build year?
> The *Meridian Wren*, a 68-foot ketch built in 1859 at Cold Harbor by the shipwright Anselm Roke. *(p5)*

**Q3.** What bearing and distance did Osrin Fale report from the Kettle Light?
> Bearing 212 degrees, eleven nautical miles. *(p4)*

**Q4.** How many glass trees stood in the orchard on Quill Island, and how tall was the tallest?
> 208 trees; the tallest measured 6.1 metres. *(p7)*

**Q5.** How did the Chart Hall vote, and on what question?
> Twenty-three to nineteen to withhold the Guild seal — on whether a chart could be sealed on the verification of two Deepmarks who had sailed in the same hull. *(p8)*

**Q6.** What five words are on Ilvane Corrow's headstone?
> "SHE COUNTED FORTY-EIGHT." *(p10)*

---

## Medium (needs synthesis within a section, or two nearby chunks)

**Q7.** Name the four Guild ranks in order, and state which of them may sign a chart.
> Tally, Chartwright, Deepmark, Anchor-Master. Only Deepmarks and Anchor-Masters may sign. (Anchor-Masters are capped at seven living at once.) *(p3, Table 1)*

**Q8.** What three penalties did the Guild impose on Ilvane Corrow, and which did she later say was the one that hurt?
> A fine of 600 crowns, suspension from all Guild function for three years, and revocation of her ink privileges — the last of these was the part she said hurt. *(p8)*

**Q9.** How is salt-ink made, and how must it be stored?
> Cuttlefish sac, iron gall, and 3 grams of powdered pumice per litre. It dries in 40 seconds and must be kept below 12 °C or it separates within a week. *(p3)*

**Q10.** Who financed the expedition, with how much money, and under what condition?
> Perrin Sable, the widow of a shipowner, gave 1,200 crowns on the sole condition that no part of the island be named after her. *(p4)*

**Q11.** Trace the photographic plates: how many were exposed, how many were usable, and what became of them?
> 14 plates were exposed; 5 fogged and 9 came out clear. Eight of the nine were destroyed in the Chart Hall fire of Deepmonth 1877; the one survivor was out on loan, shows the obelisk, and is held in case twelve. *(p7 + p9 — two-hop)*

**Q12.** What happened on 14 Tessin 1871, and what different event happened on 14 Vessin 1871?
> 14 Tessin: Osrin Fale logged the sighting ("Land where there is no land. Bearing 212. Eleven miles."). 14 Vessin, one month later: the Guild's annual audit found nineteen charts missing from the third floor, covering the southern approaches. No connection was ever demonstrated. *(p4 — distractor discrimination)*

**Q13.** Who was Bela Massin, and what did she do after the penalty vote?
> Ilvane's master from 1863 (a Chartwright of eleven years' standing when she took her on), later a Deepmark and the expedition's chart verifier. She resigned from the Guild after 41 years rather than sit on the panel that voted the penalty, never drew another chart, and lived until 1888. *(p2 + p8 — two-hop)*

---

## Hard (multi-hop, arithmetic, or adversarial)

**Q14.** State Rule Nine, then explain exactly how the Verity Expedition failed to satisfy it — naming the individuals responsible for the failure.
> Rule Nine: no chart may be published under Guild seal until the same water has been verified by two Deepmarks working from **separate boats**. The expedition's only two Deepmarks — Ilvane Corrow and Bela Massin — sailed aboard the same hull, the *Meridian Wren*. Corrow's workaround was to sound twice from opposite ends of one vessel. *(p3 rule + p5 roster — the rule and the violation never appear in the same chunk)*

**Q15.** On what date was Quill Island found to be gone, and why is that date personally significant to Ilvane Corrow?
> The *Nine Sisters* found open water on the third of Marrowtide, 1873 — Ilvane's twenty-fourth birthday (born 3 Marrowtide 1849). *(p2 + p9)*

**Q16.** Give both soundings taken at the island's position and the difference between them.
> The *Meridian Wren* sounded 240 fathoms in open water on 6 Hollowmonth 1871, two days before landfall. The *Nine Sisters* sounded 244 fathoms at the exact position in Marrowtide 1873 — four fathoms deeper. *(p6 log + p9)*

**Q17.** In what year did Osrin Fale begin keeping the Kettle Light? Show two independent routes in the text that give the same answer.
> 1840. Route one: he had kept the light for 31 years as of 1871 (1871 − 31 = 1840). Route two: he died in 1879 having kept it 39 years in total (1879 − 39 = 1840). *(p4 + p10 — arithmetic across pages)*

**Q18.** The number nineteen appears in at least five unrelated roles in this document. List them.
> (1) The Slack occurs every 19 days; (2) 19 charts went missing in the Vessin 1871 audit; (3) Marek Oduya was a Deepmark of 19 years' standing; (4) the Chart Hall vote was 23 to **19**; (5) the voyage lasted 19 days (2–21 Hollowmonth). *(p1, p4, p8, p6 — tests whether retrieval collapses on numeric near-duplicates)*

**Q19.** *(Negative control.)* What language was the obelisk inscription written in, and what does it say?
> **The document does not say.** The eleven characters are in "a script that has never been identified and that resembles no writing known on this coast or any other." Ovid Kesh's rendering — *"we counted, and were counted"* — is explicitly described as a guess, and probably a sentimental one. A correct system reports the non-identification and flags the translation as speculative. *(p7)*

**Q20.** *(Negative control.)* What was the name of the vessel that Ilvane's brother Wick sailed on when he left the islands?
> **Not in the document.** Wick left at sixteen and "was never mentioned again in any document that survives." Note the nearby lure: the *Grey Petrel* is her **father's** trawler, lost off Bitter Shoal in 1856 with fourteen hands. A system that answers "*Grey Petrel*" has retrieved the right chunk and reasoned wrongly. *(p2)*

---

## Suggested scoring

| Band | Questions | What a failure tells you |
|---|---|---|
| Easy | Q1–Q6 | Chunking or embedding is broken at the basics |
| Medium | Q7–Q13 | Chunks are too small, or top-*k* is too low to gather 2 passages |
| Hard, multi-hop | Q14–Q18 | No re-ranking / query decomposition; try HyDE or multi-query |
| Negative controls | Q19–Q20 | Prompt isn't enforcing groundedness; the model is filling gaps |

Useful extras to log per question: whether the gold chunk appeared in top-*k* at all (retrieval recall) versus whether the answer was right (end-to-end), since Q14–Q18 fail most often when recall is fine and the generator only got one of the two needed passages.
