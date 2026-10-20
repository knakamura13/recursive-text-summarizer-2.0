# Human-written summaries as a quality baseline

Researched 2026-10-19. This document asks whether human-written summaries of long texts, such as study guides, dataset references, and expert abstracts, can serve as a quality baseline for this project. Every claim links to the primary source it came from: a dataset paper, an official repo, a dataset card, a license file, or a site's own terms. **UNCONFIRMED** marks anything that could not be checked in a primary source. The terms-of-use notes are a reading of public pages, not legal advice.

## Recommendation

Do not adopt SparkNotes or another study-guide corpus as the baseline. Use human summaries only as a source of **key points**, and score our output on two separate axes:

- **Coverage:** recall of those key points.
- **Grounding:** precision, measured by the existing claim verification against the source.

Three things drive this.

1. **No public dataset matches the use case.** No public dataset pairs textbook or self-help chapters with human reference summaries (see [Fit for this project](#fit-for-this-project)). The nearest are LCFO, which has expert summaries at 20%, 10% and 5% of source length, and CC BY sets such as SQuALITY and GovReport. They make a useful public calibration set, but the main benchmark still needs key-point checklists written for the user's own chapters. That is the approach the September chapter-comparison study already uses.
2. **Study guides are the wrong tool.** They cover fiction rather than textbooks, their terms forbid automated access, and models have probably memorized them.
3. **ROUGE does not fit.** Its scores track length and wording more than content, and our 25–50% outputs are several times longer than any available reference.

## Datasets

The table gives the average document and summary length, in the unit each paper reports.

| Dataset | Domain | Doc length | Summary length | Written by | License | How to obtain |
|---|---|---|---|---|---|---|
| BookSum chapter | Novels, plays | 5,101.88 tokens | 505.42 tokens | Study-guide authors, scraped | Code BSD-3; summaries not licensed | Re-scrape via Wayback |
| SQuALITY | Public-domain sci-fi stories | 5,199.4 tokens | 441.9 (plot) | Hired writers | Summaries CC BY | Direct download |
| GovReport | GAO/CRS reports | 9,409.4 words | 553.4 words | Agency experts | CC BY 4.0 | Direct download |
| QMSum | Meetings | 9,069.8 | 69.6 | Trained students | Repo MIT | Direct download |
| SummScreen | TV transcripts | 6,420.7–7,605.4 tokens | 113.7–380.6 tokens | Fans, Wikipedia | None stated | Google Drive |
| arXiv / PubMed | Papers | 4,938 / 3,016 words | 220 / 203 words | Paper authors (abstracts) | HF field "unknown" | Google Drive, HF |
| LCFO | 7 domains | ~5k words | 20% / 10% / 5% of source | Expert writers | CC-BY-NC 4.0 | HF; most sources must be rebuilt |
| Multi-LexSum | Civil-rights cases | ~75,543 words (long-summary subset) | 646.5 / 130.2 / 24.7 words | Lawyers, law students | Summaries CC BY-NC | HF |
| MeetingBank | City council meetings | 2,892 tokens (segment) | 87 tokens | Official minutes (authors not named) | HF: CC BY-NC-SA 4.0 | HF |
| SciNews | Papers to news stories | 7,760.90 tokens | 694.80 tokens | Science journalists | Papers CC BY 4.0 | Paper links |

The details below cover only what the table cannot.

### BookSum

[Kryściński et al. 2021](https://arxiv.org/pdf/2105.08209), Findings of EMNLP 2022.

- **Size.** Chapter level has 12,630 examples and book level has 405, from 217 titles. Splits are 80/10/10 by title.
- **Where the summaries come from.** The collection scripts cover BookWolf, CliffsNotes, GradeSaver, NovelGuide, PinkMonkey, Shmoop, SparkNotes and TheBestNotes ([repo](https://github.com/salesforce/booksum/tree/main/scripts/data_collection)). These are commercial and community study guides, scraped rather than commissioned. Analysis sections were removed with regex, and summaries were aligned to chapters automatically, then checked by hand ([paper](https://arxiv.org/pdf/2105.08209)).
- **License.** The code is [BSD-3-Clause](https://raw.githubusercontent.com/salesforce/booksum/main/LICENSE.txt).
  - The [README](https://raw.githubusercontent.com/salesforce/booksum/main/README.md) limits the scripts to research use and says users are responsible for obtaining all rights.
  - The paper's limitations section relies on US fair use, advises legal counsel, and states that no data was stored after the research ([paper](https://arxiv.org/pdf/2105.08209)).
  - A third-party Hugging Face copy, [`kmfoda/booksum`](https://huggingface.co/datasets/kmfoda/booksum), redistributes the summaries under a `bsd-3-clause` tag. That tag comes from the code license. It does not grant rights to the study-guide text.
- **Obtaining it.** Book texts come from a public bucket. Summaries are re-scraped from Web Archive links, which the README calls "often ... unreliable". The repo is [archived](https://github.com/salesforce/booksum).
- **Quality.**
  - Partial re-scrapes yield different counts ([Multi-LexSum, footnote](https://arxiv.org/pdf/2206.10883)).
  - The predecessor dataset, NovelChapters, says outright that its authors lack the rights to redistribute. It also dropped summaries that "often contain a lot of commentary" ([Ladhak et al. 2020](https://arxiv.org/html/2005.01840)).
  - Study-guide "summaries" mix plot with interpretation. This is the commentary those authors had to filter.

### SQuALITY

[Wang et al. 2022](https://arxiv.org/pdf/2205.11465).

- **Source.** Gutenberg science-fiction stories of 3,000–6,000 words.
- **How the summaries were written.** Every story asks "What is the plot of the story?" plus four other questions, with four responses each, at 75–500 words. Writers were Upwork freelancers and undergraduates. They reviewed and ranked each other's responses and were told the text would be released for AI development.
- **License.** Stories are under the Gutenberg license and summaries under CC BY ([repo](https://github.com/nyu-mll/SQuALITY)).
- **Versions.** v1.1 has 127 stories (50/25/52). v1.3 fixed missing text in 10 stories.
- **Metrics.** Within model outputs, ROUGE-1's correlation with human judgment is r ≈ -0.07 (Table 7 reports r × 100), close to zero and not significant. Multiple references "still correlate poorly" ([paper](https://arxiv.org/pdf/2205.11465)).

### GovReport

[Huang et al. 2021](https://arxiv.org/pdf/2104.02112).

- **Size and splits.** 19,466 reports: 12,228 GAO and 7,238 CRS. Splits by date are 17,519 / 974 / 973. Compression ratio is 19.0.
- **How the summaries were written.** GAO summaries are expert-written highlights with the sections "Why GAO did this study", "What GAO found" and "What GAO recommends". CRS reports were kept only when they carried an expert summary.
- **License.** [CC BY 4.0](https://gov-report-data.github.io/), also on [HF](https://huggingface.co/datasets/launch/gov_report). GAO products are not copyrighted in the US ([GAO](https://www.gao.gov/copyright)). CRS copyright status is **UNCONFIRMED**.
- **Quality.** The "What GAO recommends" section is excluded by default ([HF card](https://huggingface.co/datasets/launch/gov_report)). SCROLLS dropped examples whose summary appears verbatim in the report, or whose length ratio is extreme ([Shaham et al. 2022](https://arxiv.org/pdf/2201.03533)).

### QMSum

[Zhong et al. 2021](https://arxiv.org/pdf/2104.05938).

- **Size.** 1,808 query–summary pairs over 232 meetings (AMI, ICSI, Welsh and Canadian parliaments).
- **How the summaries were written.** Summaries are query-focused at 20–150 words. Writers were trained English-literature undergraduates, reviewed by an expert.
- **Length unit.** The paper's length table does not say whether it counts words or tokens (**UNCONFIRMED**).
- **License.** The repo is [MIT](https://github.com/Yale-LILY/QMSum). AMI is [CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml). ICSI and the parliamentary transcripts are **UNCONFIRMED**.

### SummScreen

[Chen et al. 2022](https://arxiv.org/pdf/2104.07091).

- **Source.** TV transcripts from fan sites, paired with fan or Wikipedia recaps. FD has 4,348 episodes and TMS has 22,503.
- **Quality.** The paper notes that many recap details are absent from the transcript, because episodes are audio-visual. It also notes that more than half the characters in a transcript are missing from the recap. A reference can therefore reward content the source does not contain.
- **License.** None stated in the [repo](https://github.com/mingdachen/SummScreen).

### arXiv and PubMed

[Cohan et al. 2018](https://arxiv.org/pdf/1804.05685).

- **Size.** arXiv has 215K papers and PubMed 133K. The abstract is the reference.
- **License and maintenance.** The [HF card](https://huggingface.co/datasets/armanc/scientific_papers) lists the license as "unknown". The [repo](https://github.com/armancohan/long-summarization) is Apache-2.0 and not maintained.
- **Quality.**
  - [Koh et al.](https://arxiv.org/pdf/2207.00939) hand-checked 743 arXiv test references and found noise in more than 60%.
  - Salient content is concentrated at the start of the source ([BookSum paper](https://arxiv.org/pdf/2105.08209)).
  - 54% of PubMed reference bigrams appear in the source ([Krishna et al. 2023](https://arxiv.org/pdf/2301.13298)).
  - An abstract is written to advertise a paper, not to cover its body. No primary source measuring how well abstracts cover the full body was found (**UNCONFIRMED**).

### Datasets from 2022–2026 with human references

- **LCFO** ([Costa-jussà et al. 2024](https://arxiv.org/abs/2412.08268), [paper HTML](https://arxiv.org/html/2412.08268v3), [HF card](https://huggingface.co/datasets/facebook/LCFO/raw/main/README.md)).
  - **Design.** Each document of about 5k words has three expert summaries at 20%, 10% and 5% of source length, plus about 15 QA pairs. Writers had writing or editing experience and were told to "Convey ALL the main core ideas".
  - **Size.** The paper reports 252 documents, while the card says 251. The set is test-only.
  - **License.** CC-BY-NC 4.0. Most source texts must be rebuilt from their original datasets.
  - **Quality.** GPT-4o-mini was rated about 7% above the human short summaries. Automatic metrics correlate about 0.4 with human scores. LCFO is the only set found that is graded by ratio of source length, which is how this project specifies targets.
- **Multi-LexSum** ([paper](https://arxiv.org/pdf/2206.10883), [HF](https://huggingface.co/datasets/allenai/multi_lexsum)). It has 9,280 expert summaries at three lengths for about 4,500 cases. Summaries are CC BY-NC and source documents are public domain.
- **MeetingBank** ([paper](https://arxiv.org/pdf/2305.17529), [HF](https://huggingface.co/datasets/huuuyeah/meetingbank)). References are official minutes, which the paper says do not always follow the discussion. San Francisco forbids redistributing its minutes.
- **SciNews** ([paper](https://arxiv.org/pdf/2403.17768)). Journalist-written explainers paired with CC BY papers. This is a lay-explanation task, closer to textbook prose than an abstract is.
- **Rejected or derivative sets.**
  - [ECTSum](https://arxiv.org/pdf/2210.12467) uses bullet summaries of about 50 words.
  - [ACLSum](https://arxiv.org/pdf/2403.05303) takes inputs of about 915 words.
  - [BigPatent](https://arxiv.org/pdf/1906.03741) and [LoRaLay](https://arxiv.org/pdf/2301.11312) use abstracts as references.
  - [VT-SSum](https://arxiv.org/abs/2106.05606) has extractive, slide-derived lecture summaries that no human wrote.
  - [ODSum](https://arxiv.org/pdf/2309.08960) repackages SQuALITY and QMSum.
  - Benchmarks such as [SCROLLS](https://arxiv.org/pdf/2201.03533), [ZeroSCROLLS](https://arxiv.org/pdf/2305.14196), [LongBench](https://huggingface.co/datasets/THUDM/LongBench), [L-Eval](https://arxiv.org/pdf/2307.11088) and [InfiniteBench](https://arxiv.org/pdf/2402.13718) reuse the sets above and score them with ROUGE. InfiniteBench's En.Sum footnotes sparknotes.com and cliffsnotes.com as sources for its novels. Its gold summaries are "sourced from the web" with no named site.

### Book-length evaluation without references

- **FABLES** ([Kim et al. 2024](https://arxiv.org/pdf/2404.01261), [repo](https://github.com/mungg/FABLES), MIT).
  - It has 3,158 claim-level faithfulness labels on LLM summaries of 26 books published in 2023–24, labeled by annotators who had read the books.
  - There are no reference summaries, and the book texts are not released.
- **BooookScore** ([Chang et al. 2023](https://arxiv.org/pdf/2310.00785), ICLR 2024). This is a reference-free coherence metric. The authors chose new books because BookSum "[is] in the pretraining data of most public LLMs".
- **Wu et al. 2021** ([OpenAI](https://arxiv.org/pdf/2109.10862)).
  - The authors had labelers write summaries of popular 2020 books. They "cannot rule out overlap" with BookSum.
  - The paper points to a website for "all human and model summaries". That the human summaries are actually on it was not checked (**UNCONFIRMED**).

## Study-guide and book-summary sites

The table covers what each site's terms allow for private evaluation, meaning content that is never redistributed and is used only to score this project's output. These are readings of the terms, not legal advice.

| Site | Terms (operator, date) | Relevant clauses | Private evaluation use |
|---|---|---|---|
| SparkNotes | [Terms](https://www.sparknotes.com/terms-of-use/) (Barnes & Noble, updated 2022-04-07) | License "solely for your personal, non-commercial use". Bans "any robot, spider, data miner, crawler, scraper". | Manual reading allowed. Automated collection forbidden. |
| CliffsNotes | [Service Terms](https://www.cliffsnotes.com/terms-of-service) (Learneo, effective 2026-08-21) plus [Learneo ToS](https://www.learneo.com/legal/terms-of-service) | Content is for "educational, non-commercial and personal use only". Learneo bans "crawling or scraping ... (through manual or automated means)" and use "to create, train, or improve a third party product, ... including any machine learning models". It also bans "performance benchmarking", which most plausibly means benchmarking Learneo's own services. | Probably forbidden. Using guide text to evaluate and improve a summarizer falls most naturally under the "improve a third party product" ban. |
| LitCharts | [Terms](https://www.litcharts.com/terms) (Learneo, updated 2026-08-21) | Use is for "personal and educational use only". The terms "supplement the Learneo Terms of Service", which govern all other matters. | Probably forbidden, for the same Learneo clauses as CliffsNotes. |
| GradeSaver | [Terms](https://www.gradesaver.com/terms-of-use) | Content may be "downloaded or printed for personal use" and "may not be distributed for any reason". No scraping or AI clause. | Ambiguous. Personal download is allowed, and evaluation use is not addressed. |
| Shmoop | [Terms](https://www.shmoop.com/public/terms/) (as of 2020-04-28) | "personal, non-commercial use only". Bans spiders and robots. | Automated collection forbidden. |
| Shortform | [Terms](https://www.shortform.com/terms) (2025-04-23) | Bans "data mining, robots, or similar data gathering". | Automated collection forbidden. |
| getAbstract | [Terms](https://www.getabstract.com/en/terms-of-use) | Bans scrapers and use to "develop, train, fine-tune, or improve" AI. | Forbidden. |
| Blinkist | Not retrieved (HTTP 403) | **UNCONFIRMED** | **UNCONFIRMED** |

CliffsNotes' [robots.txt](https://www.cliffsnotes.com/robots.txt) also blocks AI crawlers. US fair use is decided case by case on four factors ([Copyright Office](https://www.copyright.gov/fair-use/)). The [Copyright Office's AI report, Part 3](https://www.copyright.gov/ai/), was not read for this note.

**Conclusion.** None of these sites offers a clean path. Copying a guide by hand into a private, local evaluation file would avoid the anti-scraping clauses. It would still run against the Learneo ban (covering CliffsNotes and LitCharts) on using content to improve a third-party product, and it may run against the personal-use limits of the others. Such files could never be committed to this public repo. The domain mismatch, covered next, makes the question moot for the main use case.

## Fit for this project

The main use case is whole chapters of textbooks and self-help books at 25–50% of source length. The goal is faithful coverage of explanations, distinctions, practical guidance, examples, and the relationships between them.

### Domain mismatch

The study-guide corpora (BookSum, NovelChapters) cover fiction, where a summary means plot plus commentary. No dataset of human reference summaries for textbook or nonfiction chapters was found, from any year. Recent book-level work (FABLES, BooookScore) gave up on references altogether. The closest nonfiction references are:

- LCFO (expository, multi-domain);
- GovReport (structured findings and recommendations);
- SciNews (lay explanation of technical work);
- Multi-LexSum (legal).

The closest publicly available sources still differ from this project's target texts. This is a finding from searching, not a proof that no such dataset exists.

### Length mismatch

The two sets of numbers below are not on the same scale, so they cannot be compared directly:

- Every reference above is 5–20% of its source (LCFO) or shorter. GovReport's compression ratio is 19.0, and BookSum chapter summaries are about 10%.
- Our targets are 25–50% of the source.

A 50% output can contain everything a 10% reference contains and much more. So a reference can tell us what must not be missing (a recall floor), but it cannot tell us whether the extra material is good. Two published length corrections exist:

- RoSE's nACU penalizes outputs longer than the reference ([Liu et al. 2023](https://arxiv.org/pdf/2212.07981)). That is the wrong direction for this project, where longer is the requested behavior.
- L-Eval gives the model the reference's word count to remove length bias ([An et al. 2023](https://arxiv.org/pdf/2307.11088)). That would mean changing our targets.

Neither correction applies as published.

### Training-data contamination

- Models have memorized many copyrighted books, and do better on the ones they memorized ([Chang et al. 2023, "Speak, Memory"](https://arxiv.org/pdf/2305.00118)). That paper names LitCharts, CourseHero and Quizlet as web sources of book excerpts.
- BooookScore's authors report that BookSum's books and references are in common pretraining data ([Chang et al. 2023](https://arxiv.org/pdf/2310.00785)).
- FABLES chose 2023–24 books for the same reason ([Kim et al. 2024](https://arxiv.org/pdf/2404.01261)).

A primary source showing LLMs reproduce SparkNotes text specifically was not found (**UNCONFIRMED**).

For this project, a popular self-help book such as Atomic Habits is likely to have widely circulated summaries in training data. A model could match a study-guide-style reference from memory instead of from the supplied chapter. Claim verification against the supplied source still catches unsupported additions, but a reference-overlap score would not.

### Why ROUGE and other n-gram metrics fit poorly

- Humans preferred GPT-3 summaries that scored about 7 ROUGE-L points lower ([Goyal et al. 2022](https://arxiv.org/pdf/2209.12356)).
- News references were rated worse than most systems' outputs ([Zhang et al. 2023](https://arxiv.org/pdf/2301.13848)).
- Metric–human correlation is weak to moderate on coherence and relevance ([SummEval](https://arxiv.org/pdf/2007.12626)), near zero within a single system type ([SQuALITY](https://arxiv.org/pdf/2205.11465)), and about 0.4 on LCFO ([LCFO](https://arxiv.org/abs/2412.08268)).
- ZeroSCROLLS reports low scores for semantically equivalent outputs and penalties for short summaries ([Shaham et al. 2023](https://arxiv.org/pdf/2305.14196)). L-Eval documents length bias in n-gram metrics ([An et al. 2023](https://arxiv.org/pdf/2307.11088)).
- With our outputs two to ten times the reference length, ROUGE precision would fall by construction, and its recall would reward wording over meaning. ROUGE also cannot detect an unfaithful claim.

## Recommended design

This design reuses existing repo machinery: the claim verifier in `summarizer/verification.py`, with verdicts `supported`, `contradicted`, `insufficiently_supported` and `not_meaningfully_verifiable`, and the source-backed checklists from `experiments/chapter_comparison/`.

1. **Turn human summaries into weighted key-point checklists.**
   - Split each reference into atomic key points, following the Pyramid content units ([Nenkova & Passonneau 2004](https://aclanthology.org/N04-1019.pdf)) and ACUs ([Liu et al. 2023](https://arxiv.org/pdf/2212.07981)). Then check each key point against the source, and drop any the source does not support. SummScreen shows that references can contain material missing from the source.
   - With LCFO, weight each key point by the shortest summary that contains it: 5% first, then 10%, then 20%. This adapts the Pyramid idea, where agreement across more summaries means higher weight. Weighting by nested lengths is my own proposal, not a published method.
   - For the user's own chapters, keep writing checklists from the source before viewing any output, as the September study did. Add categories for this use case: explanation, distinction, practical guidance, example, relationship.
2. **Score coverage and grounding separately, as recall and precision.**
   - Coverage is the weighted share of key points the output entails. FineSurE takes this keyfact approach, though its authors note it was tested mainly on news ([Song et al. 2024](https://arxiv.org/pdf/2407.00908)).
   - Grounding is the share of output claims the existing verifier marks `supported`, with `contradicted` reported separately.
   - Never merge the two into one number. A 50% output can score full coverage while adding unsupported material, and only grounding catches that.
3. **Handle length mismatch by reading references as a floor.**
   - Report the key-point recall from the shortest reference as "must-cover", and from the longest as "should-cover".
   - Do not penalize extra length against the reference. Judge the extra content by grounding, and by checklist items drawn from the source that no reference covers.
   - Report output words and target words beside the scores, so length is visible but separate. RoSE recommends treating length as its own dimension ([Liu et al. 2023](https://arxiv.org/pdf/2212.07981)).
4. **Judge clauses, not whole summaries.** Use clause-level or key-point-level judgments rather than whole-summary Likert scores. Fine-grained judgment cut annotator standard deviation from 18.5 to 6.8, on SQuALITY, and labeling half the units kept 0.78–0.89 Kendall's τ with full labeling ([LongEval](https://arxiv.org/pdf/2301.13298)).
5. **Keep licenses clean.**
   - Commit only CC BY material and fixtures written by the project: SQuALITY, GovReport, and Gutenberg or GAO sources.
   - Keep LCFO (CC-BY-NC) and the user's books private, beside the existing private study artifacts.
   - Do not collect study-guide content.
6. **Guard against contamination.** Prefer chapters from books or editions published after the evaluated model's training cutoff, or the user's own lesser-known texts. Report results on popular titles separately.

A small public calibration set (a few LCFO or SQuALITY documents run at 20% and 25%) would show whether the key-point scorer agrees with expert summaries before it is trusted on private chapters. Running it is out of scope for this note, which made no model requests.

## Unconfirmed

- Blinkist terms: HTTP 403 on fetch.
- What the Learneo ToS means by "performance benchmarking".
- CRS report copyright status.
- Licenses for ICSI and the parliamentary transcripts (QMSum), SummScreen, ECTSum, ACLSum, SciNews news text, and LoRaLay.
- The QMSum length unit (words or tokens).
- A primary measurement of how well arXiv/PubMed abstracts cover the paper body.
- LLM reproduction of SparkNotes text specifically.
- Any textbook or nonfiction-chapter human-summary dataset, from 2025–26 or earlier.
- Whether the Wu et al. 2021 website actually hosts the human book summaries.
- LongStoryEval and CaseSumm, which were seen only in search results.
- SQuALITY v1.1 question and response counts, which are not stated in the source.
