# 100 upgrades: the stacking roadmap

A concrete backlog of 100 upgrades that build on what Gauge already has and integrate through the
existing seams (story agents, the critic score, config switches, the daemon, the learning loop). Each is
one shippable change. Effort is a rough size: **S** a session, **M** a few, **L** a sustained batch.
Items marked *(needs X)* depend on a capability or external data. This is the plan to work through in
tested batches — not a promise that all 100 exist yet.

## A. Prose and craft realism (builds on the realism layer)  — DONE (gauge.write.prose + Craft Editor)
1. [done] Subtext editor: flag on-the-nose dialogue where characters state exactly what they feel/want. **M**
2. [done] Dialogue realism: detect everyone speaking in the same register; reward interruptions, overlap, silence. **M**
3. [done] Metaphor freshness: flag cliché similes against a cliché lexicon; reward novel, apt images. **S**
4. [done] Paragraph-rhythm pass: vary paragraph length deliberately; flag monotone blocks. **S**
5. [done] White-space/beat pacing: ensure action beats break up dialogue; flag talking-heads scenes. **S**
6. [done] POV discipline: detect head-hopping within a scene for limited-POV projects. **M**
7. [done] Tense/aspect consistency checker across a chapter. **S**
8. [done] Sentence-opener variety: flag repeated openings (She… She… She…). **S**
9. [done] Concrete-verb pass: flag weak verb + adverb pairs ("walked slowly" → "trudged"). **S**
10. [done] Period/register diction guard for historical settings (anachronism lexicon). **M** *(needs setting metadata)*

## B. Characters and psychology  — mostly done (gauge.write.ensemble + Cast Tracker); 15/17/19 need an LLM pass
11. [done] Want-vs-need tracker per character across the arc; flag a protagonist with no visible need. **M**
12. [done] Relationship-graph drift: visualise and flag relationships that never change. **M**
13. Character secrets ledger: track who knows what; flag a reveal to someone who already knew. **M**
14. [done] Idiolect enforcement: per-character vocabulary/catchphrase adherence (extends Voice Coach). **M**
15. Off-page life: characters act when not on screen; a lightweight "what did they do this chapter" sim. **M**
16. [done] Physical tells: consistent gestures/habits per character, flagged when absent or swapped. **S**
17. Arc-plan auto-proposal: suggest a personality-drift plan from the bible for each major character. **S**
18. [done] Ensemble balance: flag a cast member who has vanished for N chapters. **S**
19. Antagonist competence meter (extends Schemer): flag a villain who only reacts. **M**
20. [done] Character voice embeddings: cluster dialogue to catch two characters converging in style. **M**

## C. World and continuity
21. Map/geography memory: a place graph with distances; flag impossible travel times. **M**
22. Object/prop ledger: track items; flag a prop that vanishes or reappears. **M**
23. Timeline builder: absolute chapter timeline; flag events out of order or overlapping. **M**
24. Weather/season model: a simple climate so weather follows the calendar. **S**
25. Economy/resource tracking for the world (money, supplies) with consequences. **M**
26. Injury/healing model: wounds persist and heal at plausible rates. **S**
27. Day-night/clock within a chapter (extends the Continuity Auditor to explicit times). **M**
28. Named-entity canon index with fuzzy matching to catch spelling drift (extends Archivist). **M**
29. World rules as assertions the critic can check each chapter. **M**
30. Grounding dossier caching and reuse per recurring location *(needs Scout)*. **S**

## D. Plot and structure
31. Promise/payoff heatmap over the whole serial (extends Showrunner's ledger). **M**
32. Try-fail cycle detector: flag protagonists who succeed too easily. **M**
33. Scene-goal/conflict/outcome tagging; flag scenes with no conflict. **M**
34. Subplot tracker: weave and resolve B-plots; flag dropped threads. **M**
35. Chapter-shape planner: schedule action/quiet/reveal shapes across an arc (extends Pacing). **S**
36. Midpoint/climax placement guidance from arc position. **S**
37. Foreshadow planting and payoff linker. **M**
38. Cliffhanger variety: avoid the same cliff type repeating. **S**
39. Outline revision from reader drop-off data *(needs reader feedback)*. **M**
40. Multi-book series bible that carries arcs across books. **L**

## E. Learning and the engagement model
41. Gradient-boosted engagement model with calibration *(needs scikit-learn optional)*. **M**
42. Per-genre engagement models, selected by project genre. **M**
43. Comment-topic modelling to learn what readers react to. **M** *(needs comment data)*
44. Reader-cohort retention curves (binge vs weekly). **M** *(needs feedback)*
45. Active-learning: ask the author to rate the few chapters the model is least sure about. **S**
46. A/B publish experiments with proper significance testing *(needs feedback)*. **M**
47. Feature attribution report: which measured features move real retention. **S**
48. Technique-card effectiveness tracking: which absorbed moves correlate with retention. **M**
49. Contrastive fine-tuning data pipeline from hit-vs-miss pairs (extends contrastive). **M**
50. Self-distillation loop: the project model teaches GaugeLM its preferences. **L**

## F. GaugeLM (the native model)
51. BPE vocabulary growth and retraining on the author's corpus. **M**
52. LoRA-style adapters so one base model specialises per project. **L**
53. KV-cache for faster native sampling. **M**
54. Quantised (int8) weights for CPU speed. **M**
55. Longer context via sliding-window attention. **L**
56. Instruction-tuning run from the SFT data already curated. **M**
57. Speculative decoding with a tiny draft model. **L**
58. A proper eval suite (perplexity + held-out task scores) in CI. **S**
59. Mixture-of-experts tiny config for capacity without latency. **L**
60. Native embeddings so the whole stack can run with no external model. **M**

## G. Compute and performance
61. Prompt/response caching keyed by content hash to cut repeat calls. **S**
62. Batched agent reviews in one model call where schemas allow. **M**
63. Streaming drafts to disk so a crash never loses tokens (extends checkpoints). **S**
64. Per-chapter cost budget with graceful degradation when exceeded. **S**
65. Distributed drafting across worker pool for multi-project runs. **M**
66. GPU auto-tuning of batch size from the device detected. **S**
67. Incremental embedding index instead of full rebuilds. **M**
68. Async pipeline so research and drafting overlap. **M**
69. Telemetry dashboard (static HTML) of cost/latency/scores over time. **S**
70. Model-router: cheap model for bookkeeping, strong model for prose, learned per task. **M**

## H. Reader growth and analytics (legitimate marketing only)
71. UTM-tagged links and a privacy-respecting click log for owned channels. **S**
72. Best-time-to-post learning from your own engagement history. **M** *(needs own data)*
73. A/B subject lines for the opt-in newsletter (consented list only). **S**
74. Reader-funnel report: chapters → follows → newsletter → retention. **M** *(needs feedback)*
75. Cross-posting scheduler across owned accounts with per-platform formatting. **M**
76. Review-request drafting to send readers who finished (your own list). **S**
77. Serial-platform export formats (Royal Road, Scribble Hub markup). **S**
78. SEO metadata and sitemap for the static site. **S**
79. Accessibility pass on the published site (alt text, contrast, headings). **S**
80. Reader comment ingestion from your own site into the learning loop. **M**

## I. Platform, site, and newsdesk
81. Themeable static-site templates and a dark/light toggle. **S**
82. Full-text search on the published site (client-side index). **S**
83. Multi-author bylines with per-author pages and disclosure. **S**
84. Newsdesk source-diversity check: flag over-reliance on one outlet. **S**
85. Newsdesk fact-check pass that cites for each claim before publishing *(needs Scout)*. **M**
86. Topic-desk specialisation (finance, tech) with per-desk stances. **M**
87. Scheduled digest email of the week's commentary to the opt-in list. **S**
88. Comment moderation queue with the sanitiser already built. **S**
89. RSS/Atom output for the opinion blog and the serial. **S** *(partly done)*
90. Translation of the author's own pieces into other languages, labelled as translations. **M**

## J. Safety, reliability, and operations
91. Content-provenance manifest: sign published pieces so edits are detectable. **M**
92. Expand the adversarial drill with new obfuscation families each release. **S**
93. Red-team the newsdesk against planted misinformation sources. **M**
94. [done] Backup/restore of the whole data dir with integrity verification. **S**
95. [done] Config migration framework for schema changes across versions. **S**
96. Structured logging and an error budget the daemon respects. **S**
97. A "dry-run everything" mode for a safe first run of any automation. **S**
98. Property-based tests across the pipeline (extends the mutation gate). **M**
99. A one-command Windows installer update channel with signed releases. **M**
100. [done] A plain-language audit report: every automation, what it can do, what it cannot. **S**

---

Suggested order: finish the realism cluster (A, B, C), then the measurement-dependent items (E, H, D)
once real reader data flows, in parallel with foundations (G, J). The native-model items (F) are the
longest and least certain at GaugeLM's size — worth doing, but honestly scoped as research, not a
promise of parity with large models.
