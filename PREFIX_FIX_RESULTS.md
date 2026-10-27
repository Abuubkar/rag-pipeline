# Prefix fix: steps 10–15 before and after

## What changed

nomic-embed-text expects `search_document: ` before stored text and `search_query: ` before questions (step 8).
Our `NomicEmbeddings` (step 9) was meant to add them, but:

| | Before (bug) | After (fix) |
|---|---|---|
| Stored chunks (Postgres) | no prefix: `PGVectorStore` called `aembed_documents`, which we hadn't overridden | `search_document: …` |
| Questions (Postgres) | no prefix: same reason (`aembed_query`) | `search_query: …` |
| Questions (step 9, in memory) | `search_document: search_query: …` (both) | `search_query: …` |

All 8,448 chunks were re-embedded (about 6 minutes). Same chunks, same ids, same questions: only the vectors changed.

## Summary

| Step | Question | Before | After | Verdict |
|---|---|---|---|---|
| 10 | Kindle and Echo discounts | Kindle deals article 3rd | Prime Day (Oct 2023) 2nd, Kindle deals 4th | ≈ same |
| 10, 11 | Why is the FTC suing Amazon? | the Amazon/FTC article ranked 1–4 | 2 Google-antitrust articles ranked 1–2, Amazon/FTC 3–5. Both Google chunks contain the sentence *"The FTC sued Amazon in September for using anticompetitive and unfair strategies…"* | ≈ different, not wrong |
| 11 | MMR, same question | 4 articles | 4 articles (Google antitrust now first) | ≈ same |
| 12 | SBF, "Verge and TechCrunch" (multi-hop) | "Insufficient information." | **"Sam Bankman-Fried (also referred to as SBF)"** | ✅ better |
| 12 | Pizza (off-topic) | refused | refused | same |
| 13 | FTC with citations | correct, cited [1][2][5] | correct, cited [1][2][3][4]; [1][2] are the Google articles, which do contain the cited sentence | same |
| 14 | keyword search | — | identical (keyword search doesn't use vectors) | same |
| 14, 15 | SBF (both questions), vector side | 4–5 SBF articles | 5 SBF articles, with new ones ("The jury finally hears from Sam Bankman-Fried", "SBF Trial: latest updates") | ✅ better |
| 15 | hybrid, FTC question | Google articles 1–2 | Google articles 1–2 | same |

Takeaways:
- Clear win: the multi-hop SBF question went from "Insufficient information." to the right answer, because the vector side found more SBF articles.
- Looks worse but isn't: the FTC question now ranks two Google-trial articles first, and each contains a one-sentence answer about the FTC suing Amazon. Check the chunk text before calling a result wrong.
- Three hand-picked questions can't prove retrieval got better overall. Step 24 (evaluation on the 2,556 gold questions) is how to measure it.
- Distances got smaller overall (e.g. best Kindle match 0.247 → 0.195). Scores from different embedding setups **can't be compared**; only the order within one run means something.

---

## Raw outputs

### step10_postgres

**Before**

```text
8448 chunks · 8448 already in Postgres · 0 to embed

Q: Where can I get discounts on Kindle and Echo devices?
  0.247  200+ of the best deals from Amazon's Cyber Monday sale
  0.272  The 53 best Black Friday deals we could find at Amazon, Walm
  0.274  Here are the best Kindle deals right now
  0.275  The 30 best Amazon Prime Day deals for October 2023
  0.278  Here are the best deals on headphones and wireless earbuds w

Q: Why is the FTC suing Amazon?
  0.226  Amazon sellers sound off on the FTC's 'long-overdue' antitru
  0.240  Amazon sellers sound off on the FTC's 'long-overdue' antitru
  0.260  Amazon sellers sound off on the FTC's 'long-overdue' antitru
  0.265  Amazon sellers sound off on the FTC's 'long-overdue' antitru
  0.284  5 things we learned so far about the Google antitrust case
```

**After**

```text
8448 chunks · 8448 already in Postgres · 0 to embed

Q: Where can I get discounts on Kindle and Echo devices?
  0.195  200+ of the best deals from Amazon's Cyber Monday sale
  0.233  The 30 best Amazon Prime Day deals for October 2023
  0.244  The 53 best Black Friday deals we could find at Amazon, Walm
  0.251  Here are the best Kindle deals right now
  0.253  Amazon brings conversational AI to kids with launch of ‘Expl

Q: Why is the FTC suing Amazon?
  0.196  5 things we learned so far about the Google antitrust case
  0.207  More from the US v Google trial: vertical search, pre-instal
  0.210  Amazon sellers sound off on the FTC's 'long-overdue' antitru
  0.267  Amazon sellers sound off on the FTC's 'long-overdue' antitru
  0.300  Amazon sellers sound off on the FTC's 'long-overdue' antitru
```

### step11_retriever

**Before**

```text
similarity: 5 documents from 2 articles
  Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
  Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
  Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
  Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
  5 things we learned so far about the Google antitrust case

mmr: 5 documents from 4 articles
  Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
  Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
  The inside story of Dave Clark's tumultuous last days at Flexport
  ChatGPT: Everything you need to know about the AI-powered chatbot
  Today the FTX jury suffered through a code review

Where can I get discounts on Kindle and Echo devices?
  → 200+ of the best deals from Amazon's Cyber Monday sale

Who won the Raiders vs Lions game?
  → Raiders vs. Lions live score, updates, highlights from NFL 'Monday Nig
```

**After**

```text
similarity: 5 documents from 3 articles
  5 things we learned so far about the Google antitrust case
  More from the US v Google trial: vertical search, pre-installs and the
  Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
  Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
  Amazon sellers sound off on the FTC's 'long-overdue' antitrust case

mmr: 5 documents from 4 articles
  5 things we learned so far about the Google antitrust case
  Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
  Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
  The inside story of Dave Clark's tumultuous last days at Flexport
  Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas

Where can I get discounts on Kindle and Echo devices?
  → 200+ of the best deals from Amazon's Cyber Monday sale

Who won the Raiders vs Lions game?
  → Raiders vs. Lions live score, updates, highlights from NFL 'Monday Nig
```

### step12_rag_chain

**Before**

```text
Q: Why is the FTC suing Amazon?
A: The FTC is suing Amazon for allegedly using its monopoly power to squeeze merchants and thwart rivals. Specifically, the complaint alleges that Amazon:

*   Uses an "anti-discounting strategy" and a "massive web-crawling apparatus" to track online prices and stifle competition.
*   Punishes third-party sellers who offer cheaper products elsewhere by threatening to disqualify them from appearing in the "buy box."
*   Steadily hikes the fees it charges sellers and prevents them from discounting on other sites, which the FTC argues creates an "artificial price floor" and leads to elevated prices across the web.
*   Uses anticompetitive and unfair strategies to illegally maintain its monopoly power.

Q: Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch?
A: Insufficient information.

Q: What is the capital of Australia's best-selling pizza brand?
A: Insufficient information.
```

**After**

```text
Q: Why is the FTC suing Amazon?
A: The FTC sued Amazon for using anticompetitive and unfair strategies to illegally maintain its monopoly power. Specifically, the agency accused Amazon of:

*   **Squeezing merchants and thwarting rivals:** The FTC alleges Amazon uses its monopoly power to harm competitors.
*   **Anti-discounting strategy:** The FTC claims Amazon uses a "massive web-crawling apparatus" to track online prices and punishes third-party sellers who offer cheaper products elsewhere by threatening to disqualify them from the "buy box."
*   **Creating an "artificial price floor":** By hiking fees and preventing sellers from discounting on other sites, the FTC argues Amazon's tactics lead to elevated prices across the web and a degraded shopping experience for consumers.

Q: Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch?
A: Sam Bankman-Fried (also referred to as SBF) is the individual associated with the cryptocurrency industry facing trial for fraud and conspiracy charges.

Q: What is the capital of Australia's best-selling pizza brand?
A: Insufficient information.
```

### step13_citations

**Before**

```text
Keys returned: ['question', 'documents', 'context', 'answer']

Answer:
The FTC is suing Amazon for allegedly using its monopoly power to squeeze merchants and thwart rivals [2]. Specifically, the complaint alleges that Amazon:

*   Uses an "anti-discounting strategy" and a "massive web-crawling apparatus" to track online prices and stifle competition [2].
*   Punishes third-party sellers who offer cheaper products on other sites by threatening to disqualify them from the "buy box" [2].
*   Uses tactics that result in "elevated prices across the web" by steadily hiking fees for sellers and preventing them from discounting on other sites, which creates an "artificial price floor" [1, 2].
*   Employs "anticompetitive and unfair strategies" to illegally maintain its monopoly power [5].

Sources:
  [1] Amazon sellers sound off on the FTC's 'long-overdue' antitru — Cnbc | World Business News Leader
      https://www.cnbc.com/2023/10/06/amazon-sellers-sound-off-on-the-ftcs-long-overdue-antitrust-case.html
  [2] Amazon sellers sound off on the FTC's 'long-overdue' antitru — Cnbc | World Business News Leader
      https://www.cnbc.com/2023/10/06/amazon-sellers-sound-off-on-the-ftcs-long-overdue-antitrust-case.html
  [3] Amazon sellers sound off on the FTC's 'long-overdue' antitru — Cnbc | World Business News Leader
      https://www.cnbc.com/2023/10/06/amazon-sellers-sound-off-on-the-ftcs-long-overdue-antitrust-case.html
  [4] Amazon sellers sound off on the FTC's 'long-overdue' antitru — Cnbc | World Business News Leader
      https://www.cnbc.com/2023/10/06/amazon-sellers-sound-off-on-the-ftcs-long-overdue-antitrust-case.html
  [5] 5 things we learned so far about the Google antitrust case — TechCrunch
      https://techcrunch.com/2023/10/30/5-things-we-learned-so-far-about-the-google-antitrust-case/
```

**After**

```text
Keys returned: ['question', 'documents', 'context', 'answer']

Answer:
The FTC sued Amazon because the agency accused the company of using anticompetitive and unfair strategies to illegally maintain its monopoly power [1, 2]. Specifically, the FTC alleges that Amazon:

*   Uses its monopoly power to squeeze merchants and thwart rivals [3].
*   Employs an "anti-discounting strategy" and a "massive web-crawling apparatus" to track online prices and stifle competition [3].
*   Punishes third-party sellers who offer cheaper products on other sites by threatening to disqualify them from the "buy box" [3].
*   Increases fees charged to sellers and prevents them from discounting products on other websites, which the FTC argues creates an "artificial price floor" and leads to elevated prices across the web [3, 4].

Sources:
  [1] 5 things we learned so far about the Google antitrust case — TechCrunch
      https://techcrunch.com/2023/10/30/5-things-we-learned-so-far-about-the-google-antitrust-case/
  [2] More from the US v Google trial: vertical search, pre-instal — TechCrunch
      https://techcrunch.com/2023/11/13/more-from-the-us-v-google-trial-vertical-search-pre-installs-and-the-case-of-firefox-yahoo/
  [3] Amazon sellers sound off on the FTC's 'long-overdue' antitru — Cnbc | World Business News Leader
      https://www.cnbc.com/2023/10/06/amazon-sellers-sound-off-on-the-ftcs-long-overdue-antitrust-case.html
  [4] Amazon sellers sound off on the FTC's 'long-overdue' antitru — Cnbc | World Business News Leader
      https://www.cnbc.com/2023/10/06/amazon-sellers-sound-off-on-the-ftcs-long-overdue-antitrust-case.html
  [5] Amazon sellers sound off on the FTC's 'long-overdue' antitru — Cnbc | World Business News Leader
      https://www.cnbc.com/2023/10/06/amazon-sellers-sound-off-on-the-ftcs-long-overdue-antitrust-case.html
```

### step14_keyword

**Before**

```text
Q: Kindle Echo discounts
  keyword: 2 results
    The best anti-Prime Big Deal Day deals
    The 26 best Amazon Prime Day deals you can still shop for today
  vector: 5 results
    200+ of the best deals from Amazon's Cyber Monday sale
    The 53 best Black Friday deals we could find at Amazon, Walmart, Targe
    The 53 best Black Friday deals we could find at Amazon, Walmart, Targe
    Here are the best deals on headphones and wireless earbuds we could fi
    Here are the best deals on headphones and wireless earbuds we could fi

Q: Sam Bankman-Fried trial
  keyword: 5 results
    The FTX trial is bigger than Sam Bankman-Fried
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok
    SBF, riding high on FTX, reportedly offered $5B to Trump to cancel Pre
    SBF’s trial starts soon, but how did he — and FTX — get here?
  vector: 5 results
    How is it still getting worse for Sam Bankman-Fried?
    The FTX trial is bigger than Sam Bankman-Fried
    SBF, riding high on FTX, reportedly offered $5B to Trump to cancel Pre
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening

Q: Why is the FTC suing Amazon?
  keyword: 2 results
    More from the US v Google trial: vertical search, pre-installs and the
    5 things we learned so far about the Google antitrust case
  vector: 5 results
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    5 things we learned so far about the Google antitrust case

Q: Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch?
  keyword: 0 results
  vector: 5 results
    The FTX trial is bigger than Sam Bankman-Fried
    Sam Bankman-Fried was a terrible boyfriend
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    How is it still getting worse for Sam Bankman-Fried?
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
```

**After**

```text
Q: Kindle Echo discounts
  keyword: 2 results
    The best anti-Prime Big Deal Day deals
    The 26 best Amazon Prime Day deals you can still shop for today
  vector: 5 results
    200+ of the best deals from Amazon's Cyber Monday sale
    The 53 best Black Friday deals we could find at Amazon, Walmart, Targe
    The 30 best Amazon Prime Day deals for October 2023
    The 53 best Black Friday deals we could find at Amazon, Walmart, Targe
    Here are the best deals on headphones and wireless earbuds we could fi

Q: Sam Bankman-Fried trial
  keyword: 5 results
    The FTX trial is bigger than Sam Bankman-Fried
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok
    SBF, riding high on FTX, reportedly offered $5B to Trump to cancel Pre
    SBF’s trial starts soon, but how did he — and FTX — get here?
  vector: 5 results
    How is it still getting worse for Sam Bankman-Fried?
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    The FTX trial is bigger than Sam Bankman-Fried
    SBF Trial: The latest updates from the FTX collapse’s courtroom drama
    In the end, the FTX trial was about the friends screwed along the way

Q: Why is the FTC suing Amazon?
  keyword: 2 results
    More from the US v Google trial: vertical search, pre-installs and the
    5 things we learned so far about the Google antitrust case
  vector: 5 results
    5 things we learned so far about the Google antitrust case
    More from the US v Google trial: vertical search, pre-installs and the
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case

Q: Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch?
  keyword: 0 results
  vector: 5 results
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    How is it still getting worse for Sam Bankman-Fried?
    Sam Bankman-Fried was a terrible boyfriend
    The jury finally hears from Sam Bankman-Fried
    SBF, riding high on FTX, reportedly offered $5B to Trump to cancel Pre
```

### step15_hybrid

**Before**

```text
Q: Kindle Echo discounts
  keyword: 2 results
    The best anti-Prime Big Deal Day deals
    The 26 best Amazon Prime Day deals you can still shop for today
  vector: 5 results
    200+ of the best deals from Amazon's Cyber Monday sale
    The 53 best Black Friday deals we could find at Amazon, Walmart, Targe
    The 53 best Black Friday deals we could find at Amazon, Walmart, Targe
    Here are the best deals on headphones and wireless earbuds we could fi
    Here are the best deals on headphones and wireless earbuds we could fi
  hybrid: 5 results
    200+ of the best deals from Amazon's Cyber Monday sale
    The best anti-Prime Big Deal Day deals
    The 53 best Black Friday deals we could find at Amazon, Walmart, Targe
    The 26 best Amazon Prime Day deals you can still shop for today
    The 53 best Black Friday deals we could find at Amazon, Walmart, Targe

Q: Sam Bankman-Fried trial
  keyword: 5 results
    The FTX trial is bigger than Sam Bankman-Fried
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok
    SBF, riding high on FTX, reportedly offered $5B to Trump to cancel Pre
    SBF’s trial starts soon, but how did he — and FTX — get here?
  vector: 5 results
    How is it still getting worse for Sam Bankman-Fried?
    The FTX trial is bigger than Sam Bankman-Fried
    SBF, riding high on FTX, reportedly offered $5B to Trump to cancel Pre
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening
  hybrid: 5 results
    The FTX trial is bigger than Sam Bankman-Fried
    SBF, riding high on FTX, reportedly offered $5B to Trump to cancel Pre
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    How is it still getting worse for Sam Bankman-Fried?
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas

Q: Why is the FTC suing Amazon?
  keyword: 2 results
    More from the US v Google trial: vertical search, pre-installs and the
    5 things we learned so far about the Google antitrust case
  vector: 5 results
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    5 things we learned so far about the Google antitrust case
  hybrid: 5 results
    5 things we learned so far about the Google antitrust case
    More from the US v Google trial: vertical search, pre-installs and the
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case

Q: Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch?
  keyword: 0 results
  vector: 5 results
    The FTX trial is bigger than Sam Bankman-Fried
    Sam Bankman-Fried was a terrible boyfriend
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    How is it still getting worse for Sam Bankman-Fried?
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
  hybrid: 5 results
    The FTX trial is bigger than Sam Bankman-Fried
    Sam Bankman-Fried was a terrible boyfriend
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    How is it still getting worse for Sam Bankman-Fried?
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
```

**After**

```text
Q: Kindle Echo discounts
  keyword: 2 results
    The best anti-Prime Big Deal Day deals
    The 26 best Amazon Prime Day deals you can still shop for today
  vector: 5 results
    200+ of the best deals from Amazon's Cyber Monday sale
    The 53 best Black Friday deals we could find at Amazon, Walmart, Targe
    The 30 best Amazon Prime Day deals for October 2023
    The 53 best Black Friday deals we could find at Amazon, Walmart, Targe
    Here are the best deals on headphones and wireless earbuds we could fi
  hybrid: 5 results
    200+ of the best deals from Amazon's Cyber Monday sale
    The best anti-Prime Big Deal Day deals
    The 53 best Black Friday deals we could find at Amazon, Walmart, Targe
    The 26 best Amazon Prime Day deals you can still shop for today
    The 30 best Amazon Prime Day deals for October 2023

Q: Sam Bankman-Fried trial
  keyword: 5 results
    The FTX trial is bigger than Sam Bankman-Fried
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok
    SBF, riding high on FTX, reportedly offered $5B to Trump to cancel Pre
    SBF’s trial starts soon, but how did he — and FTX — get here?
  vector: 5 results
    How is it still getting worse for Sam Bankman-Fried?
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    The FTX trial is bigger than Sam Bankman-Fried
    SBF Trial: The latest updates from the FTX collapse’s courtroom drama
    In the end, the FTX trial was about the friends screwed along the way
  hybrid: 5 results
    The FTX trial is bigger than Sam Bankman-Fried
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    SBF, riding high on FTX, reportedly offered $5B to Trump to cancel Pre
    SBF’s trial starts soon, but how did he — and FTX — get here?
    How is it still getting worse for Sam Bankman-Fried?

Q: Why is the FTC suing Amazon?
  keyword: 2 results
    More from the US v Google trial: vertical search, pre-installs and the
    5 things we learned so far about the Google antitrust case
  vector: 5 results
    5 things we learned so far about the Google antitrust case
    More from the US v Google trial: vertical search, pre-installs and the
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
  hybrid: 5 results
    5 things we learned so far about the Google antitrust case
    More from the US v Google trial: vertical search, pre-installs and the
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case
    Amazon sellers sound off on the FTC's 'long-overdue' antitrust case

Q: Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch?
  keyword: 0 results
  vector: 5 results
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    How is it still getting worse for Sam Bankman-Fried?
    Sam Bankman-Fried was a terrible boyfriend
    The jury finally hears from Sam Bankman-Fried
    SBF, riding high on FTX, reportedly offered $5B to Trump to cancel Pre
  hybrid: 5 results
    Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas
    How is it still getting worse for Sam Bankman-Fried?
    Sam Bankman-Fried was a terrible boyfriend
    The jury finally hears from Sam Bankman-Fried
    SBF, riding high on FTX, reportedly offered $5B to Trump to cancel Pre
```
