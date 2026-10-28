# Step 18 comparison: metadata filtering

Baseline recorded **before** step 18 (with step 16's multi-query and vector search, k=5 per query), so the
step 18 code can be run on the same questions and compared.

"Gold" = an article the MultiHop-RAG dataset lists as evidence for the question.
"Contains fact" = the chunk holds the exact sentence the dataset lists as the evidence fact.

---

## Question 1: Davis Cup + rugby

**Question:** Does 'The Age' article suggest that Australia's Davis Cup team is aiming for an improvement in their
performance compared to the previous year, while the 'Sporting News' article indicates that the South Africa
national rugby team has already achieved an improvement to reach the Rugby World Cup semi-finals?

**Correct answer:** Yes

### Facts the answer needs

| # | Source | Article | Fact |
|---|---|---|---|
| A | The Age | ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final | "Hopefully we can go one better this year," he added, recalling the 2-0 defeat by Canada in 2022. |
| B | Sporting News | Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal | …in South Africa, they face a side who have taken their game to new heights just to reach the semis. |

### Before step 18: query `"The Age" Australia Davis Cup team performance improvement`

| Rank | Source | Article | Chunk starts with | Useful? |
|---|---|---|---|---|
| 1 | The Age | Biggest win of my career… | "It will be Australia's 49th appearance in the Davis Cup final…" | ✅ contains fact A |
| 2 | The Age | Biggest win of my career… | "While the Italy tie didn't finish until 8.15pm…" | gold article, no fact |
| 3 | The Roar | Tennis News: Kyrgios escalates beef with Becker… | "Andy Murray has added another event to his Australian Open warm-up…" | ❌ other publication |
| 4 | The Age | Biggest win of my career… | "It's nerves, it's pressure – you're playing for everybody back home…" | gold article, no fact |
| 5 | The Guardian | West Indies v England: fifth and deciding men's T20 cricket international – live | "Jos Buttler's reaction: We probably didn't adapt quickly enough…" | ❌ other publication (cricket) |

### Before step 18: query `"Sporting News" South Africa rugby team Rugby World Cup semi-finals`

| Rank | Source | Article | Chunk starts with | Useful? |
|---|---|---|---|---|
| 1 | Sporting News | Where to watch England vs South Africa… | "The 2019 Rugby World Cup final between England and South Africa…" | gold article, no fact |
| 2 | TalkSport | Liverpool vs Everton begins epic Saturday with World Cup semi-final… | "If you've had your football fix for the day, there's also… rugby…" | ❌ other publication |
| 3 | Sporting News | Where to watch England vs South Africa… | "The last time Cheslin Kolbe played England at a Rugby World Cup…" | ✅ contains fact B |
| 4 | Sporting News | Where to watch England vs South Africa… | "South Africa replacements: Deon Fourie, Ox Nche…" | gold article, no fact |
| 5 | Sporting News | Where to watch England vs South Africa… | "The Springboks have done it tough, too…" | gold article, no fact |

**Verdict before step 18:** ✅ both facts retrieved. The other publications (The Roar, The Guardian, TalkSport)
are noise only; step 17's reranking scored them 0.

**What step 18 should change:** no chunks from other publications; facts A and B still retrieved.

---

## Question 2: betting platform

**Question:** Which online betting platform provides a welcome bonus of up to $1000 in bonus bets for new customers'
first losses, runs NBA betting promotions, and is anticipated to extend the same sign-up offer to new users in
Vermont, as reported by both CBSSports.com and Sporting News?

**Correct answer:** Caesars Sportsbook

### Facts the answer needs

| # | Source | Article | Fact |
|---|---|---|---|
| A | CBSSports.com | 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet | New customers can get up to $1000 in bonus bets if they lose their first bet. |
| B | Sporting News | NBA Rookie of the Year Odds - Wembanyama the early favorite | Additionally, Caesars has a competitive welcome bonus for new players and runs NBA betting promos for existing players. |
| C | Sporting News | Vermont Sportsbook Promos and Sports Betting Launch News | We expect Caesars to offer the same offer in VT is has elsewhere, with new sign-ups eligible to receive up to a $1,000 Bonus Bet if their first bet loses. |

### Before step 18: only the original question was searched (step 16's query writer crashed → fallback)

| Rank | Source | Article | Chunk starts with | Useful? |
|---|---|---|---|---|
| 1 | Sporting News | Vermont Sportsbook Promos… | "Caesars Vermont promo code… We expect Caesars to offer the same offer in VT…" | ✅ contains fact C |
| 2 | Sporting News | Vermont Sportsbook Promos… | "The best sports betting sites in the country are focused on Vermont…" | same article, no fact |
| 3 | Sporting News | Vermont Sportsbook Promos… | "In other states, the ESPN Bet welcome offer…" | same article, no fact |
| 4 | Sporting News | Vermont Sportsbook Promos… | "You only have a week to use all $200 of your Bonus Bets…" | same article, no fact |
| 5 | Sporting News | Vermont Sportsbook Promos… | "DraftKings Vermont promo code…" | same article, no fact |

**Verdict before step 18:** ❌ 1 of 3 facts. All 5 chunks come from one article.
- Fact A (CBSSports.com) missing: no CBSSports.com chunk at all, although the question names it.
- Fact B (NBA Rookie of the Year article) missing.

**What step 18 should change:** a `source = CBSSports.com` filter should bring back the Kentucky article (fact A).

---

## Summary before step 18

| Question | Facts needed | Retrieved | Problem |
|---|---|---|---|
| Davis Cup + rugby | 2 | 2 ✅ | noise from other publications |
| Betting platform | 3 | 1 ❌ | CBSSports.com and NBA articles missing |

---

## After step 18

Step 18's LLM plans searches as (query, source). A search with a source only looks at that publication's
chunks; the original question is still searched too, unfiltered.

### Question 1: Davis Cup + rugby (13 unique chunks)

| Search | Filter | Results | Facts |
|---|---|---|---|
| Original question | none | The Roar ×2 (Eddie Jones, All Blacks), TalkSport (Liverpool vs Everton), Sporting News ×2 (gold) | — |
| `Australia Davis Cup team performance improvement compared to previous year` | source = The Age | **The Age ×5**, all from the gold article | ✅ A at rank 1 |
| `South Africa national rugby team reach Rugby World Cup semi-finals` | source = Sporting News | **Sporting News ×5**, all from the gold article | ✅ B at rank 4 |

**Verdict:** ✅ both facts, as before. The filtered searches return only the right publication (before: The Roar,
The Guardian and TalkSport got in). The remaining noise comes only from the unfiltered original-question search.

### Question 2: betting platform (22 unique chunks)

| Search | Filter | Results | Facts |
|---|---|---|---|
| Original question | none | Sporting News: Vermont article ×5 | ✅ C |
| `online betting platform welcome bonus up to $1000 in bonus bets for first losses` | none | Vermont article ×4, Monday Night Football bonus offers ×1 | — |
| `NBA betting promotions` | none | Sporting News: "The Best NBA Betting Sites and Apps for the 2023-24 Season" ×5 (**not gold**) | ❌ B missed |
| `betting platform sign-up offer Vermont` | none | Vermont article ×5 | ✅ C |
| `betting platform $1000 bonus bets Vermont` | source = **CBSSports.com** | **CBSSports.com: "2023 Kentucky online sports betting sites…" ×5 (gold)** | ✅ **A at rank 1** (new) |
| `betting platform $1000 bonus bets Vermont` | source = Sporting News | Vermont article ×5 | ✅ C |

**Verdict:** improved from 1 to 2 of 3 facts. The CBSSports.com filter brought back the Kentucky article
(fact A), which no search found before.
- Fact B is still missing: the "NBA betting promotions" search had no filter and found a different Sporting News
  NBA article. Fact B's chunk ("…Caesars has a competitive welcome bonus… runs NBA betting promos…") is in
  "NBA Rookie of the Year Odds", whose title says nothing about promotions.
- Cost: 6 searches and 22 chunks (before: 1 search, 5 chunks), so reranking (step 17) matters more.

## Summary: before vs after

| Question | Facts needed | Before step 18 | After step 18 | Chunks before → after |
|---|---|---|---|---|
| Davis Cup + rugby | 2 | 2 ✅ (noise from 3 other publications) | 2 ✅ (filtered searches: right publication only) | 12 → 13 |
| Betting platform | 3 | 1 ❌ | **2** (A new; B still missing) | 5 → 22 |
