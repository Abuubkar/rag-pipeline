# Step 16 fix: also search the original question

## What changed

```python
# before: only the LLM's queries are searched
RunnablePassthrough.assign(queries=query_writer | (lambda result: result.queries))

# after: the original question first, then the LLM's queries
RunnablePassthrough.assign(queries=lambda inputs: [inputs["question"]] + query_writer.invoke(inputs).queries)
```

Why: multi-query is only as good as the queries the LLM writes. If they're poor, chunks the original
question would have found get lost. Searching the original too means multi-query can only add chunks.

Same model, temperature 0, same questions: the LLM wrote the same queries both times.

## Summary

| Question | One search | Multi-query before | Multi-query after |
|---|---|---|---|
| SBF ("The Verge and TechCrunch") | 5 chunks: Verge ✅ TechCrunch ✅ | 6 chunks: Verge ✅ **TechCrunch ❌** | 8 chunks: Verge ✅ TechCrunch ✅ |
| Davis Cup (The Age) + rugby (Sporting News) | 5 chunks: The Age ❌ Sporting News ✅ | 10 chunks: The Age ✅ Sporting News ✅ | 12 chunks: The Age ✅ Sporting News ✅ |

Takeaways:
- SBF: the LLM's 2 queries ignored the two publications, so "before" lost the TechCrunch chunk. "After" keeps it.
- Davis Cup: the LLM's queries were good, so "before" already found both gold articles. "After" adds the
  original question's 5 chunks, 2 of them new (Eddie Jones, All Blacks): more chunks, a bit more noise.
- The cost of the fix: one extra search per question, and longer context for the LLM to read.
- Still open: the LLM wrote 2 queries although the schema asked for 3. A field description is a request,
  not a rule.

---

## Raw outputs

### Before

```text
Q: Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch?

  one search: 5 chunks
    Cnbc | World Busines  Sex, Signal messages, and sabotaging FTX's code: SBF execs a
    The Verge             How is it still getting worse for Sam Bankman-Fried?
    The Verge             Sam Bankman-Fried was a terrible boyfriend
    The Verge             The jury finally hears from Sam Bankman-Fried
    TechCrunch            SBF, riding high on FTX, reportedly offered $5B to Trump to 

  multi-query: ['cryptocurrency fraud conspiracy trial The Verge TechCrunch', 'cryptocurrency executive fraud conspiracy charges news']
  → 6 unique chunks
    Cnbc | World Busines  Sex, Signal messages, and sabotaging FTX's code: SBF execs a
    The Verge             Sam Bankman-Fried was a terrible boyfriend
    Cnbc | World Busines  Sex, Signal messages, and sabotaging FTX's code: SBF execs a
    Cnbc | World Busines  Sex, Signal messages, and sabotaging FTX's code: SBF execs a
    The Verge             How is it still getting worse for Sam Bankman-Fried?
    Cnbc | World Busines  Sex, Signal messages, and sabotaging FTX's code: SBF execs a

Q: Does 'The Age' article suggest that Australia's Davis Cup team is aiming for an improvement in their performance compared to the previous year, while the 'Sporting News' article indicates that the South Africa national rugby team has already achieved an improvement to reach the Rugby World Cup semi-finals?

  one search: 5 chunks
    The Roar | Sports Wr  Eddie Jones says he needs to ‘give myself an uppercut’ but ‘
    Sporting News         Where to watch England vs South Africa: Live stream, TV chan
    Sporting News         Where to watch England vs South Africa: Live stream, TV chan
    TalkSport             Liverpool vs Everton begins epic Saturday with World Cup sem
    The Roar | Sports Wr  ‘It happens’: All Blacks defend star centre after spray as C

  multi-query: ['"The Age" Australia Davis Cup team performance improvement', '"Sporting News" South Africa rugby team Rugby World Cup semi-finals']
  → 10 unique chunks
    The Age               ‘Biggest win of my career’: De Minaur, Popyrin power Austral
    The Age               ‘Biggest win of my career’: De Minaur, Popyrin power Austral
    The Roar | Sports Wr  Tennis News: Kyrgios escalates beef with Becker, change ‘war
    The Age               ‘Biggest win of my career’: De Minaur, Popyrin power Austral
    The Guardian          West Indies v England: fifth and deciding men’s T20 cricket 
    Sporting News         Where to watch England vs South Africa: Live stream, TV chan
    TalkSport             Liverpool vs Everton begins epic Saturday with World Cup sem
    Sporting News         Where to watch England vs South Africa: Live stream, TV chan
    Sporting News         Where to watch England vs South Africa: Live stream, TV chan
    Sporting News         Where to watch England vs South Africa: Live stream, TV chan
```

### After

```text
Q: Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch?

  one search: 5 chunks
    Cnbc | World Busines  Sex, Signal messages, and sabotaging FTX's code: SBF execs a
    The Verge             How is it still getting worse for Sam Bankman-Fried?
    The Verge             Sam Bankman-Fried was a terrible boyfriend
    The Verge             The jury finally hears from Sam Bankman-Fried
    TechCrunch            SBF, riding high on FTX, reportedly offered $5B to Trump to 

  multi-query: ['Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch?', 'cryptocurrency fraud conspiracy trial The Verge TechCrunch', 'cryptocurrency executive fraud conspiracy charges news']
  → 8 unique chunks
    Cnbc | World Busines  Sex, Signal messages, and sabotaging FTX's code: SBF execs a
    The Verge             How is it still getting worse for Sam Bankman-Fried?
    The Verge             Sam Bankman-Fried was a terrible boyfriend
    The Verge             The jury finally hears from Sam Bankman-Fried
    TechCrunch            SBF, riding high on FTX, reportedly offered $5B to Trump to 
    Cnbc | World Busines  Sex, Signal messages, and sabotaging FTX's code: SBF execs a
    Cnbc | World Busines  Sex, Signal messages, and sabotaging FTX's code: SBF execs a
    Cnbc | World Busines  Sex, Signal messages, and sabotaging FTX's code: SBF execs a

Q: Does 'The Age' article suggest that Australia's Davis Cup team is aiming for an improvement in their performance compared to the previous year, while the 'Sporting News' article indicates that the South Africa national rugby team has already achieved an improvement to reach the Rugby World Cup semi-finals?

  one search: 5 chunks
    The Roar | Sports Wr  Eddie Jones says he needs to ‘give myself an uppercut’ but ‘
    Sporting News         Where to watch England vs South Africa: Live stream, TV chan
    Sporting News         Where to watch England vs South Africa: Live stream, TV chan
    TalkSport             Liverpool vs Everton begins epic Saturday with World Cup sem
    The Roar | Sports Wr  ‘It happens’: All Blacks defend star centre after spray as C

  multi-query: ["Does 'The Age' article suggest that Australia's Davis Cup team is aiming for an improvement in their performance compared to the previous year, while the 'Sporting News' article indicates that the South Africa national rugby team has already achieved an improvement to reach the Rugby World Cup semi-finals?", '"The Age" Australia Davis Cup team performance improvement', '"Sporting News" South Africa rugby team Rugby World Cup semi-finals']
  → 12 unique chunks
    The Roar | Sports Wr  Eddie Jones says he needs to ‘give myself an uppercut’ but ‘
    Sporting News         Where to watch England vs South Africa: Live stream, TV chan
    Sporting News         Where to watch England vs South Africa: Live stream, TV chan
    TalkSport             Liverpool vs Everton begins epic Saturday with World Cup sem
    The Roar | Sports Wr  ‘It happens’: All Blacks defend star centre after spray as C
    The Age               ‘Biggest win of my career’: De Minaur, Popyrin power Austral
    The Age               ‘Biggest win of my career’: De Minaur, Popyrin power Austral
    The Roar | Sports Wr  Tennis News: Kyrgios escalates beef with Becker, change ‘war
    The Age               ‘Biggest win of my career’: De Minaur, Popyrin power Austral
    The Guardian          West Indies v England: fifth and deciding men’s T20 cricket 
    Sporting News         Where to watch England vs South Africa: Live stream, TV chan
    Sporting News         Where to watch England vs South Africa: Live stream, TV chan
```
