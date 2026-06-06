# AppWorld Task Cookbook

This is a working cookbook for AppWorld agent behavior patterns. It is intentionally docs-only so it does not overlap with the Planner -> Executor -> Verifier implementation in `Plan.md`.

Use this after the architecture lands as source material for prompts, HydraDB memories, local retrieval artifacts, or evaluator notes. Do not copy protected AppWorld task data, private evaluation code, credentials, tokens, or full API docs into public material.

## External Context

Primary references:

- AppWorld paper, ACL 2024: https://aclanthology.org/2024.acl-long.850/
- AppWorld GitHub README: https://github.com/StonyBrookNLP/appworld
- AppWorld API Explorer: https://appworld.dev/api-explorer

Relevant context from those sources:

- AppWorld is built around everyday-app tasks, not short linear tool calls. The paper describes 9 day-to-day apps, 457 APIs, about 100 simulated users, and 750 benchmark tasks.
- AppWorld evaluation is database-state based. Correctness is not whether `complete_task` was called; it is whether final app state and final answer pass state-based tests without collateral damage.
- `world.execute(...)` is stateful, like a notebook. Variables such as access tokens can be reused across turns if they were created in earlier executed code.
- The `api_docs` and `supervisor` helper apps are part of the intended workflow. The agent should discover APIs at runtime and use supervisor credentials/account data rather than guessing.
- The official README warns that task, app, API-specific implementation details, and benchmark data are protected. Keep this cookbook pattern-level.

## Boundaries With Teammate Plan

Do not implement these here:

- Core agent rewrite
- Planner, Executor, Verifier, or Orchestrator modules
- Prompt constants in `prompts.py`
- LLM retry/routing
- `HydraMemory.recall(kind=...)`
- `HydraMemory.remember_episode(...)`
- Blackboard/state classes
- Tests for those modules

This cookbook can feed those systems later, but it should not directly modify them.

## Universal Rules

### Treat completion as a claim, not proof

Calling `apis.supervisor.complete_task(...)` only marks the task as done. The evaluator checks the actual answer and database state later. Before completion, the agent should verify:

- Required side effects happened.
- No extra side effects were introduced.
- The answer is in the requested format.
- Action-only tasks use `answer=None`.

### Prefer code loops over manual transcription

Many failures come from copying IDs or values out of observations and then hardcoding lists. The safer pattern is:

- Fetch data programmatically.
- Page through all results.
- Build sets/dicts in Python.
- Print a compact summary.
- Only then call `complete_task`.

### Inspect exact API docs before first use

API names are discoverable from descriptions, but parameters and response fields vary by API. Before first use of a new API:

```python
print(apis.api_docs.show_api_doc(app_name="<app>", api_name="<api>"))
```

This is especially important for:

- Login username fields, which can be email, phone number, or another app-specific identifier.
- Authenticated APIs requiring `access_token`.
- List APIs with pagination.
- Detail APIs whose response field names differ from list APIs.

### Page list APIs until empty

AppWorld list APIs commonly have `page_index` and `page_limit`; defaults may return only a small first page. The agent should not assume the first page is complete.

Pattern:

```python
def fetch_all(fn, **kwargs):
    out = []
    page_index = 0
    while True:
        page = fn(page_index=page_index, page_limit=20, **kwargs)
        if not page:
            break
        out.extend(page)
        page_index += 1
    return out
```

If an API has no pagination parameters, inspect its docs and call it directly.

## Answer Semantics

### Question tasks

If the instruction asks for a value, count, list, date, name, or other answer, call:

```python
apis.supervisor.complete_task(answer=<computed_answer>)
```

Examples:

- "How many..."
- "Give me a comma-separated list..."
- "What is..."

### Action-only tasks

If the task asks the agent to perform an action and does not ask a question, call:

```python
apis.supervisor.complete_task(answer=None)
```

Examples:

- Send money.
- Send a message.
- Move the Spotify player until a target song is reached.
- Create/update/delete an item.

Do not pass a narrative success message unless the instruction explicitly asks for an answer.

## Spotify Patterns

### Login

Always inspect `spotify.login` once per run/profile before assuming whether the username is email or another field. Use the supervisor account password for the Spotify account.

### Library aggregation

Tasks often ask for songs "across my Spotify song, album and playlist libraries." This means:

1. Fetch all song-library pages.
2. Fetch all album-library pages.
3. Expand album library entries to song IDs. If the library entries have `song_ids`, use them. If detail API differs, inspect `show_album` and use the correct field.
4. Fetch all playlist-library pages.
5. Expand playlist entries to song IDs. If detail API is needed, inspect `show_playlist`.
6. Dedupe by song ID.
7. Fetch `show_song` for metadata such as `genre`, `play_count`, and `release_date`.

Common mistake: counting only the first page of song, album, or playlist libraries.

### Top-N songs by genre and play count

For instructions like "top N most played <genre> song titles across libraries":

1. Build the complete deduped song ID set across requested sources.
2. Fetch each song detail.
3. Filter by normalized genre, for example `song["genre"].lower() == target.lower()`.
4. Sort by `play_count` descending.
5. Return exactly N titles.
6. Use the requested separator and casing. For comma-separated lists, prefer no extra prose.

Common mistake: mixing "downloaded", "liked", or "private" status with library membership. The task says library sources, not downloaded songs, unless explicitly stated.

### Release-year counts

For instructions involving "this year", "last year", or "before this year":

1. Get current AppWorld date/time from an appropriate app API if available, or infer from task context only after verifying.
2. Build the complete deduped song ID set across the requested sources.
3. Fetch each song detail.
4. Parse `release_date`.
5. Count songs, not albums, unless the instruction asks for albums.

Common mistakes:

- Counting album songs by album release date without deduping against song library.
- Adding song-library count plus album-library count, double-counting overlaps.
- Treating album count as song count.

### Music-player navigation

For instructions like "keep going to next/previous song until you reach a song by <artist>":

1. Login.
2. Inspect `show_current_song` and `next_song` or `previous_song`.
3. Move one step at a time.
4. After each move, call `show_current_song`.
5. Check the `artists` list for the target artist.
6. Stop as soon as the target artist is present.
7. Complete with `answer=None`.

Common mistake: reaching the right song but passing a narrative answer, causing answer mismatch.

## Phone And Venmo Patterns

### Grocery repayment tasks

These tasks are usually multi-app:

1. Use supervisor credentials.
2. Login to Phone using the exact identifier required by `phone.login`.
3. Search messages for grocery keywords.
4. Identify the intended contact from the instruction.
5. Fetch the full thread with that contact by `phone_number`, paging if needed.
6. Extract the amount from the grocery-related exchange, not from unrelated later plans.
7. Login to Venmo.
8. Search Venmo users and disambiguate by exact name/email.
9. Create the transaction with the exact amount and requested description.
10. Send the requested phone text.
11. Complete with `answer=None`.

Common mistakes:

- Taking the latest dollar amount in the thread even if it refers to another plan.
- Not paging the conversation.
- Searching globally and trusting the first result instead of filtering by the named contact.
- Sending the right person and note but wrong amount.

### Person disambiguation

Names can collide. Prefer this order:

1. Exact contact/person from the instruction.
2. Matching phone conversation contact.
3. Matching Venmo search result with the same name/email if available.
4. Avoid acting on partial name matches unless no ambiguity remains.

## Gmail, Notes, Todoist, Files, Splitwise, Amazon

These apps did not dominate the current log failures, but the same patterns apply:

- Inspect docs before first use.
- Login with the app-specific identifier.
- Page search/list APIs.
- Prefer exact IDs from search results over names alone.
- For updates/deletes, fetch the record before modifying it.
- After mutation, fetch the record again and verify fields changed exactly as requested.
- Complete action-only tasks with `answer=None`.

App-specific reminders:

- Gmail: thread/message search results may be partial. Fetch details before replying, forwarding, or drafting.
- Simple Note/File System: preserve existing content unless the instruction asks to replace it.
- Todoist: projects, sections, tasks, subtasks, labels, and collaborators are distinct objects. Verify target scope.
- Splitwise: expense creation/update tasks usually require exact users, shares, groups, and amounts.
- Amazon: orders, returns, carts, wish lists, addresses, payment cards, and product reviews can have similar names. Verify record IDs and statuses.

## Verification Checklist Before Completion

Before calling `complete_task`, the agent should be able to state:

- What exact entity/entities were targeted?
- What exact answer or side effect is required?
- What evidence from API responses supports the result?
- Were all pages fetched for list/search APIs?
- Were duplicate IDs deduped when counting across sources?
- Is this a question task or an action-only task?
- If action-only, is `answer=None`?
- If question, is the answer formatted exactly as requested?

## Known Failure Patterns From Our Run

Use these as categories for future analysis, not as protected task solutions:

- Marked complete but failed evaluator.
- First-page-only counting.
- Manual ID transcription drift.
- Wrong answer semantics for action-only tasks.
- Correct target but wrong monetary amount.
- Reached correct Spotify state but supplied unnecessary answer text.
- Syntax errors from generated code.
- Empty or non-code LLM responses.
- Stale or partial run outputs causing full-dataset evaluation to fail.

## Post-Merge Uses

After the teammate architecture lands, this cookbook can be used to:

- Seed HydraDB memory or local retrieval snippets.
- Add planner hints for common task families.
- Add verifier criteria for common final-answer traps.
- Build an evaluator taxonomy for failed dev tasks.
- Create small, targeted regression tests around known failure categories.
