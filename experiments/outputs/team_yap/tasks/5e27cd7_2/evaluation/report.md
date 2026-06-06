───────────────────────────────────────────────────────────────────────────── Overall Stats ──────────────────────────────────────────────────────────────────────────────
Num Passed Tests : 3
Num Failed Tests : 1
Num Total  Tests : 4
───────────────────────────────────────────────────────────────────────────────── Passes ─────────────────────────────────────────────────────────────────────────────────
>> Passed Requirement
assert answers match.
>> Passed Requirement
assert model changes match gmail.Draft.
>> Passed Requirement
obtain added, updated, deleted gmail.Draft records using models.changed_records,
and assert 0 have been updated or added.
───────────────────────────────────────────────────────────────────────────────── Fails ──────────────────────────────────────────────────────────────────────────────────
>> Failed Requirement
if public_data.operation is "and",
assert private_data.empty_subject_and_body_draft_ids match the deleted draft IDs (ignore order).
otherwise,
assert private_data.empty_subject_or_body_draft_ids match the deleted draft IDs (ignore order).
```python
with test(
    """
    if public_data.operation is "and",
    assert private_data.empty_subject_and_body_draft_ids match the deleted draft IDs (ignore order).
    otherwise,
    assert private_data.empty_subject_or_body_draft_ids match the deleted draft IDs (ignore order).
    """
):
```
----------
AssertionError:  [27, 85] == [24, 27, 52, 84, 85, 209, 293, 476]

In right but not left:
[24, 52, 84, 209, 293, 476]

Original values:
[85, 27] == [476, 209, 52, 293, 24, 85, 84, 27]