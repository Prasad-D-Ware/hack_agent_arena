───────────────────────────────────────────────────────────────────────────── Overall Stats ──────────────────────────────────────────────────────────────────────────────
Num Passed Tests : 2
Num Failed Tests : 8
Num Total  Tests : 10
───────────────────────────────────────────────────────────────────────────────── Passes ─────────────────────────────────────────────────────────────────────────────────
>> Passed Requirement
assert answers match.
>> Passed Requirement
obtain added, updated, deleted gmail.Draft using models.changed_records, and
assert 0 is updated, 0 is deleted.
───────────────────────────────────────────────────────────────────────────────── Fails ──────────────────────────────────────────────────────────────────────────────────
>> Failed Requirement
assert model changes match gmail.Draft.
```python
with test(
    """
    assert model changes match gmail.Draft.
    """
):
    changed_model_names = models.changed_model_names()
    test.case(changed_model_names, "==", {"gmail.Draft"})
```
----------
AssertionError:  set() == {'gmail.Draft'}

In right but not left:
['gmail.Draft']
>> Failed Requirement
assert the number of added drafts is equal to the len(private_data.expected_draft_data)
```python
with test(
    """
    assert the number of added drafts is equal to the len(private_data.expected_draft_data)
    """
):
    test.case(len(added_drafts), "==", len(private_data.expected_draft_data))
```
----------
AssertionError:  0 == 5
>> Failed Requirement
assert the added drafts subjects match the subjects from
private_data.expected_draft_data (ignore order, normalize_text=True)
```python
with test(
    """
    assert the added drafts subjects match the subjects from
    private_data.expected_draft_data (ignore order, normalize_text=True)
    """
):
    added_drafts_subjects = list_of(added_drafts, "subject")
    expected_draft_subjects = list_of(private_data.expected_draft_data, "subject")
```
----------
AssertionError:
[]
==
["meeting 'customer feedback and improvement session' starting soon", "meeting 'it security briefing and training' starting soon", "meeting 'leadership training session'
starting soon", "meeting 'project kickoff meeting' starting soon", "meeting 'sales pipeline analysis meeting' starting soon"]

In right but not left:
["meeting 'customer feedback and improvement session' starting soon", "meeting 'it security briefing and training' starting soon", "meeting 'leadership training session'
starting soon", "meeting 'project kickoff meeting' starting soon", "meeting 'sales pipeline analysis meeting' starting soon"]

Original values:
[]
==
["Meeting 'Leadership Training Session' Starting Soon", "Meeting 'Project Kickoff Meeting' Starting Soon", "Meeting 'Customer Feedback and Improvement Session' Starting
Soon", "Meeting 'IT Security Briefing and Training' Starting Soon", "Meeting 'Sales Pipeline Analysis Meeting' Starting Soon"]
>> Failed Requirement
assert all the added drafts bodies are empty (after stripping)
```python
with test(
    """
    assert all the added drafts bodies are empty (after stripping)
    """
):
    added_drafts_bodies =
    test.case(added_drafts_bodies, "all ==", "")
```
----------
AssertionError: The left modifier is 'all' with allow_left_empty=False, but left operand is empty.
>> Failed Requirement
assert the added drafts have scheduled_send_at matching the scheduled_send_at
from private_data.expected_draft_data (ignore order, round_to="minute")
```python
with test(
    """
    assert the added drafts have scheduled_send_at matching the scheduled_send_at
    from private_data.expected_draft_data (ignore order, round_to="minute")
    """
):
    added_drafts_scheduled_send_at = list_of(added_drafts, "scheduled_send_at")
    expected_draft_scheduled_send_at = list_of(
```
----------
AssertionError:
[]
==
[DateTime(2023, 5, 23, 13, 40, 0), DateTime(2023, 5, 24, 15, 40, 0), DateTime(2023, 5, 25, 12, 40, 0), DateTime(2023, 5, 26, 10, 40, 0), DateTime(2023, 5, 26, 14, 40, 0)]

In right but not left:
[DateTime(2023, 5, 23, 13, 40, 0), DateTime(2023, 5, 24, 15, 40, 0), DateTime(2023, 5, 25, 12, 40, 0), DateTime(2023, 5, 26, 10, 40, 0), DateTime(2023, 5, 26, 14, 40, 0)]
>> Failed Requirement
assert the added drafts have recipient_ids matching the recipient_ids
from private_data.expected_draft_data (ignore order)
```python
with test(
    """
    assert the added drafts have recipient_ids matching the recipient_ids
    from private_data.expected_draft_data (ignore order)
    """
):
    added_drafts_recipient_ids = list_of(added_drafts, "recipient_ids")
    expected_draft_recipient_ids = list_of(private_data.expected_draft_data, "recipient_ids")
```
----------
AssertionError:  [] == [[25, 36, 41], [25, 36, 59], [25, 70], [59, 80], [59, 99]]

In right but not left:
[[25, 36, 41], [25, 36, 59], [25, 70], [59, 80], [59, 99]]

Original values:
[] == [[25, 36, 59], [36, 41, 25], [59, 99], [80, 59], [25, 70]]
>> Failed Requirement
assert the subject to recipient_ids dict from the added drafts
matches that from the private_data.expected_draft_data.
```python
with test(
    """
    assert the subject to recipient_ids dict from the added drafts
    matches that from the private_data.expected_draft_data.
    """
):
    subject_to_recipient_ids = dict_of(added_drafts, "subject", "recipient_ids")
    expected_subject_to_recipient_ids = dict_of(
```
----------
AssertionError:
{}
==
{"Meeting 'Leadership Training Session' Starting Soon": [25, 36, 59], "Meeting 'Project Kickoff Meeting' Starting Soon": [36, 41, 25], "Meeting 'Customer Feedback and
Improvement Session' Starting Soon": [59, 99], "Meeting 'IT Security Briefing and Training' Starting Soon": [80, 59], "Meeting 'Sales Pipeline Analysis Meeting' Starting
Soon": [25, 70]}
>> Failed Requirement
assert the subject to scheduled_send_at dict from the added drafts
matches that from the private_data.expected_draft_data.
```python
with test(
    """
    assert the subject to scheduled_send_at dict from the added drafts
    matches that from the private_data.expected_draft_data.
    """
):
    subject_to_scheduled_send_at = dict_of(added_drafts, "subject", "scheduled_send_at")
    expected_subject_to_scheduled_send_at = dict_of(
```
----------
AssertionError:
{}
==
{"Meeting 'Leadership Training Session' Starting Soon": DateTime(2023, 5, 23, 13, 40, 0), "Meeting 'Project Kickoff Meeting' Starting Soon": DateTime(2023, 5, 24, 15, 40,
0), "Meeting 'Customer Feedback and Improvement Session' Starting Soon": DateTime(2023, 5, 25, 12, 40, 0), "Meeting 'IT Security Briefing and Training' Starting Soon":
DateTime(2023, 5, 26, 10, 40, 0), "Meeting 'Sales Pipeline Analysis Meeting' Starting Soon": DateTime(2023, 5, 26, 14, 40, 0)}