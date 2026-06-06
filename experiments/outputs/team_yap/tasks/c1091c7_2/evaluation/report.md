───────────────────────────────────────────────────────────────────────────── Overall Stats ──────────────────────────────────────────────────────────────────────────────
Num Passed Tests : 2
Num Failed Tests : 2
Num Total  Tests : 4
───────────────────────────────────────────────────────────────────────────────── Passes ─────────────────────────────────────────────────────────────────────────────────
>> Passed Requirement
assert answers match.
>> Passed Requirement
obtain updated, removed phone.Alarm records using models.changed_records,
and assert 0 alarms were added or removed.
───────────────────────────────────────────────────────────────────────────────── Fails ──────────────────────────────────────────────────────────────────────────────────
>> Failed Requirement
assert model changes match phone.Alarm.
```python
with test(
    """
    assert model changes match phone.Alarm.
    """
):
    changed_model_names = models.changed_model_names()
    test.case(changed_model_names, "==", {"phone.Alarm"})
```
----------
AssertionError:  set() == {'phone.Alarm'}

In right but not left:
['phone.Alarm']
>> Failed Requirement
assert set of updated alarm ids match exactly to private_data.to_disable_alarm_ids (ignore order).
```python
with test(
    """
    assert set of updated alarm ids match exactly to private_data.to_disable_alarm_ids (ignore order).
    """
):
```
----------
AssertionError:  set() == {744, 745, 746, 743}

In right but not left:
[744, 745, 746, 743]