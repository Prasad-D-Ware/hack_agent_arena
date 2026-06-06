───────────────────────────────────────────────────────────────────────────── Overall Stats ──────────────────────────────────────────────────────────────────────────────
Num Passed Tests : 2
Num Failed Tests : 6
Num Total  Tests : 8
───────────────────────────────────────────────────────────────────────────────── Passes ─────────────────────────────────────────────────────────────────────────────────
>> Passed Requirement
assert answers match.
>> Passed Requirement
assert 0 records have been deleted from phone.UserTextMessage using models.changed_records.
───────────────────────────────────────────────────────────────────────────────── Fails ──────────────────────────────────────────────────────────────────────────────────
>> Failed Requirement
assert model changes match phone.GlobalTextMessage, phone.UserTextMessage
```python
with test(
    """
    assert model changes match phone.GlobalTextMessage, phone.UserTextMessage
    """
):
    changed_model_names = models.changed_model_names()
    test.case(changed_model_names, "==", {"phone.GlobalTextMessage", "phone.UserTextMessage"})
```
----------
AssertionError:  set() == {'phone.GlobalTextMessage', 'phone.UserTextMessage'}

In right but not left:
['phone.GlobalTextMessage', 'phone.UserTextMessage']
>> Failed Requirement
assert 1 record has been added to phone.GlobalTextMessage using models.changed_records.
```python
with test(
    """
    assert 1 record has been added to phone.GlobalTextMessage using models.changed_records.
    """
):
    added_messages, _, _ = models.changed_records("phone.GlobalTextMessage")
    test.case(len(added_messages), "==", 1)
```
----------
AssertionError:  0 == 1
>> Failed Requirement
assert the receiver of the message is the public_data.partner_relation.
```python
with test(
    """
    assert the receiver of the message is the public_data.partner_relation.
    """
):
    message = added_messages[0]
    test.case(message.receiver.id, "==", main_user.partner.id)
```
----------
>> Failed Requirement
assert all keys of private_data.product_name_to_total_price are mentioned in the message (ignoring case).
```python
with test(
    """
    assert all keys of private_data.product_name_to_total_price are mentioned in the message (ignoring case).
    """
):
    test.case(
        private_data.product_name_to_total_price.keys(),
        "all in",
```
----------
>> Failed Requirement
assert all values of private_data.product_name_to_total_price are mentioned in the message.
```python
with test(
    """
    assert all values of private_data.product_name_to_total_price are mentioned in the message.
    """
):
    test.case(
        ,
        "all in",
```
----------
>> Failed Requirement
parse the message content into product_name_to_total_price map and assert
it matches data.private.product_name_to_total_price (ignore_case, tolerance=1.01)
```python
with test(
    """
    parse the message content into product_name_to_total_price map and assert
    it matches data.private.product_name_to_total_price (ignore_case, tolerance=1.01)
    """
):
    message_content = message.message
    product_name_to_total_price = {}
```
----------