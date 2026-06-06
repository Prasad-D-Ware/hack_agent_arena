───────────────────────────────────────────────────────────────────────────── Overall Stats ──────────────────────────────────────────────────────────────────────────────
Num Passed Tests : 2
Num Failed Tests : 6
Num Total  Tests : 8
───────────────────────────────────────────────────────────────────────────────── Passes ─────────────────────────────────────────────────────────────────────────────────
>> Passed Requirement
assert answers match.
>> Passed Requirement
assert 0 records have been updated or deleted from amazon.Address using models.changed_records.
───────────────────────────────────────────────────────────────────────────────── Fails ──────────────────────────────────────────────────────────────────────────────────
>> Failed Requirement
assert model changes match
amazon.Order, amazon.OrderItem, amazon.Product,
file_system.File, file_system.Directory, gmail.GlobalEmailThread, gmail.Attachment,
gmail.Email, gmail.UserEmailThread, ignoring amazon.CartEntry, amazon.Address.
```python
with test(
    """
    assert model changes match
    amazon.Order, amazon.OrderItem, amazon.Product,
    file_system.File, file_system.Directory, gmail.GlobalEmailThread, gmail.Attachment,
    gmail.Email, gmail.UserEmailThread, ignoring amazon.CartEntry, amazon.Address.
    """
):
```
----------
AssertionError:
set()
==
{'gmail.UserEmailThread', 'gmail.Email', 'gmail.GlobalEmailThread', 'file_system.Directory', 'gmail.Attachment', 'file_system.File', 'amazon.OrderItem', 'amazon.Product',
'amazon.Order'}

In right but not left:
['gmail.UserEmailThread', 'gmail.Email', 'gmail.GlobalEmailThread', 'file_system.Directory', 'gmail.Attachment', 'file_system.File', 'amazon.OrderItem', 'amazon.Product',
'amazon.Order']
>> Failed Requirement
assert there is 1 new amazon.Order using models.changed_records.
```python
with test(
    """
    assert there is 1 new amazon.Order using models.changed_records.
    """
):
    added_orders, _, _ = models.changed_records("amazon.Order")
    test.case(len(added_orders), "==", 1)
    order = added_orders[0]
```
----------
AssertionError:  0 == 1
>> Failed Requirement
assert this new order has 1 quantity of a single product.
```python
with test(
    """
    assert this new order has 1 quantity of a single product.
    """
):
    order_items = order.order_items
    test.case(len(order_items), "==", 1)
    order_item = order_items[0]
```
----------
>> Failed Requirement
assert the ordered product is of type data.public.product_type.
```python
with test(
    """
    assert the ordered product is of type data.public.product_type.
    """
):
    ordered_product = order.products[0]
    test.case(ordered_product.product_type, "==", public_data.product_type)
```
----------
>> Failed Requirement
assert the ordered product has price <= data.public.max_price.
```python
with test(
    """
    assert the ordered product has price <= data.public.max_price.
    """
):
    test.case(ordered_product.price, "<=", public_data.max_price)
```
----------
>> Failed Requirement
assert the ordered product's seller is in data.private.trusted_seller_ids.
```python
with test(
    """
    assert the ordered product's seller is in data.private.trusted_seller_ids.
    """
):
    test.case(ordered_product.seller.id, "in", private_data.trusted_seller_ids)
```
----------