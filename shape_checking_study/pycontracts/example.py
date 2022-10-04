from contracts import contract

class A:
    b = 3
    b_2 = 4

@contract(a="attr(b:int)")  # Works.
def foo(a: A) -> None:
    pass

foo(A())

@contract(a="attr(b_2:int)")  # AssertionError: Bug in syntax
def foo2(a: A) -> None:
    pass

foo2(A())
