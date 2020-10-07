from trees.Tree import Tree

class SuffixTree(Tree):

    def __init__(self, string=None):
        Tree.__init__(self, string)
        


# Testing
def test1():
    t = SuffixTree()
    root = t.root()
    b = t._add(root, 'b')
    c = t._add(root, 'c')
    d = t._add(root, 'd')
    e = t._add(b, 'e')
    f = t._add(e, 'f')
    g = t._add(e, 'g')

    h = t._replace(e, 'h')
    [print(c.element()) for c in t.children(b)]

def test2():
    t = SuffixTree()
    root = t.root()

    # insert axabx
    x1 = t._add(root, 'axabx')

    # insert xabx
    x2 = t._add(root, 'xabx')

    # TODO implement in _insert_between
    x3 = t._add(x1, 'xabx')
    x4 = t._add(x1, 'bx')
    t._replace(x1, 'a')

    print('root')
    # [print(c.element()) for c in t.children(root)]
    print(t._validate(x1)._children)
    print('x1')
    [print(c.element()) for c in t.children(x1)]


test2()
