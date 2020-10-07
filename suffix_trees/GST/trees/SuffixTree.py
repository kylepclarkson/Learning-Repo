from trees.TreeADT import Tree

class SuffixTree(Tree):
    class _Node:
        """ A nonpublic wrapper for storing a node. """
        __slots__ = '_element', '_parent', '_children'

        def __init__(self, element, parent=None, children=None):
            self._element = element
            self._parent = parent
            self._children = {} if children is None else children

    # === An abstract representation of a element in the tree.
    class Position(Tree.Position):

        def __init__(self, container, node):
            self._container = container
            self._node = node

        def element(self):
            return self._node._element

        def __eq__(self, other):
            return type(other) is type(self) and other._node is self._node
    # === End Position Class ===

    # ================ Wrapper Functions ================
    def _validate(self, pos):
        """Unpack node from position, if it is valid.
        Returns _Node object.
        """
        if not isinstance(pos, self.Position):
            raise TypeError("pos must be of Position type")

        if pos._container is not self:
            raise ValueError('pos does not belong to this container')

        return pos._node

    def _make_position(self, node):
        """Pack node into position instance (or None if no node.)
        Returns Position object.
        """
        return self.Position(self, node) if node is not None else None
    # ================ End wrapper functions ================


    # ================ Constructors ================
    def __init__(self, e=None):
        """ Create tree with root element e. """
        self._root = None
        self._add_root(e)

    def __len__(self):
        return self._size

    # ================ Accessors ================
    def root(self):
        """Return root Position of tree. """
        return self._make_position(self._root)

    def parent(self, pos):
        """ Return the Position of pos' parent. """
        node = self._validate(pos)
        return self._make_position(node._parent)

    def children(self, pos):
        """ Generate the children of Position pos. """
        node = self._validate(pos)
        return [self._make_position(child) for child in node._children.keys()]
        # [self._make_position(child) for child in node._children]

    def num_children(self, pos):
        node = self._validate(pos)
        return len(node._children)

    # ================ Insert functions ================
    def _add_root(self, e):
        """ Place element e at root of empty tree. """
        if self._root is not None:
            raise ValueError('Root exists')

        self._size = 1
        self._root = self._Node(e)
        return self._make_position(self._root)

    def _add(self, pos, e):
        """ Create new child for Position pos with element e.
        Return the Position of new node.
        """
        parent_node = self._validate(pos)
        self._size += 1
        child = self._Node(e, parent_node)
        # add child to parent node
        parent_node._children[child] = child

        return self._make_position(child)

    def _replace(self, pos, e):
        """ Replace element at Position pos with e.
        Return the old element.
        """
        node = self._validate(pos)
        old = node._element
        node._element = e
        return old


# Testing
t = SuffixTree('root')
root = t.root()
b = t._add(root, 'b')
c = t._add(root, 'c')
d = t._add(root, 'd')
e = t._add(b, 'e')
f = t._add(e, 'f')
g = t._add(e, 'g')

h = t._replace(e, 'h')
[print(c.element()) for c in t.children(b)]
