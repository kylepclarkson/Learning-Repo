from trees import TreeADT

class SuffixTree(TreeADT):
    class _Node:
        """ A nonpublic wrapper for storing a node. """
        __slots__ = '_element', '_parent', '_children'

        def __init__(self, element, parent=None, children=None):
            self._element = element
            self._parent = parent
            self._children = children

    # === An abstract representation of a element in the tree.
    class Position(TreeADT.Position):

        def __init__(self, container, node):
            self._container = container
            self._node = node

        def element(self):
            return self._node._element

        def __eq__(self, other):
            return type(other) is type(self) and other._node is self._node
    # === End Position Class ===

    def _validate(self, pos):
        """Unpack node from position, if it is valid."""
        if not isinstance(pos, self.Position):
            raise TypeError("pos must be of Position type")

        if pos._container is not self:
            raise ValueError('pos does not belong to this container')

        return pos._node

    def _make_position(self, node):
        """Pack node into position instance (or None if no node.)"""
        return self.Position(self, node) if node is not None else None


    # === Constructor ===
    def __init__(self):
        """ Create empty tree. """
        self._root = None
        self._size = 0

    # === public accessors ===
    def __len__(self):
        return self._size

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
        yield [self._make_position(child) for child in node._children]

    def num_children(self, pos):
        children = self.children(pos)
        count = 0
        for c in children:
            count += 1
        return count

t = SuffixTree()
root = t.root()
print(t.num_children(root))