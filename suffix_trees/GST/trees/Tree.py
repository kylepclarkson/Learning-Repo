"""Abstract class for Tree data structure. """
class TreeADT:

    # === Nest Position class (Container for node) ===
    class Position:
        """An abstracting representing a node of the tree. """

        def element(self):
            raise NotImplementedError()

        def __eq__(self, other):
            raise NotImplementedError()

        def __ne__(self, other):
            return not (self == other)

    # === Abstract methods to be implemented by subclass ===
    def root(self):
        raise NotImplementedError()

    def parent(self, pos):
        raise NotImplementedError()

    def children(self, pos):
        # Generate an iteration of Positions representing pos' children.
        raise NotImplementedError()

    def num_children(self, pos):
        raise NotImplementedError()

    def __len__(self):
        """Number of elements in tree."""
        raise NotImplementedError()

    # === Class methods ===
    def is_root(self, pos):
        return self.root() == pos

    def is_leaf(self, pos):
        return self.num_children() == 0

    def is_empty(self):
        return len(self) == 0

    def height(self, pos):
        """ Return height of tree rooted at pos."""
        if pos is None:
            p = self.root()

        if self.is_leaf(pos):
            return 0
        else:
            return 1 + max(self.height(c) for c in self.children(pos))

""" The Tree data structure. """
class Tree(TreeADT):
    class _Node:
        """ A nonpublic wrapper for storing a node. """
        __slots__ = '_element', '_parent', '_children'

        def __init__(self, element, parent=None, children=None):
            self._element = element
            self._parent = parent
            self._children = {} if children is None else children

        def __repr__(self):
            return  str(self._element)

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

    # ================ Wrapper Functions ================
    def _validate(self, pos):
        """Unpack node from position, if it is valid.
        Returns _Node object.
        """
        if not isinstance(pos, self.Position):
            raise TypeError("pos must be of Position type")

        if pos._container is not self:
            raise ValueError('pos does not belong to this container')

        # convention: when removing a node, set its parent to itself.
        if pos._node._parent is pos._node:
            raise ValueError('pos is no longer valid')

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

    def _remove(self, pos):
        """ Delete the node at Position pos.
        If pos has a single child, make the child's parent pos.
        If pos has more than one child, throw error.
        Return element that was stored at pos.
        """
        node = self._validate(pos)

        if self.num_children(pos) > 1:
            raise ValueError('pos have more than one child')

        if len(node._children) == 0:
            node._parent = node         # deprecate node.
        else:
            # single child. Swap
            child = node._children[0]
            child._parent = node._parent
            self._size -= 1
            node._parent = node         # deprecate node.
        return node._element

    def _insert_between(self, pos_parent, pos_child, e):
        """ Insert new Position with element e in between pos_parent and pos_child.
        pos_child will be a child of the new Position, and the new Position will
        be a child of pos_parent.
        Return Position of new node.
        """

        parent_node = self._validate(pos_parent)
        child_node = self._validate(pos_child)
        # Check that parent-child relationship holds.
        if not child_node in parent_node._children:
            raise ValueError('pos_parent is not a parent of pos_child')
        self._size += 1
        new_node = self._Node(e, parent_node)
        child_node._parent = new_node
        return self._make_position(new_node)