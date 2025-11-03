"""
Elementary Data Structures Implementation and Analysis
Includes: Arrays, Stacks, Queues, Linked Lists, and Rooted Trees
"""

# ============================================================================
# 1. ARRAYS AND MATRICES
# ============================================================================

class DynamicArray:
    """
    Dynamic array implementation with automatic resizing.
    Time Complexity:
    - Access: O(1)
    - Insertion (end): O(1) amortized, O(n) worst case
    - Insertion (arbitrary): O(n)
    - Deletion: O(n)
    - Search: O(n)
    """
    
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.size = 0
        self.array = [None] * capacity
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        """Access element at index - O(1)"""
        if not 0 <= index < self.size:
            raise IndexError("Index out of bounds")
        return self.array[index]
    
    def __setitem__(self, index, value):
        """Set element at index - O(1)"""
        if not 0 <= index < self.size:
            raise IndexError("Index out of bounds")
        self.array[index] = value
    
    def append(self, value):
        """Insert at end - O(1) amortized"""
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        self.array[self.size] = value
        self.size += 1
    
    def insert(self, index, value):
        """Insert at arbitrary position - O(n)"""
        if not 0 <= index <= self.size:
            raise IndexError("Index out of bounds")
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        
        # Shift elements to the right
        for i in range(self.size, index, -1):
            self.array[i] = self.array[i - 1]
        
        self.array[index] = value
        self.size += 1
    
    def delete(self, index):
        """Delete element at index - O(n)"""
        if not 0 <= index < self.size:
            raise IndexError("Index out of bounds")
        
        value = self.array[index]
        
        # Shift elements to the left
        for i in range(index, self.size - 1):
            self.array[i] = self.array[i + 1]
        
        self.array[self.size - 1] = None
        self.size -= 1
        
        # Shrink if size is 1/4 of capacity
        if self.size > 0 and self.size == self.capacity // 4:
            self._resize(self.capacity // 2)
        
        return value
    
    def _resize(self, new_capacity):
        """Resize internal array - O(n)"""
        new_array = [None] * new_capacity
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array
        self.capacity = new_capacity
    
    def __str__(self):
        return str([self.array[i] for i in range(self.size)])


class Matrix:
    """
    2D Matrix implementation with basic operations.
    Time Complexity:
    - Access: O(1)
    - Row/Column operations: O(n) or O(m)
    """
    
    def __init__(self, rows, cols, default=0):
        self.rows = rows
        self.cols = cols
        self.data = [[default for _ in range(cols)] for _ in range(rows)]
    
    def __getitem__(self, pos):
        """Access element at (row, col) - O(1)"""
        row, col = pos
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("Index out of bounds")
        return self.data[row][col]
    
    def __setitem__(self, pos, value):
        """Set element at (row, col) - O(1)"""
        row, col = pos
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("Index out of bounds")
        self.data[row][col] = value
    
    def get_row(self, row):
        """Get entire row - O(m) where m is number of columns"""
        if not 0 <= row < self.rows:
            raise IndexError("Row index out of bounds")
        return self.data[row][:]
    
    def get_col(self, col):
        """Get entire column - O(n) where n is number of rows"""
        if not 0 <= col < self.cols:
            raise IndexError("Column index out of bounds")
        return [self.data[row][col] for row in range(self.rows)]
    
    def transpose(self):
        """Transpose matrix - O(n*m)"""
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result[j, i] = self.data[i][j]
        return result
    
    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.data])


# ============================================================================
# 2. STACKS (Array-based)
# ============================================================================

class ArrayStack:
    """
    Stack implementation using dynamic array.
    Time Complexity:
    - Push: O(1) amortized
    - Pop: O(1)
    - Peek: O(1)
    - isEmpty: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.items = DynamicArray()
    
    def push(self, item):
        """Push item onto stack - O(1) amortized"""
        self.items.append(item)
    
    def pop(self):
        """Pop item from stack - O(1)"""
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self.items.delete(len(self.items) - 1)
    
    def peek(self):
        """View top item without removing - O(1)"""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self.items[len(self.items) - 1]
    
    def is_empty(self):
        """Check if stack is empty - O(1)"""
        return len(self.items) == 0
    
    def size(self):
        """Return size of stack - O(1)"""
        return len(self.items)
    
    def __str__(self):
        return f"Stack({self.items})"


# ============================================================================
# 3. QUEUES (Array-based with circular buffer)
# ============================================================================

class ArrayQueue:
    """
    Queue implementation using circular array.
    Time Complexity:
    - Enqueue: O(1) amortized
    - Dequeue: O(1)
    - Peek: O(1)
    - isEmpty: O(1)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.array = [None] * capacity
        self.front = 0
        self.rear = 0
        self.size = 0
    
    def enqueue(self, item):
        """Add item to rear of queue - O(1) amortized"""
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        
        self.array[self.rear] = item
        self.rear = (self.rear + 1) % self.capacity
        self.size += 1
    
    def dequeue(self):
        """Remove item from front of queue - O(1)"""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        
        item = self.array[self.front]
        self.array[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        
        if self.size > 0 and self.size == self.capacity // 4:
            self._resize(self.capacity // 2)
        
        return item
    
    def peek(self):
        """View front item without removing - O(1)"""
        if self.is_empty():
            raise IndexError("Peek from empty queue")
        return self.array[self.front]
    
    def is_empty(self):
        """Check if queue is empty - O(1)"""
        return self.size == 0
    
    def _resize(self, new_capacity):
        """Resize internal array - O(n)"""
        new_array = [None] * new_capacity
        for i in range(self.size):
            new_array[i] = self.array[(self.front + i) % self.capacity]
        self.array = new_array
        self.front = 0
        self.rear = self.size
        self.capacity = new_capacity
    
    def __str__(self):
        items = [self.array[(self.front + i) % self.capacity] 
                 for i in range(self.size)]
        return f"Queue({items})"


# ============================================================================
# 4. LINKED LISTS
# ============================================================================

class Node:
    """Node for singly linked list"""
    def __init__(self, data):
        self.data = data
        self.next = None


class SinglyLinkedList:
    """
    Singly Linked List implementation.
    Time Complexity:
    - Access: O(n)
    - Insertion (head): O(1)
    - Insertion (tail): O(1) with tail pointer, O(n) without
    - Insertion (arbitrary): O(n)
    - Deletion (head): O(1)
    - Deletion (arbitrary): O(n)
    - Search: O(n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def is_empty(self):
        """Check if list is empty - O(1)"""
        return self.head is None
    
    def insert_head(self, data):
        """Insert at beginning - O(1)"""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        if self.tail is None:
            self.tail = new_node
        self.size += 1
    
    def insert_tail(self, data):
        """Insert at end - O(1)"""
        new_node = Node(data)
        if self.is_empty():
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1
    
    def insert_at(self, index, data):
        """Insert at specific position - O(n)"""
        if index < 0 or index > self.size:
            raise IndexError("Index out of bounds")
        
        if index == 0:
            self.insert_head(data)
            return
        
        if index == self.size:
            self.insert_tail(data)
            return
        
        new_node = Node(data)
        current = self.head
        for _ in range(index - 1):
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
        self.size += 1
    
    def delete_head(self):
        """Delete first node - O(1)"""
        if self.is_empty():
            raise IndexError("Delete from empty list")
        
        data = self.head.data
        self.head = self.head.next
        if self.head is None:
            self.tail = None
        self.size -= 1
        return data
    
    def delete_at(self, index):
        """Delete node at specific position - O(n)"""
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds")
        
        if index == 0:
            return self.delete_head()
        
        current = self.head
        for _ in range(index - 1):
            current = current.next
        
        data = current.next.data
        current.next = current.next.next
        
        if current.next is None:
            self.tail = current
        
        self.size -= 1
        return data
    
    def search(self, data):
        """Search for data in list - O(n)"""
        current = self.head
        index = 0
        while current:
            if current.data == data:
                return index
            current = current.next
            index += 1
        return -1
    
    def traverse(self):
        """Return list of all elements - O(n)"""
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result
    
    def __str__(self):
        return " -> ".join(map(str, self.traverse())) + " -> None"
    
    def __len__(self):
        return self.size


# ============================================================================
# 5. LINKED LIST BASED STACK AND QUEUE
# ============================================================================

class LinkedStack:
    """
    Stack implementation using linked list.
    Time Complexity: All operations O(1)
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.list = SinglyLinkedList()
    
    def push(self, item):
        """Push item onto stack - O(1)"""
        self.list.insert_head(item)
    
    def pop(self):
        """Pop item from stack - O(1)"""
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self.list.delete_head()
    
    def peek(self):
        """View top item - O(1)"""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self.list.head.data
    
    def is_empty(self):
        """Check if empty - O(1)"""
        return self.list.is_empty()
    
    def size(self):
        """Get size - O(1)"""
        return len(self.list)


class LinkedQueue:
    """
    Queue implementation using linked list.
    Time Complexity: All operations O(1)
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.list = SinglyLinkedList()
    
    def enqueue(self, item):
        """Add item to rear - O(1)"""
        self.list.insert_tail(item)
    
    def dequeue(self):
        """Remove item from front - O(1)"""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self.list.delete_head()
    
    def peek(self):
        """View front item - O(1)"""
        if self.is_empty():
            raise IndexError("Peek from empty queue")
        return self.list.head.data
    
    def is_empty(self):
        """Check if empty - O(1)"""
        return self.list.is_empty()


# ============================================================================
# 6. ROOTED TREES (Using Linked Lists)
# ============================================================================

class TreeNode:
    """Node for rooted tree with arbitrary number of children"""
    def __init__(self, data):
        self.data = data
        self.children = []  # List of child nodes
        self.parent = None


class RootedTree:
    """
    Rooted tree implementation using linked structure.
    Time Complexity:
    - Insert: O(1) if parent is known, O(n) if search needed
    - Delete: O(n) for subtree deletion
    - Search: O(n)
    - Traversal: O(n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self, root_data):
        self.root = TreeNode(root_data)
    
    def insert_child(self, parent_node, child_data):
        """Insert child node under parent - O(1)"""
        child_node = TreeNode(child_data)
        child_node.parent = parent_node
        parent_node.children.append(child_node)
        return child_node
    
    def search(self, data, node=None):
        """Search for node with given data - O(n)"""
        if node is None:
            node = self.root
        
        if node.data == data:
            return node
        
        for child in node.children:
            result = self.search(data, child)
            if result:
                return result
        
        return None
    
    def preorder_traversal(self, node=None):
        """Preorder traversal (root, children) - O(n)"""
        if node is None:
            node = self.root
        
        result = [node.data]
        for child in node.children:
            result.extend(self.preorder_traversal(child))
        
        return result
    
    def postorder_traversal(self, node=None):
        """Postorder traversal (children, root) - O(n)"""
        if node is None:
            node = self.root
        
        result = []
        for child in node.children:
            result.extend(self.postorder_traversal(child))
        result.append(node.data)
        
        return result
    
    def level_order_traversal(self):
        """Level-order (breadth-first) traversal - O(n)"""
        if not self.root:
            return []
        
        result = []
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            result.append(node.data)
            queue.extend(node.children)
        
        return result
    
    def height(self, node=None):
        """Calculate height of tree - O(n)"""
        if node is None:
            node = self.root
        
        if not node.children:
            return 0
        
        return 1 + max(self.height(child) for child in node.children)
    
    def count_nodes(self, node=None):
        """Count total nodes in tree - O(n)"""
        if node is None:
            node = self.root
        
        count = 1
        for child in node.children:
            count += self.count_nodes(child)
        
        return count


# ============================================================================
# 7. DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_all():
    """Demonstrate all data structures"""
    
    print("=" * 70)
    print("ELEMENTARY DATA STRUCTURES DEMONSTRATION")
    print("=" * 70)
    
    # Dynamic Array
    print("\n1. DYNAMIC ARRAY")
    print("-" * 70)
    arr = DynamicArray()
    for i in range(5):
        arr.append(i * 10)
    print(f"After appending 0,10,20,30,40: {arr}")
    arr.insert(2, 15)
    print(f"After inserting 15 at index 2: {arr}")
    arr.delete(3)
    print(f"After deleting index 3: {arr}")
    print(f"Access index 2: {arr[2]}")
    
    # Matrix
    print("\n2. MATRIX")
    print("-" * 70)
    matrix = Matrix(3, 3)
    for i in range(3):
        for j in range(3):
            matrix[i, j] = i * 3 + j + 1
    print("3x3 Matrix:")
    print(matrix)
    print(f"Row 1: {matrix.get_row(1)}")
    print(f"Column 2: {matrix.get_col(2)}")
    
    # Array Stack
    print("\n3. ARRAY-BASED STACK")
    print("-" * 70)
    stack = ArrayStack()
    for i in [1, 2, 3, 4, 5]:
        stack.push(i)
        print(f"Pushed {i}: {stack}")
    print(f"Peek: {stack.peek()}")
    print(f"Pop: {stack.pop()}")
    print(f"After pop: {stack}")
    
    # Array Queue
    print("\n4. ARRAY-BASED QUEUE (Circular)")
    print("-" * 70)
    queue = ArrayQueue()
    for i in [1, 2, 3, 4, 5]:
        queue.enqueue(i)
        print(f"Enqueued {i}: {queue}")
    print(f"Peek: {queue.peek()}")
    print(f"Dequeue: {queue.dequeue()}")
    print(f"After dequeue: {queue}")
    
    # Singly Linked List
    print("\n5. SINGLY LINKED LIST")
    print("-" * 70)
    ll = SinglyLinkedList()
    for i in [1, 2, 3, 4, 5]:
        ll.insert_tail(i)
    print(f"After inserting 1-5: {ll}")
    ll.insert_head(0)
    print(f"After inserting 0 at head: {ll}")
    ll.insert_at(3, 2.5)
    print(f"After inserting 2.5 at index 3: {ll}")
    print(f"Search for 3: index {ll.search(3)}")
    ll.delete_at(3)
    print(f"After deleting index 3: {ll}")
    
    # Linked Stack
    print("\n6. LINKED LIST-BASED STACK")
    print("-" * 70)
    lstack = LinkedStack()
    for i in [10, 20, 30]:
        lstack.push(i)
    print(f"Size: {lstack.size()}")
    print(f"Pop: {lstack.pop()}")
    print(f"Peek: {lstack.peek()}")
    
    # Linked Queue
    print("\n7. LINKED LIST-BASED QUEUE")
    print("-" * 70)
    lqueue = LinkedQueue()
    for i in [10, 20, 30]:
        lqueue.enqueue(i)
    print(f"Dequeue: {lqueue.dequeue()}")
    print(f"Peek: {lqueue.peek()}")
    
    # Rooted Tree
    print("\n8. ROOTED TREE")
    print("-" * 70)
    tree = RootedTree("A")
    b = tree.insert_child(tree.root, "B")
    c = tree.insert_child(tree.root, "C")
    d = tree.insert_child(tree.root, "D")
    tree.insert_child(b, "E")
    tree.insert_child(b, "F")
    tree.insert_child(c, "G")
    
    print(f"Preorder: {tree.preorder_traversal()}")
    print(f"Postorder: {tree.postorder_traversal()}")
    print(f"Level-order: {tree.level_order_traversal()}")
    print(f"Height: {tree.height()}")
    print(f"Total nodes: {tree.count_nodes()}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_all()