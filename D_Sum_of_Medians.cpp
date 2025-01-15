#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

// Constants
const int MAXN = 1e5 + 5;
const ll INF = 1e18;

// Splay Tree Node Structure
struct Node {
    int leftChild;  // Index of left child in the 'tree' array
    int rightChild; // Index of right child in the 'tree' array
    int parent;     // Index of parent node
    int size;       // Size of the subtree rooted at this node
    ll sum[5];      // Sum of values based on position modulo 5
    ll value;       // Value stored in this node

    // Constructor for creating a new node with a specific value
    Node() : leftChild(0), rightChild(0), parent(0), size(1), value(0) {
        memset(sum, 0, sizeof(sum));
    }

    Node(ll val) : leftChild(0), rightChild(0), parent(0), size(1), value(val) {
        memset(sum, 0, sizeof(sum));
        sum[0] = val;
    }
};

// Global variables
Node treeNodes[MAXN];
int totalNodes = 0; // Total number of nodes created
int root = 0;        // Index of the root node

// Helper macros for accessing children
#define LS(x) treeNodes[x].leftChild
#define RS(x) treeNodes[x].rightChild

// Function to update the size and sum[5] of a node
inline void updateNode(int x) {
    if (x == 0) return; // Null node

    treeNodes[x].size = 1; // Current node counts as size 1

    // Reset sum array
    memset(treeNodes[x].sum, 0, sizeof(treeNodes[x].sum));

    // Calculate sum based on left subtree
    if (LS(x)) {
        treeNodes[x].size += treeNodes[LS(x)].size;
        for (int i = 0; i < 5; ++i) {
            treeNodes[x].sum[i] += treeNodes[LS(x)].sum[i];
        }
    }

    // Determine the position of the current node within its group
    int leftSize = LS(x) ? treeNodes[LS(x)].size : 0;
    int positionMod5 = (leftSize) % 5;

    // Add current node's value to the appropriate sum slot
    treeNodes[x].sum[positionMod5] += treeNodes[x].value;

    // Calculate the shift for the right subtree sums
    int shift = (leftSize + 1) % 5;

    // Add right subtree's sum with shifted positions
    if (RS(x)) {
        for (int i = 0; i < 5; ++i) {
            int shiftedIndex = (i + shift) % 5;
            treeNodes[x].sum[shiftedIndex] += treeNodes[RS(x)].sum[i];
        }
        treeNodes[x].size += treeNodes[RS(x)].size;
    }
}

// Function to perform a left rotation
inline void rotateLeft(int x) {
    int y = treeNodes[x].parent;
    int z = treeNodes[y].parent;

    // y becomes the left child of x
    treeNodes[y].rightChild = treeNodes[x].leftChild;
    if (treeNodes[x].leftChild)
        treeNodes[treeNodes[x].leftChild].parent = y;

    treeNodes[x].leftChild = y;
    treeNodes[y].parent = x;

    // Update parent of x to z
    treeNodes[x].parent = z;
    if (z) {
        if (treeNodes[z].leftChild == y)
            treeNodes[z].leftChild = x;
        else
            treeNodes[z].rightChild = x;
    }

    // Update the subtree information
    updateNode(y);
    updateNode(x);
}

// Function to perform a right rotation
inline void rotateRight(int x) {
    int y = treeNodes[x].parent;
    int z = treeNodes[y].parent;

    // y becomes the right child of x
    treeNodes[y].leftChild = treeNodes[x].rightChild;
    if (treeNodes[x].rightChild)
        treeNodes[treeNodes[x].rightChild].parent = y;

    treeNodes[x].rightChild = y;
    treeNodes[y].parent = x;

    // Update parent of x to z
    treeNodes[x].parent = z;
    if (z) {
        if (treeNodes[z].leftChild == y)
            treeNodes[z].leftChild = x;
        else
            treeNodes[z].rightChild = x;
    }

    // Update the subtree information
    updateNode(y);
    updateNode(x);
}

// Function to splay node x to the goal's position
inline void splay(int x, int goal) {
    while (treeNodes[x].parent != goal) {
        int y = treeNodes[x].parent;
        int z = treeNodes[y].parent;

        if (z != goal) {
            if ((treeNodes[z].leftChild == y) == (treeNodes[y].leftChild == x)) {
                // Zig-Zig case
                if (treeNodes[y].leftChild == x)
                    rotateRight(y);
                else
                    rotateLeft(y);
            }
            else {
                // Zig-Zag case
                if (treeNodes[y].leftChild == x)
                    rotateRight(x);
                else
                    rotateLeft(x);
            }
        }

        // Perform single rotation
        if (treeNodes[y].leftChild == x)
            rotateRight(x);
        else
            rotateLeft(x);
    }

    if (goal == 0)
        root = x; // Update root if splayed to root
}

// Function to insert a new value into the splay tree
inline void insertValue(ll value) {
    int x = root;
    int parent = 0;

    // Traverse the tree to find the insertion point
    while (x != 0) {
        parent = x;
        if (value < treeNodes[x].value)
            x = treeNodes[x].leftChild;
        else
            x = treeNodes[x].rightChild;
    }

    // Create new node
    int newNode = ++totalNodes;
    treeNodes[newNode].value = value;
    treeNodes[newNode].sum[0] = value;
    treeNodes[newNode].size = 1;
    treeNodes[newNode].parent = parent;

    // Attach the new node to its parent
    if (parent != 0) {
        if (value < treeNodes[parent].value)
            treeNodes[parent].leftChild = newNode;
        else
            treeNodes[parent].rightChild = newNode;
    }

    // Splay the new node to the root
    splay(newNode, 0);
}

// Function to find a node with a given value and splay it to the root
inline int findValue(ll value) {
    int x = root;
    while (x != 0) {
        if (value == treeNodes[x].value)
            break;
        if (value < treeNodes[x].value)
            x = treeNodes[x].leftChild;
        else
            x = treeNodes[x].rightChild;
    }

    if (x != 0)
        splay(x, 0); // Splay found node to root

    return x;
}

// Function to find the node to attach during erase
inline int findAffix(ll value, bool isSuccessor) {
    int x = root;
    while (x != 0) {
        if ((isSuccessor && treeNodes[x].value > value) || 
            (!isSuccessor && treeNodes[x].value < value)) {
            if (isSuccessor)
                x = treeNodes[x].leftChild;
            else
                x = treeNodes[x].rightChild;
        }
        else {
            if (isSuccessor)
                x = treeNodes[x].leftChild;
            else
                x = treeNodes[x].rightChild;
        }
    }

    return x;
}

// Function to erase a value from the splay tree
inline void eraseValue(ll value) {
    int node = findValue(value);
    if (node == 0 || treeNodes[node].value != value)
        return; // Value not found

    // Splay the node to be deleted to the root
    splay(node, 0);

    // If left child exists, find the maximum in left subtree
    if (treeNodes[node].leftChild != 0) {
        int predecessor = treeNodes[node].leftChild;
        while (treeNodes[predecessor].rightChild != 0)
            predecessor = treeNodes[predecessor].rightChild;

        splay(predecessor, node);

        // Attach right subtree
        treeNodes[predecessor].rightChild = treeNodes[node].rightChild;
        if (treeNodes[node].rightChild != 0)
            treeNodes[treeNodes[node].rightChild].parent = predecessor;

        // Update the parent of predecessor
        treeNodes[treeNodes[node].rightChild].parent = predecessor;

        // Update node's parent
        treeNodes[predecessor].parent = 0;
        root = predecessor;

        // Update aggregate information
        updateNode(predecessor);
    }
    else {
        // If no left child, replace root with right child
        root = treeNodes[node].rightChild;
        if (root != 0)
            treeNodes[root].parent = 0;
    }

    // Node is now deleted, no need to keep it
}

// Function to compute the sum of medians
inline ll computeSumOfMedians() {
    // The sum of medians is stored in sum[4] of the root node
    // Adjustments are made based on the initial code's handling

    if (root == 0)
        return 0;

    ll medianSum = treeNodes[root].sum[4];

    // Handle special cases if necessary
    if (medianSum < 0)
        medianSum += INF;
    else if (medianSum >= INF)
        medianSum -= INF;

    return medianSum;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    // Initialize the tree with sentinel nodes
    insertValue(-INF);
    insertValue(INF);

    int n;
    cin >> n;
    while(n--){
        string operation;
        cin >> operation;
        if(operation == "add"){
            ll x;
            cin >> x;
            insertValue(x);
        }
        else if(operation == "del"){
            ll x;
            cin >> x;
            eraseValue(x); 
        }
        else if(operation == "sum"){
            ll result = computeSumOfMedians();
            cout << result << "\n";
        }
    }

    return 0;
}

