#pragma GCC target("popcnt")
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ll unsigned long long
#define int long long
#define pb push_back
#define vi vector<int>
#define pii pair<int, int>
#define vpii vector<pair<int, int>>
#define f first
#define s second
#define ar(x) array<int, x>
const static int INF = 1LL << 61;
const static int MOD = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

int modExpo(int base, int exp, int mod)
{
    int res = 1;
    while(exp)
    {
        if(exp & 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return res;
}
struct Node
{
    char ch;
    Node* left, *right;
    int pri, size, reverse;
    Node(char ch) : reverse(false), ch(ch), size(1), pri(rand()), left(nullptr), right(nullptr) {}
};

void push(Node* treap)
{
    if(!treap->reverse) return;
    treap->reverse = 0;
    swap(treap->left, treap->right);
    if(treap->left) treap->left->reverse ^= 1;
    if(treap->right) treap->right->reverse ^= 1;
}

int size(Node* treap)
{
    if(!treap) return 0;
    return treap->size;
}
void update(Node* treap)
{
    if(!treap) return;
    treap->size = size(treap->left) + size(treap->right) + 1;
}

void split(Node* treap, Node*& left, Node*& right, int k)
{
    if(!treap)
    {
        left = right = nullptr;
        return;
    }

    push(treap);
    if(size(treap->left) >= k)
    {
        split(treap->left, left, treap->left, k);
        right = treap;
    }
    else
    {
        split(treap->right, treap->right, right, k - size(treap->left) - 1);
        left = treap;
    }
    update(treap);
}

void merge(Node*& treap, Node* left, Node* right)
{
    if(!left || !right) 
    {
        treap = left ? left : right;
        return;
    }
    push(left), push(right);
    if(left->pri < right->pri)
    {
        merge(left->right, left->right, right);
        treap = left;
    }
    else
    {
        merge(right->left, left, right->left);
        treap = right;
    }
    update(treap);
}

void print(Node* treap)
{
    if(!treap) return;
    push(treap);
    print(treap->left);
    cout << treap->ch;
    print(treap->right);
}

void solve()
{
    int n, m; cin >> n >> m;
    string s; cin >> s;
    Node* treap = 0;
    for(auto& ch : s) merge(treap, treap, new Node(ch));
    while(m--)
    {
        int a, b; cin >> a >> b;
        Node* A, *B, *C;
        split(treap, A, B, a - 1);
        split(B, B, C, b - a + 1);
        B->reverse ^= 1;
        merge(treap, A, B);
        merge(treap, treap, C);
    }
    print(treap);
    cout << endl;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

