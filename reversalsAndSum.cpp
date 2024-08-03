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
    Node* left, *right;
    int pri = rand(), val = 0, sum = 0, size = 1, reverse = 0;
};

static Node nodes[(int)2e5 + 1];

class Treap
{
    public:
    Node* root;
    int count = 0;
    Node* newNode(int v)
    {
        nodes[count] = Node();
        Node* n = &nodes[count++];
        n->val = n->sum = v;
        return n;
    }

    Treap(vi& arr)
    {
        root = 0;
        for(auto& v : arr) merge(root, root, newNode(v));
    }

    pii data(Node* treap)
    {
        if(!treap) return {0, 0};
        return {treap->size, treap->sum};
    }

    void update(Node* treap)
    {
        if(!treap) return;
        auto [lSize, lSum ] = data(treap->left);
        auto [rSize, rSum] = data(treap->right);
        treap->size = lSize + rSize + 1;
        treap->sum = lSum + rSum + treap->val;
    }

    void push(Node* treap)
    {
        if(!treap || !treap->reverse) return;
        treap->reverse = 0;
        swap(treap->left, treap->right);
        if(treap->left) treap->left->reverse ^= 1;
        if(treap->right) treap->right->reverse ^= 1;
    }

    void split(Node* treap, Node*& left, Node*& right, int k)
    {
        if(!treap)
        {
            left = right = nullptr;
            return;
        }
        push(treap);
        int sz = data(treap->left).f;
        if(sz >= k)
        {
            split(treap->left, left, treap->left, k);
            right = treap;
        }
        else
        {
            split(treap->right, treap->right, right, k - sz - 1);
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

    int cal(int a, int b, bool rev)
    {
        Node* A, *B, *C;
        split(root, A, B, a - 1);
        split(B, B, C, b - a + 1);
        if(rev) B->reverse ^= 1;
        int res = B->sum;
        merge(root, A, B);
        merge(root, root, C);
        return res;
    }

    
};

void solve()
{
    int n, m; cin >> n >> m;
    vi arr(n);
    for(auto& it : arr) cin >> it;
    Treap root(arr);
    while(m--)
    {
        int type, a, b; cin >> type >> a >> b;
        if(type == 1) root.cal(a, b, true);
        else cout << root.cal(a, b, false) << endl;
    }
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

