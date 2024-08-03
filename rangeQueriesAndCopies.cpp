#include <bits/stdc++.h>
//#include <ext/pb_ds/assoc_container.hpp>
//using namespace __gnu_pbds;
using namespace std;
//typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
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
const int MXN = 4e5 + 5, MXK = 22;
pii child[MXN * MXK];
int tree[MXN * MXK], root[MXN];
int ptr = 0, sz = 1, n, q;

void update(int curr, int prev, int left, int right, int index, int val)
{
    if(left == right)
    {
        tree[curr] = val;
        return;
    }
    int middle = left + (right - left) / 2;
    if(index <= middle)
    {
        child[curr].f = ++ptr;
        child[curr].s = child[prev].s;
        update(child[curr].f, child[prev].f, left, middle, index, val);
    }
    else
    {
        child[curr].s = ++ptr;
        child[curr].f = child[prev].f;
        update(child[curr].s, child[prev].s, middle + 1, right, index, val);
    }
    tree[curr] = tree[child[curr].f] + tree[child[curr].s];
}

int get(int curr, int left, int right, int start, int end)
{
    if(left >= start && right <= end) return tree[curr];
    if(left > end || start > right) return 0;
    int middle = left + (right - left) / 2;
    return get(child[curr].f, left, middle, start, end) + get(child[curr].s, middle + 1, right, start, end);
}

void solve()
{
    cin >> n >> q;
    root[1] = 0;
    for(int i = 0; i < n; i++)
    {
        int val; cin >> val;
        int prev = root[1];
        root[1] = ++ptr;
        update(root[1], prev, 0, n - 1, i, val);
    }

    while(q--)
    {
        int type;
        cin >> type;
        if(type == 1)
        {
            int k, a, x;
            cin >> k >> a >> x;
            int newRoot = ++ptr;
            a--;
            update(newRoot, root[k], 0, n - 1, a, x);
            root[k] = newRoot;
        }
        else if(type == 2)
        {
            int k, a, b;
            cin >> k >> a >> b;
            a--, b--;
            cout << get(root[k], 0, n - 1, a, b) << endl;
        }
        else
        {
            int k;
            cin >> k;
            root[++sz] = root[k];
        }
    }
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

