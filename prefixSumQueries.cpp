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

class SegmentTree
{
    public:
    int n;
    vpii root;
    SegmentTree(vi &arr)
    {
        n = arr.size();
        root.resize(n * 4);
        build(0, 0, n - 1, arr);
    }

    pii merge(pii left, pii right)
    {
        pii res;
        res.f = left.f + right.f;
        res.s = max(left.s, left.f + right.s);
        return res;
    }

    void build(int i, int left, int right, vi& arr)
    {
        if(left == right)
        {
            root[i].f = arr[left];
            root[i].s = max(0LL, arr[left]);
            return;
        }
        int middle = left + (right - left) / 2;
        build(i * 2 + 1, left, middle, arr);
        build(i * 2 + 2, middle + 1, right, arr);
        root[i] = merge(root[i * 2 + 1], root[i * 2 + 2]);
    }

    
    void update(int index, int val)
    {
        update(0, 0, n - 1, index, val);
    }

    void update(int i, int left, int right, int index, int val)
    {
        if(left == right)
        {
            root[i].f = val;
            root[i].s = max(0LL, val);
            return;
        }
        int middle = left + (right - left) / 2;
        if(index <= middle) update(i * 2 + 1, left, middle, index, val);
        else update(i * 2 + 2, middle + 1, right, index, val);
        root[i] = merge(root[i * 2 + 1], root[i * 2 + 2]);
    }

    int get(int start, int end)
    {
        return get(0, 0, n - 1, start, end).s;
    }

    pii get(int i, int left, int right, int start, int end)
    {
        if(left >= start && right <= end) return root[i];
        if(start > right || left > end) return {0, 0};
        int middle = left + (right - left) / 2;
        return merge(get(i * 2 + 1, left, middle, start, end), get(i * 2 + 2, middle + 1, right, start, end));
    }
};

void solve()
{
    int n, q;
    cin >> n >> q;
    vi arr(n);
    for(auto& it : arr) cin >> it;
    SegmentTree root(arr);
    while(q--)
    {
        int type, a, b;
        cin >> type >> a >> b;
        if(type == 1)
        {
            root.update(--a, b);
        }
        else cout << root.get(--a, --b) << endl;
    }
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

