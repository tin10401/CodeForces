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
#define arr(x) array<int, x>
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
    vector<arr(4)> root;
    SegmentTree(int n)
    {
        this->n = n;
        root.resize(n * 4);
    }

    void update(int index, int val)
    {
        update(0, 0, n - 1, index, val);
    }

    void update(int i, int left, int right, int index, int val)
    {
        if(left == right)
        {
            for(int k = 0; k < 4; k++) root[i][k] = val;
            return;
        }

        int middle = left + (right - left) / 2;
        if(index <= middle) update(i * 2 + 1, left, middle, index, val);
        else update(i * 2 + 2, middle + 1, right, index, val);
        root[i] = merge(root[i * 2 + 1], root[i * 2 + 2]);
    }

    arr(4) merge(arr(4) left, arr(4) right)
    {
        arr(4) res;
        res[0] = max({left[0], right[0], left[2] + right[1], 0LL});
        res[1] = max({left[1], left[3] + right[1], 0LL});
        res[2] = max({right[2], left[2] + right[3], 0LL});
        res[3] = left[3] + right[3];
        return res;
    }
            
    int get()
    {
        return max(0LL, root[0][0]);
    }
};

void solve()
{
    int n, q;
    cin >> n >> q;
    SegmentTree root(n);
    for(int i = 0; i < n; i++)
    {
        int val;
        cin >> val;
        root.update(i, val);
    }
    while(q--)
    {
        int i, x;
        cin >> i >> x;
        root.update(--i, x);
        cout << root.get() << endl;
    }

}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

