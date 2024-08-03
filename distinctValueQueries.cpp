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

class FenwickTree
{
    public:
    int n;
    vi root;
    FenwickTree(int n)
    {
        this->n = n;
        root.resize(n + 1);
    }

    void update(int id, int val)
    {
        while(id <= n)
        {
            root[id] += val;
            id += (id & -id);
        }
    }

    int get(int id)
    {
        int res = 0;
        while(id)
        {
            res += root[id];
            id -= (id & -id);
        }
        return res;
    }
};

void solve()
{
    int n, q;
    cin >> n >> q;
    vi arr(n);
    for(auto& it : arr) cin >> it;
    vector<vpii> indices(n);
    for(int i = 0; i < q; i++)
    {
        int a, b;
        cin >> a >> b;
        a--, b--;
        indices[a].pb({b, i});
    }
    vi res(q);
    FenwickTree root(n);
    map<int, int> mp;
    for(int i = n - 1; i >= 0; i--)
    {
        int val = arr[i];
        if(mp.count(val)) root.update(mp[val] + 1, -1);
        mp[val] = i;
        root.update(i + 1, 1);
        for(auto& [end, index] : indices[i])
        {
            res[index] = root.get(end + 1);
        }
    }
    for(auto& it : res) cout << it << endl;

        
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

