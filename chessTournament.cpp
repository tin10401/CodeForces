#pragma GCC target("popcnt")
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ub upper_bound
#define lb lower_bound
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

void solve()
{
    int n;
    cin >> n;
    set<pii> s;
    int total = 0;
    for(int i = 1; i <= n; i++)
    {
        int val; cin >> val;
        total += val;
        s.insert({val, i});
    }
    vpii res;
    while(s.size() >= 2)
    {
        auto [x1, p1] = *s.begin(); s.erase({x1, p1});
        vpii remain;
        while(x1 && !s.empty())
        {
            auto [x2, p2] = *s.rbegin(); s.erase({x2, p2});
            res.pb({p1, p2});
            x1--, x2--;
            remain.pb({x2, p2});
        }
        for(auto& [x, p] : remain) if(x) s.insert({x, p});
    }
    if(res.size() * 2 == total)
    {
        cout << res.size() << endl;
        for(auto& [x, y] : res) cout << x << " " << y << endl;
    }
    else
    {
        cout << "IMPOSSIBLE" << endl;
    }
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

