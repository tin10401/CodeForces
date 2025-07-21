// Author : Tin Le
#pragma GCC optimize("Ofast")
#pragma GCC optimize ("unroll-loops")
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
#define db double
#define ll unsigned long long
#define int long long
#define vi vector<int>
#define pii pair<int, int>
#define vpii vector<pair<int, int>>
#define vvi vector<vi>
#define vd vector<db>
#define ar(x) array<int, x>
#define mset(m, v) memset(m, v, sizeof(m))
#define pb push_back
#define f first
#define s second
#define rsz resize
#define sum(x) accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define rev(x) reverse(all(x))
#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

const static int INF = 1LL << 61;
const static int MX = 2e5 + 5;
const static int MOD = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
constexpr int pct(int x) { return __builtin_popcount(x); }
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
constexpr int modExpo(int base, int exp, int mod) { int res = 1; while(exp) {
    if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>=
        1; } return res; }
int n, m, timer, tin[MX], par[MX], sdom[MX], idom[MX], inv[MX], label[MX], dsu[MX];
vi G[MX], GR[MX], bucket[MX], DT[MX];
void dfs(int u = 1)
{
    tin[u] = ++timer;
    inv[timer] = u;
    label[timer] = dsu[timer] = sdom[timer] = timer;
    for(auto& v : G[u])
    {
        if(!tin[v])
        {
            dfs(v);
            par[tin[v]] = tin[u];
        }
        GR[tin[v]].pb(tin[u]);
    }
}

int find(int u, bool x = false)
{
    if(u == dsu[u]) return x ? -1 : u;
    int v = find(dsu[u], true);
    if(v < 0) return u;
    if(sdom[label[dsu[u]]] < sdom[label[u]]) label[u] = label[dsu[u]];
    dsu[u] = v;
    return x ? v : label[u];
}

void build()
{
    dfs();
    for(int u = n; u; u--)
    {
        for(auto& v : GR[u])
        {
            sdom[u] = min(sdom[u], sdom[find(v)]);
        }
        if(u > 1) bucket[sdom[u]].pb(u);
        for(auto& v : bucket[u])
        {
            idom[v] = (sdom[v] == sdom[find(v)] ? sdom[v] : find(v));
        }
        if(u > 1) dsu[u] = par[u];
    }
    for(int u = 2; u <= n; u++)
    {
        if(idom[u] != sdom[u]) idom[u] = idom[idom[u]];
        DT[inv[u]].pb(inv[idom[u]]);
        DT[inv[idom[u]]].pb(inv[u]);
    }
}

vi ans;
bool dfs_tree(int u = 1, int p = -1)
{
    bool good = u == n;
    for(auto& v : DT[u])
    {
        if(v != p) good |= dfs_tree(v, u);
    }
    if(good) ans.pb(u);
    return good;
}
void solve()
{
    cin >> n >> m;
    for(int i = 0; i < m; i++)
    {
        int a, b; cin >> a >> b;
        G[a].pb(b);
    }
    build(), dfs_tree();
    int k = ans.size();
    cout << k << endl;
    srt(ans);
    for(auto& it : ans) cout << it << " ";
    cout << endl;
}

signed main()
{
    IOS;
    int t = 1;
    // cin >> t;
    while(t--) solve();
    return 0;
}

