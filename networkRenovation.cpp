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
vi res;
vector<vi> graph;
void dfs(int node, int par)
{
    for(auto& nei : graph[node])
    {
        if(nei != par) dfs(nei, node);
    }
    if(graph[node].size() == 1) res.pb(node);
}
void solve()
{
    int n; cin >> n;
    graph.resize(n + 1);
    for(int i = 0; i < n - 1; i++)
    {
        int a, b; cin >> a >> b;
        graph[a].pb(b);
        graph[b].pb(a);
    }
    dfs(1, -1);
    int k = res.size();
    if(k & 1) res.pb(res[0]), k++;
    cout << k / 2 << endl;
    for(int i = 0; i < k / 2; i++) cout << res[i] << " " << res[i + k / 2] << endl;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

