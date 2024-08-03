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
vi insertTime, lowestTime, vis, res;
vector<vi> graph;
int t = 0;
void dfs(int node, int par)
{
    if(vis[node]) return;
    vis[node] = true;
    insertTime[node] = lowestTime[node] = t++;
    int child = 0;
    for(auto& nei : graph[node])
    {
        if(nei == par) continue;
        if(!vis[nei])
        {
            dfs(nei, node);
            lowestTime[node] = min(lowestTime[node], lowestTime[nei]);
            child++;
            if(lowestTime[nei] >= insertTime[node] && par != -1) res[node] = true;
        }
        else lowestTime[node] = min(lowestTime[node], insertTime[nei]);
    }
    if(par == -1 && child > 1) res[node] = true;
}
void solve()
{
   int n, m; cin >> n >> m; 
   graph.resize(n + 1), vis.resize(n + 1), insertTime.resize(n + 1), lowestTime.resize(n + 1), res.resize(n + 1);
   for(int i = 0; i < m; i++)
    {
        int a, b; cin >> a >> b;
        graph[a].pb(b), graph[b].pb(a);
    }
    dfs(1, -1);
    cout << count(all(res), true) << endl;
    for(int i = 1; i <= n; i++)
    {
        if(res[i]) cout << i << " ";
    }
    cout << endl;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

