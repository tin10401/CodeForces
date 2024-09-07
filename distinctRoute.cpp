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
const static int INF = 1e18;
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
 
int modExpo(ll base, ll exp)
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
 
vi path, vis;
vector<vi> graph, adj;
vector<vpii> gr;
void dfs(int node, int n)
{
   path.push_back(node);
   if(node == n) return;
   for(auto& [nei, id] : gr[node])
   {
       if(adj[node][nei] == 0 && !vis[id])
       {
           vis[id] = true;
           dfs(nei, n);
           return;
       }
   }
}
 
 
void solve()
{
    int n, m;
    cin >> n >> m;
    vis.resize(m + 1);
    graph.resize(n + 1), adj.resize(n + 1, vi(n + 1));
    gr.resize(n + 1);
    vi parent(n + 1);
    for(int i = 0; i < m; i++)
    {
        int a, b;
        cin >> a >> b;
        gr[a].pb({b, i});
        graph[a].pb(b), graph[b].pb(a);
        adj[a][b]++;
    }
    auto reachable = [&]() -> int
    {
        fill(all(parent), 0);
        queue<pii> q;
        q.push({1, INF});
        while(!q.empty())
        {
            auto [node, aug] = q.front(); q.pop();
            if(node == n) return aug;
            for(auto& nei : graph[node]) if(adj[node][nei] && parent[nei] == 0) parent[nei] = node, q.push({nei, min(aug, adj[node][nei])});
        }
        return 0;
    };
    int a = 0, flow = 0;
    while(a = reachable())
    {
        flow += a;
        for(int v = n; v != 1; v = parent[v])
        {
            int u = parent[v];
            adj[v][u] += a;
            adj[u][v] -= a;
        }
    }
    cout << flow << endl;
    fill(all(vis), false);
    while(flow--)
    {
        path = {};
        dfs(1, n);
        cout << path.size() << endl;
        for(auto& it : path) cout << it << " ";
        cout << endl;
    }
}
 
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

