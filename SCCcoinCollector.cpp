// https://cses.fi/problemset/task/1686/
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll unsigned long long
#define int long long
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
 
class SCC
{
    public:
    int n, curr = 0;
    vector<vector<int>> graph, revGraph;
    vector<int> vis, comp;
    stack<int> s;
 
    SCC() {}
 
    void init(int n)
    {
        this->n = n;
        graph.resize(n + 1), revGraph.resize(n + 1), vis.resize(n + 1), comp.resize(n + 1, -1);
    }
 
    void add(int a, int b) { graph[a].push_back(b), revGraph[b].push_back(a);}
 
 
    void dfs(int node)
    {
        if(vis[node]) return;
        vis[node] = true;
        for(auto& nei : graph[node]) dfs(nei);
        s.push(node);
    }
 
    void dfs2(int node)
    {
        if(comp[node] != -1) return;
        comp[node] = curr;
        for(auto& nei : revGraph[node]) dfs2(nei);
    }
 
    void generate()
    {
        for(int i = 1; i <= n; i++) dfs(i);
        while(!s.empty())
        {
            int node = s.top(); s.pop();
            if(comp[node] != -1) continue;
            dfs2(node);
            curr++;
        }
    }
};
 
vector<vector<int>> graph;
vector<int> coins, group, dp;
SCC root;
int dfs(int node)
{
    int& res = dp[node];
    if(res != -1) return res;
    res = 0;
    for(auto& nei : graph[node])
    {
        res = max(res, dfs(nei));
    }
    res += group[node];
    return res;
}
 
void solve()
{
    int n, m;
    cin >> n >> m;
    coins.resize(n);
    for(auto& it : coins) cin >> it;
    root.init(n);
 
    for(int i = 0; i < m; i++)
    {
        int a, b;
        cin >> a >> b;
        root.add(a, b);
    }
 
    root.generate();
    graph.resize(root.curr), dp.resize(root.curr, -1), group.resize(root.curr);
    for(int i = 0; i < n; i++) group[root.comp[i + 1]] += coins[i];
    
    for(int i = 1; i <= n; i++)
    {
        for(auto& nei : root.graph[i])
        {
            if(root.comp[nei] == root.comp[i]) continue;
            graph[root.comp[nei]].push_back(root.comp[i]);
        }
    }
    int res = 0;
    for(int i = 0; i < root.curr; i++) res = max(res, dfs(i));
    cout << res << endl;
 
}
 
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

