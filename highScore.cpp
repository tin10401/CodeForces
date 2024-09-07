//https://cses.fi/problemset/result/9976133/
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll unsigned long long
#define int long long
const int INF = LLONG_MAX;
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

void dfs(int node, vector<vector<int>>& graph, vector<bool> & vis)
{
    vis[node] = true;
    for(auto& nei : graph[node])
    {
        if(!vis[nei]) dfs(nei, graph, vis);
    }
}

void solve()
{
    int n, m;
    cin >> n >> m;
    vector<vector<int>> graph(n + 1), revGraph(n + 1);
    vector<array<int, 3>> edge;
    for(int i = 0; i < m; i++)
    {
        int a, b, c;
        cin >> a >> b >> c;
        edge.push_back({a, b, c});
        graph[a].push_back(b);
        revGraph[b].push_back(a);
    }
    vector<bool> vis(n + 1), visR(n + 1);
    dfs(1, graph, vis), dfs(n, revGraph, visR);
    vector<int> dp(n + 1, -INF);
    bool improvement = true;
    dp[1] = 0;
    for(int i = 1; i <= n && improvement; i++)
    {
        improvement = false;
        for(auto& [a, b, c] : edge)
        {
            if(dp[a] != -INF && dp[a] + c > dp[b])
            {
                dp[b] = dp[a] + c;
                improvement = true;
                if(i == n && vis[b] && visR[b])
                {
                    cout << -1 << endl;
                    return;
                }
            }
        }
    }
    cout << dp[n] << endl;
    
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

