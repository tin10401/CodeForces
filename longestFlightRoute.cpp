#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll unsigned long long
#define int long long
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
vector<bool> vis;

void dfs(int node, vector<vector<int>>& graph)
{
    vis[node] = true;
    for(auto& nei : graph[node])
    {
        if(!vis[nei]) dfs(nei, graph);
    }
}

void solve()
{
    int n, m;
    cin >> n >> m;
    vector<vector<int>> graph(n + 1);
    vector<int> parent(n + 1, -1), degree(n + 1);
    vis.resize(n + 1, false);
    for(int i = 0; i < m; i++)
    {
        int a, b;
        cin >> a >> b;
        graph[a].push_back(b);
        degree[b]++;
    }
    dfs(1, graph);
    if(!vis[n])
    {
        cout << "IMPOSSIBLE" << endl;
        return;
    }
    queue<int> q;
    for(int i = 1; i <= n; i++)
    {
        if(degree[i] == 0) q.push(i);
    }
    vector<int> dp(n + 1, -1);
    dp[1] = 0;
    while(!q.empty())
    {
        int node = q.front();
        q.pop();
        for(auto& nei : graph[node])
        {
            if(dp[node] != -1 && dp[nei] < dp[node] + 1) dp[nei] = dp[node] + 1, parent[nei] = node;
            if(--degree[nei] == 0)
            {
                q.push(nei);
            }
        }
    }
    vector<int> path = {n};
    while(path.back() != 1) path.push_back(parent[path.back()]);
    reverse(all(path));
    cout << path.size() << endl;
    for(auto& it : path) cout << it << " ";
    cout << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

