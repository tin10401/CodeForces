// https://cses.fi/problemset/task/1136/
#include <bits/stdc++.h>
using namespace std;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll long long
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

vector<vector<int>> dp, graph;
vector<int> depth;

void dfs(int node, int par, int d)
{
    dp[node][0] = par;
    depth[node] = d;
    for(auto& nei : graph[node])
    {
        if(nei != par)
        {
            dfs(nei, node, d + 1);
        }
    }
}

int LCA(int a, int b)
{
    if(depth[a] > depth[b]) swap(a, b);
    int d = depth[b] - depth[a];
    for(int i = 0; i < 21; i++)
    {
        if((d >> i) & 1) b = dp[b][i];
    }
    if(a == b) return a;
    for(int i = 20; i >= 0; i--)
    {
        if(dp[a][i] != dp[b][i]) a = dp[a][i], b = dp[b][i];
    }
    return dp[a][0];
}

vector<int> res;
int dfs2(int node, int par)
{
    int& total = res[node];
    for(auto& nei : graph[node])
    {
        if(nei != par)
        {
           total += dfs2(nei, node);
        }
    }
    return total;
}


void solve()
{
    int n, q;
    cin >> n >> q;
    graph.resize(n + 1), dp.resize(n + 1, vector<int>(21)), depth.resize(n + 1), res.resize(n + 1);
    for(int i = 0; i < n - 1; i++)
    {
        int a, b;
        cin >> a >> b;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }
    dfs(1, 0, 0);
    for(int j = 1; j < 21; j++)
    {
        for(int i = 1; i <= n; i++)
        {
            dp[i][j] = dp[dp[i][j - 1]][j - 1];
        }
    }
    while(q--)
    {
        int a, b;
        cin >> a >> b;
        res[a]++, res[b]++;
        int lca = LCA(a, b);
        res[lca]--;
        if(lca == a || lca == b) res[dp[lca][0]]--;
    }
    dfs2(1, 0);
    for(int i = 1; i <= n; i++) cout << res[i] << " ";
    cout << endl;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

