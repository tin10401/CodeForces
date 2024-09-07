//https://cses.fi/problemset/result/9955724/
#include <bits/stdc++.h>
using namespace std;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll long long
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

vector<int> depth;
vector<vector<int>> dp, graph;

void dfs(int node, int d)
{
    depth[node] = d;
    for(auto& nei : graph[node])
    {
        dfs(nei, d + 1);
    }
}

int LCA(int a, int b)
{
    if(depth[a] > depth[b]) swap(a, b);
    int d = depth[b] - depth[a];
    for(int i = 0; i < 30; i++)
    {
        if((d >> i) & 1)
        {
            b = dp[b][i];
        }
    }
    if(a == b) return a;

    for(int i = 29; i >= 0; i--)
    {
        if(dp[a][i] != dp[b][i])
        {
            a = dp[a][i];
            b = dp[b][i];
        }
    }
    return dp[a][0];
}
void solve()
{
    int n, q;
    cin >> n >> q;
    depth.resize(n + 1), graph.resize(n + 1), dp.resize(n + 1, vector<int>(30));
    for(int i = 2; i <= n; i++)
    {
        int num;
        cin >> num;
        graph[num].push_back(i);
        dp[i][0] = num;
    }
    for(int j = 1; j < 30; j++)
    {
        for(int i = 1; i <= n; i++)
        {
            dp[i][j] = dp[dp[i][j - 1]][j - 1];
        }
    }
    dfs(1, 0);
    while(q--)
    {
        int a, b;
        cin >> a >> b;
        cout << LCA(a, b) << endl;
    }

}

int main()
{
    solve();
}
