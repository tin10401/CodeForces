#include <bits/stdc++.h>
using namespace std;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll long long
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

void dfs(int node, int par, vector<vector<int>>& graph, vector<vector<int>>& dp)
{
    for(auto& nei : graph[node])
    {
        if(nei != par)
        {
            dfs(nei, node, graph, dp);
            dp[node][0] += max(dp[nei][0], dp[nei][1]);
        }
    }
    for(auto& nei : graph[node])
    {
        if(nei != par)
        {
            dp[node][1] = max(dp[node][1], dp[nei][0] + 1 + dp[node][0] - max(dp[nei][0], dp[nei][1])); // if you choose to include this edge, you will subtract the max(dp[nei][1], dp[nei][0]) because it over calculate the result of the dp[node][0] above;
        }
    }
}
void solve()
{
    int n;
    cin >> n;
    vector<vector<int>> graph(n);
    vector<vector<int>> dp(n, vector<int>(2, 0));
    for(int i = 0; i < n - 1; i++)
    {
        int a, b;
        cin >> a >> b;
        a--, b--;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }
    dfs(0, -1, graph, dp);
    cout << max(dp[0][1], dp[0][0]) << endl;
}

int main()
{
    solve();
}
