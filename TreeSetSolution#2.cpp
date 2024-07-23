#include <bits/stdc++.h>
using namespace std;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll long long
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

int res = 0;
vector<bool> vis;
void dfs(int node, int par, vector<vector<int>>& graph)
{
    for(auto& nei : graph[node])
    {
        if(nei != par)
        {
            dfs(nei, node, graph);
            if(!vis[nei] && !vis[node])
            {
                res++;
                vis[nei] = true, vis[node] = true;
            }
        }
    }
}
void solve()
{
    int n;
    cin >> n;
    vector<vector<int>> graph(n);
    vis.resize(n);
    for(int i = 0; i < n - 1; i++)
    {
        int a, b;
        cin >> a >> b;
        a--, b--;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }
    dfs(0, -1, graph);
    cout << res << endl;
}

int main()
{
    solve();
}
