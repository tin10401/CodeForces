//https://cses.fi/problemset/result/9960810/
#include <bits/stdc++.h>
using namespace std;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll long long
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};


vector<vector<int>> graph;
vector<int> res;
void dfs(int node, int par, int d)
{
    res[node] = max(res[node], d);
    for(auto& nei : graph[node])
    {
        if(nei != par)
        {
            dfs(nei, node, d + 1);
        }
    }
}
void dfs(int node, int par, int d, pair<int, int>& a)
{
    if(d > a.first) a = {d, node};
    for(auto& nei : graph[node])
    {
        if(nei != par)
        {
            dfs(nei, node, d + 1, a);
        }
    }
}
void solve()
{
    int n;
    cin >> n;
    graph.resize(n + 1), res.resize(n + 1);
    for(int i = 0; i < n - 1; i++)
    {
        int a, b;
        cin >> a >> b;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }

    pair<int, int> s1 = {INT_MIN, -1}, s2 = {INT_MIN, -1};
    dfs(1, -1, 0, s1);
    dfs(s1.second, -1, 0, s2);
    dfs(s1.second, -1, 0);
    dfs(s2.second, -1, 0);
    for(int i = 1; i <= n; i++) cout << res[i] << " ";
    cout << endl;
}

int main()
{
    solve();
}
