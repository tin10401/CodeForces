// https://cses.fi/problemset/task/2079/
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll long long
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

int n;
vector<vector<int>> graph;
vector<int> subTree;
vector<int> parent;
int dfs(int node, int par)
{
    parent[node] = par;
    int total = 1;
    for(auto& nei : graph[node])
    {
        if(nei != par) total += dfs(nei, node);
    }
    return subTree[node] = total;
}
int dfs2(int node, int par)
{
    int next = 0, mx = 0;
    for(int nei : graph[node])
    {
        if(nei != par)
        {
           if(subTree[nei] > mx)
           {
               mx = subTree[nei];
               next = nei;
           }
        }
    }
    if(mx <= n / 2) return node;
    return dfs2(next, node);
}

void solve()
{
    cin >> n;
    graph.resize(n + 1), subTree.resize(n + 1), parent.resize(n + 1);
    for(int i = 0; i < n - 1; i++)
    {
        int a, b;
        cin >> a >> b;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }
    dfs(1, 0);
    cout << dfs2(1, 0) << endl;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

