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

void dfs(int node, vector<vector<int>> & graph, stack<int>& s, vector<bool>& vis, bool push_in)
{
    if(vis[node]) return;
    vis[node] = true;
    for(auto& nei : graph[node])
    {
        dfs(nei, graph, s, vis, push_in);
    }
    if(push_in) s.push(node);
}

void solve()
{
    int n, m;
    cin >> n >> m;
    vector<vector<int>> graph(n + 1), revGraph(n + 1);
    for(int i = 0; i < m; i++)
    {
        int a, b;
        cin >> a >> b;
        graph[a].push_back(b);
        revGraph[b].push_back(a);
    }
    vector<bool> vis(n + 1);
    stack<int> s;
    for(int i = 1; i <= n; i++) dfs(i, graph, s, vis, true);
    int last = -1;
    vector<int> arr;
    fill(all(vis), false);
    while(!s.empty())
    {
        int node = s.top(); s.pop();
        if(vis[node]) continue;
        if(last != -1)
        {
            arr = {node, last};
            break;
        }
        last = node;
        dfs(node, revGraph, s, vis, false);
    }
    if(arr.empty()) cout << yes;
    else
    {
        cout << no;
        for(auto& it : arr) cout << it << " ";
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

