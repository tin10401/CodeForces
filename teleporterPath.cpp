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

vector<vector<int>> graph;
vector<int> path;
            
void dfs(int node)
{
    while(!graph[node].empty())
    {
        int nei = graph[node].back();
        graph[node].pop_back();
        dfs(nei);
    }
    path.push_back(node);
}

void dfs(int node, vector<bool>& vis)
{
    if(vis[node]) return;
    vis[node] = true;
    for(auto& nei : graph[node]) dfs(nei, vis);
}

void solve()
{
    int n, m;
    cin >> n >> m;
    graph.resize(n + 1);
    vector<int> in(n + 1);
    for(int i = 0; i < m; i++)
    {
        int a, b;
        cin >> a >> b;
        graph[a].push_back(b);
        in[b]++;
    }
    vector<bool> vis(n + 1);
    dfs(1, vis);
    bool exist = vis[n];
    exist &= (in[1] + 1 == (int)graph[1].size());
    exist &= (in[n] - 1 == (int)graph[n].size());

    for(int i = 2; i < n; i++)
    {
        exist &= (in[i] == (int)graph[i].size());
        if(!vis[i]) exist &= (in[i] == 0 && graph[i].empty());
    }
    if(!exist)
    {
        cout << "IMPOSSIBLE" << endl;
        return;
    }

    dfs(1);

    reverse(all(path));
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

