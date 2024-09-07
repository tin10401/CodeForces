#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) (x).begin(), (x).end()
#define allr(x) (x).rbegin(), (x).rend()
#define ll unsigned long long
#define int long long
#define vi vector<int>
#define pi pair<int, int>
#define vii vector<vector<int>>
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

vector<bool> vis;
vector<vi> adj, radj;
vi parent;
int n;
bool reachable()
{
    fill(all(vis), false);
    vis[1] = true;
    queue<int> q;
    q.push(1);
    while(!q.empty())
    {
        int node = q.front();
        q.pop();
        for(int i = 1; i <= n; i++) if(adj[node][i] && !vis[i]) vis[i] = true, parent[i] = node, q.push(i);
    }
    return vis[n];
}
void solve()
{
    int m;
    cin >> n >> m;
    radj.resize(n + 1, vector<int>(n + 1)), adj.resize(n + 1, vector<int>(n + 1)), vis.resize(n + 1), parent.resize(n + 1);
    for(int i = 0; i < m; i++)
    {
        int a, b;
        cin >> a >> b;
        adj[a][b]++, adj[b][a]++, radj[a][b]++, radj[b][a]++;
    }
    while(reachable())
    {
        int flow = INF;
        for(int v = n; v != 1; v = parent[v])
        {
            int u = parent[v];
            flow = min(flow, adj[u][v]);
        }

        for(int v = n; v != 1; v = parent[v])
        {
            int u = parent[v];
            adj[u][v] -= flow;
            adj[v][u] += flow;
        }
    }

    reachable();
    vector<array<int, 2>> res;
    for(int i = 1; i <= n; i++)
    {
        for(int j = 1; j <= n; j++) if(radj[i][j] && vis[i] && !vis[j]) res.push_back({i, j});
    }

    cout << res.size() << endl;
    for(auto& it : res) cout << it[0] << " " << it[1] << endl; 
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

