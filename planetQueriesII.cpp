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

vector<int> len, vis;
vector<vector<int>> dp;

void dfs(int node)
{
    if(vis[node]) return;
    vis[node] = true;
    dfs(dp[node][0]);
    len[node] = len[dp[node][0]]+ 1;
}
int m = 23;
int LCA(int node, int dis)
{
    if(dis < 0) return -1;
    for(int i = 0; i < m; i++)
    {
        if((dis >> i) & 1) node = dp[node][i];
    }
    return node;
}

void solve()
{
    int n, q;
    cin >> n >> q;
    dp.resize(n + 1, vector<int>(m)), len.resize(n + 1), vis.resize(n + 1);
    for(int i = 1; i <= n; i++) cin >> dp[i][0];
    for(int j = 1; j < m; j++)
    {
        for(int i = 1; i <= n; i++)
        {
            dp[i][j] = dp[dp[i][j - 1]][j - 1];
        }
    }
    for(int i = 1; i <= n; i++) dfs(i);
    
    while(q--)
    {
        int a, b;
        cin >> a >> b;
        int aa = LCA(a, len[a]);
        if(LCA(a, len[a] - len[b]) == b) 
        {
            cout << len[a] - len[b] << endl;
        }
        else if(LCA(aa, len[aa] - len[b]) == b)
        {
            cout << len[a] << " " << len[aa] << " " << len[b] << endl;
            cout << len[aa] - len[b] + len[a] << endl;
        }
        else cout << -1 << endl;
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

