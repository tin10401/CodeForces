#pragma GCC target("popcnt")
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ll unsigned long long
#define int long long
#define pb push_back
#define vi vector<int>
#define pii pair<int, int>
#define vpii vector<pair<int, int>>
#define f first
#define s second
#define ar(x) array<int, x>
const static int INF = 1LL << 61;
const static int MOD = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

int modExpo(int base, int exp, int mod)
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
vi vis, ans_vis, lowestTime, insertTime;
vpii ans;
vector<vpii> graph;
int currTime = 0, cnt = 0;
void dfs(int node, int par)
{
    if(vis[node]) return;
    cnt++;
    vis[node] = true;
    lowestTime[node] = insertTime[node] = currTime++;
    for(auto& [nei, index] : graph[node])
    {
        if(nei == par) continue;
        if(!ans_vis[index])
        {
            ans.pb({node, nei});
            ans_vis[index] = true;
        }
        dfs(nei, node);
        lowestTime[node] = min(lowestTime[node], lowestTime[nei]);
        if(lowestTime[nei] > insertTime[node])
        {
            cout << "IMPOSSIBLE" << endl;
            exit(0);
        }
    }
}


 

void solve()
{
    int n, m; cin >> n >> m;
    graph.resize(n + 1), lowestTime.resize(n + 1), insertTime.resize(n + 1), vis.resize(n + 1), ans_vis.resize(m + 1);
    for(int i = 0; i < m; i++)
    {
        int a, b; cin >> a >> b;
        graph[a].pb({b, i});
        graph[b].pb({a, i});
    }
    dfs(1, 0);
    if(cnt != n)
    {
        cout << "IMPOSSIBLE" << endl;
        return;
    }
    for(auto& it : ans) cout << it.f << " " << it.s << endl;
    cout << endl;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

