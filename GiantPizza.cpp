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

vector<vector<int>> graph, revGraph;
vector<int> vis, comp;

int flip(int x)
{
    return x & 1 ? ++x : --x;
}


void merge(char c1, int a, char c2, int b)
{
    a = a * 2 - (c1 == '-');
    b = b * 2 - (c2 == '-');
    graph[flip(a)].push_back(b);
    graph[flip(b)].push_back(a);
    revGraph[a].push_back(flip(b));
    revGraph[b].push_back(flip(a));
}

void dfs(int node, stack<int>& s)
{
    if(vis[node]) return;
    vis[node] = true;
    for(auto& nei : graph[node]) dfs(nei, s);
    s.push(node);
}

void dfs(int node, int curr)
{
    if(vis[node]) return;
    vis[node] = true;
    comp[node] = curr;
    for(auto& nei : revGraph[node]) dfs(nei, curr);
}

void solve()
{
   int n, m;
   cin >> n >> m;

   graph.resize(m * 2 + 1), revGraph.resize(m * 2 + 1), comp.resize(m * 2 + 1, -1), vis.resize(m * 2 + 1);
   for(int i = 0; i < n; i++)
   {
       char c1, c2;
       int a, b;
       cin >> c1 >> a >> c2 >> b;
       merge(c1, a, c2, b);
   }

   stack<int> s;
   for(int i = 1; i <= 2 * m; i++) dfs(i, s);

   fill(all(vis), 0);
   int curr = 1; 
   while(!s.empty())
   {
       int node = s.top(); s.pop();
       if(vis[node]) continue;
       dfs(node, curr++);
   }

   bool found = true;
   vector<char> res(m);
   for(int i = 1; i < 2 * m && found; i += 2)
   {
        if(comp[i] == comp[i + 1]) found = false;
        res[i / 2] = comp[i] > comp[i + 1] ? '-' : '+';
   }

   if(!found) cout << "IMPOSSIBLE" << endl;
   else
   {
       for(auto& it : res) cout << it << " ";
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

