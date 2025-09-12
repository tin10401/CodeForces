#include<bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;
template<class T> using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
#define vt vector
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ub upper_bound
#define lb lower_bound
#define db double
#define ld long db
#define ll long long
#define ull unsigned long long
#define vi vt<int>
#define vvi vt<vi>
#define vvvi vt<vvi>
#define pii pair<int, int>
#define vpii vt<pii>
#define vvpii vt<vpii>
#define vll vt<ll>  
#define vvll vt<vll>
#define pll pair<ll, ll>    
#define vpll vt<pll>
#define vvpll vt<vpll>
#define ar(x) array<int, x>
#define var(x) vt<ar(x)>
#define vvar(x) vt<var(x)>
#define al(x) array<ll, x>
#define vall(x) vt<al(x)>
#define vvall(x) vt<vall(x)>
#define vs vt<string>
#define pb push_back
#define ff first
#define ss second
#define rsz resize
#define sum(x) (ll)accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define rev(x) reverse(all(x))
#define MAX(a) *max_element(all(a)) 
#define MIN(a) *min_element(all(a))
#define SORTED(x) is_sorted(all(x))
#define ROTATE(a, p) rotate(begin(a), begin(a) + p, end(a))
#define i128 __int128
#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
#if defined(LOCAL) && __has_include("debug.h")
  #include "debug.h"
#else
  #define debug(...)
  #define startClock
  #define endClock
  inline void printMemoryUsage() {}
#endif
template<class T> using max_heap = priority_queue<T>; template<class T> using min_heap = priority_queue<T, vector<T>, greater<T>>;
template<typename T, size_t N> istream& operator>>(istream& is, array<T, N>& arr) { for (size_t i = 0; i < N; i++) { is >> arr[i]; } return is; }
template<typename T, size_t N> istream& operator>>(istream& is, vector<array<T, N>>& vec) { for (auto &arr : vec) { is >> arr; } return is; }
template<typename T1, typename T2>  istream &operator>>(istream& in, pair<T1, T2>& input) { return in >> input.ff >> input.ss; }
template<typename T> istream &operator>>(istream &in, vector<T> &v) { for (auto &el : v) in >> el; return in; }
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
const static ll INF = 1e18;
const static int inf = 1e9 + 100;
const static int MX = 1e5 + 5;

template<typename T>
struct CD { // centroid_decomposition
    int n, rt;
    vt<vt<T>> graph;
    vi size, parent, vis;
    ll ans;
    CD(const vt<vt<T>>& _graph) : graph(_graph), n(_graph.size()) {
        ans = 0;
        size.rsz(n);
        parent.rsz(n, -1);
        vis.rsz(n);
        rt = init();
    }
 
    void get_size(int node, int par) { 
        size[node] = 1;
        for(auto& [nei, w] : graph[node]) {
            if(nei == par || vis[nei]) continue;
            get_size(nei, node);
            size[node] += size[nei];
        }
    }
 
    int get_center(int node, int par, int size_of_tree) { 
        for(auto& [nei, w] : graph[node]) {
            if(nei == par || vis[nei]) continue;
            if(size[nei] * 2 > size_of_tree) return get_center(nei, node, size_of_tree);
        }
        return node;
    }

    int get_centroid(int src) { 
        get_size(src, -1);
        int centroid = get_center(src, -1, size[src]);
        vis[centroid] = true;
        return centroid;
    }

    int mx;
    void modify(int node, int par, int depth, int delta) {
        for(auto& [nei, w] : graph[node]) {
            if(vis[nei] || nei == par) continue;
            modify(nei, node, depth + 1, delta);
        }
    }

    void cal(int node, int par, int depth) {
        for(auto& [nei, w] : graph[node]) {
            if(vis[nei] || nei == par) continue;
            cal(nei, node, depth + 1);
        }
    }
 
    int get_max_depth(int node, int par = -1, int depth = 0) {
        int max_depth = depth;
        for(auto& [nei, w] : graph[node]) {
            if(nei == par || vis[nei]) continue;
            max_depth = max(max_depth, get_max_depth(nei, node, depth + 1));
        }
        return max_depth;
    }

    void run(int root, int par) {
        mx = get_max_depth(root, par);
        for(auto& [nei, w] : graph[root]) {
            if(vis[nei] || nei == par) continue;
            cal(nei, root, 1);
            modify(nei, root, 1, 1);
        }
    }

    int init(int root = 0, int par = -1) {
        root = get_centroid(root);
        parent[root] = par;
        run(root, par);
        for(auto& [nei, w] : graph[root]) {
            if(nei == par || vis[nei]) continue;
            init(nei, root);
        }
        return root;
    }
};

void solve() {
    int n; cin >> n;
    vvpll graph(n);
    for(int i = 1; i < n; i++) {
        ll u, v, w; cin >> u >> v >> w;
        u--, v--;
        graph[u].pb({v, w});
        graph[v].pb({u, w});
    }
    CD<pll> g(graph);
    cout << g.ans << '\n';
}

signed main() {
    IOS;
    startClock
    int t = 1;
    cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
