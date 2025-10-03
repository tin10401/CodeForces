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
const static ll INF = 4e18 + 10;
const static int inf = 1e9 + 100;
const static int MX = 1e5 + 5;

struct EulerianPath {
    int nodes, edges;
    bool directed;
    vvpii graph;
    vi deg, indeg, outdeg;
    vt<bool> used;
    vpii ans_edges;

    EulerianPath(int _nodes, bool _directed = false)
      : nodes(_nodes), edges(0), directed(_directed), graph(_nodes) {
        if(directed) indeg.assign(nodes,0), outdeg.assign(nodes,0);
        else deg.assign(nodes,0);
    }

    void add_edge(int u, int v, int id) {
        graph[u].emplace_back(v, id);
        edges++;
        if(directed) {
            outdeg[u]++;
            indeg[v]++;
        } else {
            graph[v].emplace_back(u, id);
            deg[u]++;
            deg[v]++;
        }
    }

    int find_start() const {
        int start = -1;
        if(!directed) {
            int odd = 0;
            for(int i = 0; i < nodes; i++) {
                if(deg[i] & 1) {
                    odd++;
                    start = i;
                }
                if(start < 0 && deg[i] > 0) start = i;
            }
            if(start < 0) return 0;
            if(odd != 0 && odd != 2) return -1;
        } else {
            int plus1 = 0, minus1 = 0;
            for(int i = 0; i < nodes; i++) {
                int d = outdeg[i] - indeg[i];
                if(d == 1) { plus1++; start = i; }
                else if(d == -1) minus1++;
                else if(d != 0) return -1;
                if(start < 0 && outdeg[i] > 0) start = i;
            }
            if(start < 0) return 0;
            if(!((plus1 == 1 && minus1 == 1) || (plus1 == 0 && minus1 == 0))) return -1;
        }
        return start;
    }

    void dfs(int u) {
		if(used.empty()) {
			used.rsz(edges);
		}
        while(!graph[u].empty()) {
            auto [v, id] = graph[u].back();
            graph[u].pop_back();
			while((int)used.size() <= id) used.pb(0);
            if(used[id]) continue;
            used[id] = true;
            dfs(v);
            ans_edges.pb({u, v});
        }
    }

    vpii curr;

    vpii get_path() {
        int start = find_start();
        if(start < 0) return {};
        used.rsz(edges);
        dfs(start);
        if((int)ans_edges.size() != edges) return {};
        return ans_edges;
    }

};

void solve() {
    int n, m; cin >> n >> m;
    vi degree(n);
    EulerianPath graph(n, false);
    for(int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        u--, v--;
        degree[u]++;
        degree[v]++;
        graph.add_edge(u, v, i);
    }
    int odd = 0;
    int s = 0;
    for(int i = 0, last = -1; i < n; i++) {
        if(degree[i] & 1) {
            if(last == -1) last = i;
            else {
                graph.add_edge(last, i, m++);
                last = -1;
            }
            degree[i]++;
        }
        if(degree[i] % 4) {
            odd ^= 1;
            s = i;
        }
    }
    if(odd) {
        graph.add_edge(s, s, m++);
    }
    auto edges = graph.get_path();
    cout << edges.size() << '\n';
    int sw = 0;
    // a->b<-c->d<-e->f<-...
    // by doing this, all in degree is going to be even for all nodes -> make all out degree even for all nodes too
    for(auto& [u, v] : edges) {
        if(sw) swap(u, v);
        sw ^= 1;
        cout << u + 1 << ' ' << v + 1 << '\n';
    }
}

signed main() {
    IOS;
    startClock
    int t = 1;
    //cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
