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
const static int MX = 2e5 + 5;

int n, M;
struct EulerianPath {
    int nodes, edges;
    bool directed;
    vvpii graph;
    vi deg, indeg, outdeg;
    vt<bool> used;

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

    string s;
    var(3) ans_edges;
    void dfs(int u) {
        while(!graph[u].empty()) {
            auto [v, id] = graph[u].back();
            graph[u].pop_back();
            if(used[id]) continue;
            used[id] = true;
            dfs(v);
            ans_edges.pb({id, u, v});
        }
    }

    var(3) get_path() {
        used.rsz(edges);
        for(int i = 0; i < n; i++) {
            dfs(i);
        }
        s = string(edges, '?');
        rev(ans_edges);
        return ans_edges;
    }
};

void solve() {
    // d1^2 + d2^2 
    // di = d1 + d2
    // si = d1 - d2
    // di + si = 2 * d1
    // d1 = (di + si) / 2
    // d2 = (di - si) / 2
    // d1^2 + d2^2 = ((di + si)^2 + (di - si) ^ 2) / 4
    //             = (di^2 + 2disi + si^2 + di^2 + - 2disi + si^2) / 4
    //             = (di^2 + si^2) / 2
    int m; cin >> n >> m;
    EulerianPath graph(n, false);
    vi degree(n);
    vpii edges;
    for(int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        u--, v--;
        edges.pb({u, v});
        graph.add_edge(u, v, i);
        degree[u]++;
        degree[v]++;
    }
    M = m;
    int last = -1, l2 = -1;
    ll res = 0, brute = 0;
    vi L;
    for(int i = 0; i < n; i++) {
        ll x = degree[i] / 2;
        ll y = degree[i] - x;
        brute += (ll)degree[i] * degree[i];
        res += x * x + y * y;
        if(degree[i] & 1) {
            brute++;
            if(last == -1) last = i;
            else {
                graph.add_edge(last, i, m++);
                edges.pb({last, i});
                last = -1;
            }
            degree[i]++;
        }
    }
    auto ans_edges = graph.get_path();
    debug(ans_edges);
    vi s(m);
    vi d(n);
    int c = 0;
    vvpii G(n);
    for(auto& [id, u, v] : ans_edges) {
        s[id] = c ^= 1;
    }
    vvi D(n, vi(2));
    for(int i = 0; i < M; i++) {
        auto& [u, v] = edges[i];
        int g = s[i];
        D[u][g]++;
        D[v][g]++;
    }
    ll A = 0;
    for(int i = 0; i < n; i++) {
        if(abs(D[i][0] - D[i][1]) > 1) {
            debug(s);
            debug(edges, D, i);
            exit(0);
        }
        for(auto& x : D[i]) {
            A += (ll)x * x;
        }
    }
    assert(A == res);
    cout << res << ' ';
    for(auto& x : s) {
        cout << x + 1;
    }
    cout << '\n';
}

signed main() {
    IOS;
    int t = 1;
    cin >> t;
    for(int i = 1; i <= t; i++) {   
        cout << "Case #" << i << ": ";  
        solve();
    }
    return 0;
}
