//████████╗██╗███╗░░██╗  ██╗░░░░░███████╗
//╚══██╔══╝██║████╗░██║  ██║░░░░░██╔════╝
//░░░██║░░░██║██╔██╗██║  ██║░░░░░█████╗░░
//░░░██║░░░██║██║╚████║  ██║░░░░░██╔══╝░░
//░░░██║░░░██║██║░╚███║  ███████╗███████╗
//░░░╚═╝░░░╚═╝╚═╝░░╚══╝  ╚══════╝╚══════╝
//   __________________
//  | ________________ |
//  ||          ____  ||
//  ||   /\    |      ||
//  ||  /__\   |      ||
//  || /    \  |____  ||
//  ||________________||
//  |__________________|
//  \###################\
//   \###################\
//    \        ____       \
//     \_______\___\_______\
// An AC a day keeps the doctor away.

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <ctime>
#include <cassert>
#include <complex>
#include <string>
#include <cstring>
#include <chrono>
#include <random>
#include <bitset>
#include <iomanip>
#include <functional>
#include <numeric>
#include <stack>
#include <array>
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
#define vll vt<ll>  
#define vvll vt<vll>
#define pll pair<ll, ll>    
#define vpll vt<pll>
#define vvpll vt<vpll>
#define vc vt<char> 
#define vvc vt<vc>
#define vi vt<int>
#define vvi vt<vi>
#define vvvi vt<vvi>
#define pii pair<int, int>
#define vpii vt<pii>
#define vs vt<string>
#define vvs vt<vs>
#define vb vt<bool>
#define vvb vt<vb>
#define vvpii vt<vpii>
#define vd vt<db>
#define ar(x) array<int, x>
#define var(x) vt<ar(x)>
#define vvar(x) vt<var(x)>
#define al(x) array<ll, x>
#define vall(x) vt<al(x)>
#define vvall(x) vt<vall(x)>
#define mset(m, v) memset(m, v, sizeof(m))
#define pb push_back
#define ff first
#define ss second
#define sv string_view
#define MP make_pair
#define MT make_tuple
#define rsz resize
#define sum(x) (ll)accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define SORTED(x) is_sorted(all(x))
#define rev(x) reverse(all(x))
#define gcd(a, b) __gcd(a, b)
#define lcm(a, b) (a * b) / gcd(a, b)
#define MAX(a) *max_element(all(a)) 
#define MIN(a) *min_element(all(a))

//SGT DEFINE
#define lc i * 2 + 1
#define rc i * 2 + 2
#define lp lc, left, middle
#define rp rc, middle + 1, right
#define entireTree 0, 0, n - 1
#define midPoint left + (right - left) / 2
#define pushDown push(i, left, right)
#define iter int i, int left, int right

#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)

struct custom {
    static const uint64_t C = 0x9e3779b97f4a7c15; const uint32_t RANDOM = std::chrono::steady_clock::now().time_since_epoch().count();
    size_t operator()(uint64_t x) const { return __builtin_bswap64((x ^ RANDOM) * C); }
    size_t operator()(const std::string& s) const { size_t hash = std::hash<std::string>{}(s); return hash ^ RANDOM; } };
template <class K, class V> using umap = std::unordered_map<K, V, custom>; template <class K> using uset = std::unordered_set<K, custom>;
template<class T> using max_heap = priority_queue<T>;
template<class T> using min_heap = priority_queue<T, vector<T>, greater<T>>;
    
template<typename T1, typename T2>
std::ostream& operator<<(std::ostream& o, const std::pair<T1, T2>& p) { return o << "{" << p.ff << " , " << p.ss << "}"; }
auto operator<<(auto &o, const auto &x) -> decltype(end(x), o) {
    o << "{"; int i = 0; for (const auto &e : x) { if (i++) o << " , "; o << e; } return o << "}";
} // remove for leetcode
    
template <typename T1, typename T2>  istream &operator>>(istream& in, pair<T1, T2>& input) {    return in >> input.ff >> input.ss; }
    
template <typename T> istream &operator>>(istream &in, vector<T> &v) { for (auto &el : v) in >> el; return in; }

template<class T>
void output_vector(vt<T>& a, int off_set = 0) {
    int n = a.size();
    for(int i = off_set; i < n; i++) {
        cout << a[i] << (i == n - 1 ? '\n' : ' ');
    }
}
    
template<typename K, typename V>
auto operator<<(std::ostream &o, const std::map<K, V> &m) -> std::ostream& {
    o << "{"; int i = 0;
    for (const auto &[key, value] : m) { if (i++) o << " , "; o << key << " : " << value; }
    return o << "}";
}

#ifdef LOCAL
#define debug(x...) debug_out(#x, x)
void debug_out(const char* names) { std::cerr << std::endl; }
template <typename T, typename... Args>
void debug_out(const char* names, T value, Args... args) {
    const char* comma = strchr(names, ',');
    std::cerr << "[" << (comma ? std::string(names, comma) : names) << " = " << value << "]";
    if (sizeof...(args)) { std::cerr << ", "; debug_out(comma + 1, args...); }   
    else { std::cerr << std::endl; }
}
#define startClock clock_t tStart = clock();
#define endClock std::cout << std::fixed << std::setprecision(10) << "\nTime Taken: " << (double)(clock() - tStart) / CLOCKS_PER_SEC << " seconds" << std::endl;
#else
#define debug(...)
#define startClock
#define endClock

#endif
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

#define eps 1e-9
#define M_PI 3.14159265358979323846
const static ll INF = 1LL << 62;
const static int inf = 1e9 + 100;
const static int MK = 20;
const static int MX = 1e5 + 5;
const static int MOD = 1e9 + 7;
int pct(ll x) { return __builtin_popcountll(x); }
const vvi dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}; // UP, DOWN, LEFT, RIGHT
const vc dirChar = {'U', 'D', 'L', 'R'};
int modExpo(ll base, ll exp, ll mod) { ll res = 1; base %= mod; while(exp) { if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>= 1; } return res; }

class GRAPH { 
    public: 
    int n;  
    vvi dp, graph; 
    vi depth;
    GRAPH(vvi& graph) {   
        this->graph = graph;
        n = graph.size();
        dp.rsz(n, vi(MK));
        depth.rsz(n);
        dfs();
        init();
    }
    
    void dfs(int node = 0, int par = -1) {   
        for(auto& nei : graph[node]) {  
            if(nei == par) continue;    
            depth[nei] = depth[node] + 1;   
            dp[nei][0] = node;
			dfs(nei, node);
        }
    }
    
    void init() {  
        for(int j = 1; j < MK; j++) {   
            for(int i = 0; i < n; i++) {    
                dp[i][j] = dp[dp[i][j - 1]][j - 1];
            }
        }
    }

    int lca(int a, int b) { 
        if(depth[a] > depth[b]) {   
            swap(a, b);
        }
        int d = depth[b] - depth[a];    
        for(int i = MK - 1; i >= 0; i--) {  
            if((d >> i) & 1) {  
                b = dp[b][i];
            }
        }
        if(a == b) return a;    
        for(int i = MK - 1; i >= 0; i--) {  
            if(dp[a][i] != dp[b][i]) {  
                a = dp[a][i];   
                b = dp[b][i];
            }
        }
        return dp[a][0];
    }

    int dist(int u, int v) {    
        int a = lca(u, v);  
        return depth[u] + depth[v] - 2 * depth[a];
    }
};

template<class T>   
class SGT { 
    public: 
    int n;  
    vt<T> root;
    T DEFAULT;
	SGT(int n) {    
        this->n = n;
        DEFAULT = 0;
        root.rsz(n * 4);    
    }
    
    void update(int id, T val) {  
        update(entireTree, id, val);
    }
    
    void update(iter, int id, T val) {  
        if(left == right) { 
            root[i] = val;  
            return;
        }
        int middle = midPoint;  
        if(id <= middle) update(lp, id, val);   
        else update(rp, id, val);   
        root[i] = merge(root[lc], root[rc]);
    }

    T merge(T left, T right) {  
        T res;
        res = left + right;
        return res;
    }

    T queries(int start, int end) { 
        return queries(entireTree, start, end);
    }
    
    T queries(iter, int start, int end) {   
        if(left > end || start > right) return DEFAULT;
        if(left >= start && right <= end) return root[i];   
        int middle = midPoint;  
        return merge(queries(lp, start, end), queries(rp, start, end));
    }
};

class HLD {
    public:
    SGT<int> seg;
    vi id, a, tp, sz, parent;
    int ct;
    vvi graph;
    int n;
    GRAPH g;
    HLD(vvi& graph, vi& a) : seg(graph.size()), g(graph) {
        this->graph = graph;
        this->n = graph.size();
        this->a = a;
        ct = 0;
        id.rsz(n), tp.rsz(n), sz.rsz(n);
        parent.rsz(n, -1);
        dfs1();
        dfs2();
    }
     
    int dfs1(int node = 0, int par = -1) {   
        parent[node] = par;
        sz[node] = 1;   
        for(auto& nei : graph[node]) {   
            if(nei == par) continue;    
            sz[node] += dfs1(nei, node);
        }   
        return sz[node];    
    }
        
    void dfs2(int node = 0, int par = -1, int top = 0) {   
        id[node] = ct++;    
        tp[node] = top;
        int nxt = -1, max_size = -1;    
        for(auto& nei : graph[node]) {   
            if(nei == par) continue;    
            if(sz[nei] > max_size) {   
                max_size = sz[nei]; 
                nxt = nei;  
            }   
        }   
        if(nxt == -1) return;   
        dfs2(nxt, node, top);   
        for(auto& nei : graph[node]) {   
            if(nei != par && nei != nxt) dfs2(nei, node, nei);  
        }   
    }

    void update(int i, int v) {
        a[i] = v;
        seg.update(id[i], v);
    }

    int path(int node, int par) {   
        int res = 0;    
        while(node != par)  {   
            if(node == tp[node])   {   
                res += a[node];
                node = g.dp[node][0]; 
            }   else if(g.depth[tp[node]] > g.depth[par])  {   
                res += seg.queries(id[tp[node]], id[node]);    
                node = g.dp[tp[node]][0];
            }   else    {   
                res += seg.queries(id[par] + 1, id[node]);  
                break;  
            }   
        }   
        return res; 
    }

    int get_dist(int a, int b) {
        return g.dist(a, b);
    }

    int get_lca(int a, int b) {
        return g.lca(a, b);
    }
};

void solve() {
    int n, q; cin >> n >> q;
    vi a(n); cin >> a;
    vvi graph(n);
    for(int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        u--, v--;
        graph[u].pb(v);
        graph[v].pb(u);
    }
    HLD root(graph, a);
    set<int> s;
    vi degree(n);
    int cnt = 0;
    auto flip = [&](int v) {
        int p = root.parent[v];
        if(a[v] == 0) {
            ++cnt;
            a[v] = 1;
            if (p != -1) {
                degree[p]++;
                s.erase(p);
            }
            if (degree[v] == 0) {
                s.insert(v);
            }
        } else {
            --cnt;
            a[v] = 0;
            s.erase(v);
            if(p != -1) {
                if(--degree[p] == 0 && a[p]) {
                    s.insert(p);
                }
            }
        }
        root.update(v, a[v]);
    };
    for(int i = 0; i < n; i++) {
        if(a[i]) {
            a[i] = 0;
            flip(i);
        }
    }
    auto queries = [&]() -> int {
        if(cnt == 0) return false;
        if(cnt == 1) return true;
        if(s.size() == 1) return true;
        int u = *s.begin(), v = *s.rbegin();
        int c = root.get_lca(u, v);
        int sm = root.path(u, c) + root.path(v, c) + a[c];
        int d = root.get_dist(u, v) + 1;
        return d == cnt && sm == cnt;
    };
    while(q--) {
        int v; cin >> v;
        v--;
        flip(v);
        cout << (queries() ? "YES" : "NO") << endl;
    }

}

signed main() {
    IOS;
    startClock
    //generatePrime();

    int t = 1;
    cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }

    endClock
    return 0;
}

//███████████████████████████████████████████████████████████████████████████████████████████████████████
//█░░░░░░░░░░░░░░█░░░░░░██████████░░░░░░█░░░░░░░░░░░░███░░░░░░░░░░█░░░░░░██████████░░░░░░█░░░░░░░░░░░░░░█
//█░░▄▀▄▀▄▀▄▀▄▀░░█░░▄▀░░░░░░░░░░██░░▄▀░░█░░▄▀▄▀▄▀▄▀░░░░█░░▄▀▄▀▄▀░░█░░▄▀░░░░░░░░░░██░░▄▀░░█░░▄▀▄▀▄▀▄▀▄▀░░█
//█░░▄▀░░░░░░░░░░█░░▄▀▄▀▄▀▄▀▄▀░░██░░▄▀░░█░░▄▀░░░░▄▀▄▀░░█░░░░▄▀░░░░█░░▄▀▄▀▄▀▄▀▄▀░░██░░▄▀░░█░░▄▀░░░░░░░░░░█
//█░░▄▀░░█████████░░▄▀░░░░░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░███░░▄▀░░███░░▄▀░░░░░░▄▀░░██░░▄▀░░█░░▄▀░░█████████
//█░░▄▀░░░░░░░░░░█░░▄▀░░██░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░███░░▄▀░░███░░▄▀░░██░░▄▀░░██░░▄▀░░█░░▄▀░░█████████
//█░░▄▀▄▀▄▀▄▀▄▀░░█░░▄▀░░██░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░███░░▄▀░░███░░▄▀░░██░░▄▀░░██░░▄▀░░█░░▄▀░░██░░░░░░█
//█░░▄▀░░░░░░░░░░█░░▄▀░░██░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░███░░▄▀░░███░░▄▀░░██░░▄▀░░██░░▄▀░░█░░▄▀░░██░░▄▀░░█
//█░░▄▀░░█████████░░▄▀░░██░░▄▀░░░░░░▄▀░░█░░▄▀░░██░░▄▀░░███░░▄▀░░███░░▄▀░░██░░▄▀░░░░░░▄▀░░█░░▄▀░░██░░▄▀░░█
//█░░▄▀░░░░░░░░░░█░░▄▀░░██░░▄▀▄▀▄▀▄▀▄▀░░█░░▄▀░░░░▄▀▄▀░░█░░░░▄▀░░░░█░░▄▀░░██░░▄▀▄▀▄▀▄▀▄▀░░█░░▄▀░░░░░░▄▀░░█
//█░░▄▀▄▀▄▀▄▀▄▀░░█░░▄▀░░██░░░░░░░░░░▄▀░░█░░▄▀▄▀▄▀▄▀░░░░█░░▄▀▄▀▄▀░░█░░▄▀░░██░░░░░░░░░░▄▀░░█░░▄▀▄▀▄▀▄▀▄▀░░█
//█░░░░░░░░░░░░░░█░░░░░░██████████░░░░░░█░░░░░░░░░░░░███░░░░░░░░░░█░░░░░░██████████░░░░░░█░░░░░░░░░░░░░░█
//███████████████████████████████████████████████████████████████████████████████████████████████████████
