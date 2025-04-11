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
#define ull unsigned long long
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
#define MAX(a) *max_element(all(a)) 
#define MIN(a) *min_element(all(a))
#define ROTATE(a, p) rotate(begin(a), begin(a) + p, end(a))
#define i128 __int128

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
    
template<typename T, size_t N>
istream& operator>>(istream& is, array<T, N>& arr) {
    for (size_t i = 0; i < N; i++) { is >> arr[i]; } return is;
}

template<typename T, size_t N>
istream& operator>>(istream& is, vector<array<T, N>>& vec) {
    for (auto &arr : vec) { is >> arr; } return is;
}
    
template <typename T1, typename T2>  istream &operator>>(istream& in, pair<T1, T2>& input) {    return in >> input.ff >> input.ss; }
    
template <typename T> istream &operator>>(istream &in, vector<T> &v) { for (auto &el : v) in >> el; return in; }

template<class T>
void output_vector(vt<T>& a, int off_set = 0) {
    int n = a.size();
    for(int i = off_set; i < n; i++) {
        cout << a[i] << (i == n - 1 ? '\n' : ' ');
    }
}

template<typename T, typename Compare>
vi closest_left(const vt<T>& a, Compare cmp) {
    int n = a.size(); vi closest(n); iota(all(closest), 0);
    for (int i = 0; i < n; i++) {
        auto& j = closest[i];
        while(j && cmp(a[i], a[j - 1])) j = closest[j - 1];
    }
    return closest;
}

template<typename T, typename Compare> // auto right = closest_right<int>(a, std::less<int>());
vi closest_right(const vt<T>& a, Compare cmp) {
    int n = a.size(); vi closest(n); iota(all(closest), 0);
    for (int i = n - 1; i >= 0; i--) {
        auto& j = closest[i];
        while(j < n - 1 && cmp(a[i], a[j + 1])) j = closest[j + 1];
    }
    return closest;
}

template<typename T, typename V = string>
vt<pair<T, int>> encode(const V& s) {
    vt<pair<T, int>> seg;
    for(auto& ch : s) {
        if(seg.empty() || ch != seg.back().ff) seg.pb({ch, 1});
        else seg.back().ss++;
    }
    return seg;
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
template<typename T1, typename T2>
std::ostream& operator<<(std::ostream& o, const std::pair<T1, T2>& p) { return o << "{" << p.ff << " , " << p.ss << "}"; }
auto operator<<(auto &o, const auto &x) -> decltype(end(x), o) {
    o << "{"; int i = 0; for (const auto &e : x) { if (i++) o << " , "; o << e; } return o << "}";
} // remove for leetcode
#include <sys/resource.h>
#include <sys/time.h>
void printMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    double memoryMB = usage.ru_maxrss / 1024.0;
    cerr << "Memory usage: " << memoryMB << " MB" << "\n";
}

#define startClock clock_t tStart = clock();
#define endClock std::cout << std::fixed << std::setprecision(10) << "\nTime Taken: " << (double)(clock() - tStart) / CLOCKS_PER_SEC << " seconds" << std::endl;
#else
#define debug(...)
#define startClock
#define endClock

#endif
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

#define eps 1e-9
#define M_PI 3.14159265358979323846
const static string pi = "3141592653589793238462643383279";
const static ll INF = 1LL << 62;
const static int inf = 1e9 + 100;
const static int MK = 20;
const static int MX = 1e5 + 5;
ll gcd(ll a, ll b) { while (b != 0) { ll temp = b; b = a % b; a = temp; } return a; }
ll lcm(ll a, ll b) { return (a / gcd(a, b)) * b; }
int pct(ll x) { return __builtin_popcountll(x); }
ll have_bit(ll x, int b) { return x & (1LL << b); }
int min_bit(ll x) { return __builtin_ctzll(x); }
int max_bit(ll x) { return 63 - __builtin_clzll(x); } 
const vvi dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}; // UP, DOWN, LEFT, RIGHT
const vc dirChar = {'U', 'D', 'L', 'R'};
int modExpo(ll base, ll exp, ll mod) { ll res = 1; base %= mod; while(exp) { if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>= 1; } return res; }
int modExpo_on_string(ll a, string exp, int mod) { ll b = 0; for(auto& ch : exp) b = (b * 10 + (ch - '0')) % (mod - 1); return modExpo(a, b, mod); }
ll sum_even_series(ll n) { return (n / 2) * (n / 2 + 1);} 
ll sum_odd_series(ll n) {return n - sum_even_series(n);} // sum of first n odd number is n ^ 2
ll sum_of_square(ll n) { return n * (n + 1) * (2 * n + 1) / 6; } // sum of 1 + 2 * 2 + 3 * 3 + 4 * 4 + ... + n * n
string make_lower(const string& t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return tolower(c); }); return s; }
string make_upper(const string&t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return toupper(c); }); return s; }
ll sqrt(ll n) { ll t = sqrtl(n); while(t * t < n) t++; while(t * t > n) t--; return t;}

class GRAPH { 
    public: 
    int n, m; 
    vvi dp, graph; 
    vi depth, parent, subtree;
    GRAPH(vvi& graph, int root = 0) {   
        this->graph = graph;
        n = graph.size();
        m = log2(n) + 1;
        dp.rsz(n, vi(m));
        depth.rsz(n);
        parent.rsz(n, -1);
		subtree.rsz(n, 1);
        dfs(root);
        init();
    }
    
    void dfs(int node = 0, int par = -1) {   
        for(auto& nei : graph[node]) {  
            if(nei == par) continue;    
            depth[nei] = depth[node] + 1;   
            dp[nei][0] = node;
            parent[nei] = node;
			dfs(nei, node);
			subtree[node] += subtree[nei];
        }
    }

    void init() {  
        for(int j = 1; j < m; j++) {   
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
        for(int i = m - 1; i >= 0; i--) {  
            if((d >> i) & 1) {  
                b = dp[b][i];
            }
        }
        if(a == b) return a;    
        for(int i = m - 1; i >= 0; i--) {  
            if(dp[a][i] != dp[b][i]) {  
                a = dp[a][i];   
                b = dp[b][i];
            }
        }
        return dp[a][0];
    }
};

template<class T, typename F = function<T(const T&, const T&)>>
class HLD {
    public:
    vi id, tp, sz, parent;
    int ct;
    vvi graph;
    int n;
    GRAPH g;
    F func;
    HLD(vvi& graph) : g(graph, 0), graph(graph), n(graph.size()) {
        this->parent = g.parent;
        this->sz = g.subtree;
        ct = 0;
        id.rsz(n), tp.rsz(n);
        dfs();
    }
        
    void dfs(int node = 0, int par = -1, int top = 0) {   
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
        dfs(nxt, node, top);   
        for(auto& nei : graph[node]) {   
            if(nei != par && nei != nxt) dfs(nei, node, nei);  
        }   
    }

	vpii get_path(int node, int par) {
        vpii seg;
        while(node != par && node) {   
            if(node == tp[node]) {   
                seg.pb({id[node], id[node]});
                node = parent[node];
            } else if(g.depth[tp[node]] > g.depth[par]) {   
                seg.pb({id[tp[node]], id[node]});
                node = parent[tp[node]];
            } else {   
                seg.pb({id[par] + 1, id[node]});
                break;  
            } 
        }   
        seg.pb({id[par], id[par]});
        return seg;
    }

    vpii get_path_u_v(int u, int v) {
        int p = g.lca(u, v);
        auto path = get_path(u, p);
        auto other = get_path(v, p);
        other.pop_back();
        rev(other);
        for(auto& [l, r] : path) swap(l, r);
        path.insert(end(path), all(other));
        return path;
    }
};

const int MM = MX * 350;
int ptr;
ll root[MM];
pll lazy[MM];
pii child[MM];
template<typename T>
struct lazy_PSGT {
    int n;
    T DEFAULT;
    void assign(int n, T DEFAULT) {
        this->n = n;
        this->DEFAULT = DEFAULT;
    }

    T merge(T a, T b) {
        return a + b;
    }

    int create_node(int prev) {
        ++ptr;
        if(ptr > MM) {
            cout << "NO" << '\n';
            exit(0);
        }
        root[ptr] = root[prev];
        lazy[ptr] = lazy[prev];
        child[ptr] = child[prev];
        return ptr;
    }

    void apply(int curr, int left, int right, pll val) {
        ll len = right - left + 1;
        root[curr] += len * val.ff + len * (len - 1) / 2 * val.ss;
        lazy[curr].ff += val.ff;
        lazy[curr].ss += val.ss;
    }

    void push_down(int curr, int left, int right) {
        pll zero = {0, 0};
        if(lazy[curr] == zero || left == right) return;
        int middle = midPoint;
        if(child[curr].ff) {
            child[curr].ff = create_node(child[curr].ff);
            apply(child[curr].ff, left, middle, lazy[curr]); 
        }
        if(child[curr].ss) {
            child[curr].ss = create_node(child[curr].ss);
            lazy[curr].ff += lazy[curr].ss * (middle - left + 1);
            apply(child[curr].ss, middle + 1, right, lazy[curr]);
        }
        lazy[curr] = zero;
    }

    void update_range(int &i, int start, int end, pll delta, bool is_prefix) {
        update_range(i, i, delta, start, end, 0, n - 1, is_prefix);
    }

    void update_range(int& curr, int prev, pll val, int start, int end, int left, int right, bool is_prefix) {
        push_down(curr, left, right);
        if(left > end || start > right) return;
        curr = create_node(prev);
        if(start <= left && right <= end) {
            apply(curr, left, right, {val.ss * (is_prefix ? (left - start) : (end - left)) + val.ff, is_prefix ? val.ss : -val.ss});
            push_down(curr, left, right);
            return;
        }
        int middle = midPoint;
        update_range(child[curr].ff, child[prev].ff, val, start, end, left, middle, is_prefix);
        update_range(child[curr].ss, child[prev].ss, val, start, end, middle + 1, right, is_prefix);
        root[curr] = merge(root[child[curr].ff], root[child[curr].ss]);
    }

    T queries_range(int i, int start, int end) {
        return queries_range(i, start, end, 0, n - 1);
    }

    T queries_range(int curr, int start, int end, int left, int right) {
        push_down(curr, left, right);
        if(start > right || left > end) return DEFAULT;
        if(start <= left && right <= end) return root[curr];
        int middle = midPoint;
        return merge(queries_range(child[curr].ff, start, end, left, middle), queries_range(child[curr].ss, start, end, middle + 1, right));
    }

    void update_at(int& i, int id, T delta) {
        update_at(i, i, id, delta, 0, n - 1);
    }

    void update_at(int &curr, int prev, int id, T delta, int left, int right) {  
        push_down(curr, left, right);
        curr = create_node(prev);
        if(left == right) { 
			root[curr] = merge(root[curr], delta);
            return;
        }
        int middle = midPoint;
        if(id <= middle) update_at(child[curr].ff, child[prev].ff, id, delta, left, middle); 
        else update_at(child[curr].ss, child[prev].ss, id, delta, middle + 1, right);
        root[curr] = merge(root[child[curr].ff], root[child[curr].ss]);
    }

    void reset() {
        for(int i = 0; i <= ptr; i++) {
            root[i] = 0;
            child[i] = {0, 0};
            lazy[i] = {0, 0};
        }
        ptr = 0;
    }
};

lazy_PSGT<ll> Tree;
int curr_time = 0;

void solve() {
    int n, q; cin >> n >> q;
    vvi graph(n);
    Tree.assign(n, 0);
    for(int i = 1; i < n; i++) {
        int u, v; cin >> u >> v;
        u--, v--;
        graph[u].pb(v);
        graph[v].pb(u);
    }
    for(int i = 0; i < n; i++) {
        Tree.update_at(curr_time, i, 0);
    }
    HLD<ll> g(graph);
    ll last = 0;
    int total = 1;
    vi state;
    state.pb(curr_time);
    while(q--) {
        char op; cin >> op;
        if(op == 'c') {
            int x, y;
            ll a, b; cin >> x >> y >> a >> b;
            x = ((x + last) % n);
            y = ((y + last) % n);
            auto path = g.get_path_u_v(x, y);
            ll off = 0;
            for(auto& [l, r] : path) {
                bool prefix = true;
                if(l > r) prefix = false, swap(l, r);
                Tree.update_range(curr_time, l, r, {a + off * b, b}, prefix);
                off += r - l + 1;
            }
            state.pb(curr_time);
            continue;
        }
        if(op == 'q') {
            int u, v; cin >> u >> v;
            u = ((u + last) % n);
            v = ((v + last) % n);
            auto path = g.get_path_u_v(u, v);
            last = 0;
            for(auto& [l, r] : path) {
                if(l > r) swap(l, r);
                last += Tree.queries_range(curr_time, l, r);
            }
            cout << last << '\n';
            continue;
        }
        int x; cin >> x;
        curr_time = state[(x + last) % int(state.size())];
    }
    Tree.reset();
}

signed main() {
    // careful for overflow, check for long long, use unsigned long long for random generator
    // when mle, look if problem require read in file, typically old problems
    IOS;
    startClock
    //generatePrime();

    int t = 1;
    //cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }

    endClock
    #ifdef LOCAL
      printMemoryUsage();
    #endif

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

