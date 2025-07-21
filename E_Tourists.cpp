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

inline std::ostream& operator<<(std::ostream& os, i128 x) {
    if(x == 0) { os << '0'; return os; } if(x < 0) { os << '-'; x = -x; }
    string s; while (x > 0) { int digit = int(x % 10); s.pb(char('0' + digit)); x /= 10; }
    rev(s); os << s; return os;
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
ll floor(ll a, ll b) { if(b < 0) a = -a, b = -b; if (a >= 0) return a / b; return a / b - (a % b ? 1 : 0); }
ll ceil(ll a, ll b) { if (b < 0) a = -a, b = -b; if (a >= 0) return (a + b - 1) / b; return a / b; }
int pct(ll x) { return __builtin_popcountll(x); }
ll have_bit(ll x, int b) { return (x >> b) & 1; }
int min_bit(ll x) { return __builtin_ctzll(x); }
int max_bit(ll x) { return 63 - __builtin_clzll(x); } 
const vvi dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}; // UP, DOWN, LEFT, RIGHT
const vvi knight_dirs = {{-2, -1}, {-2,  1}, {-1, -2}, {-1,  2}, {1, -2}, {1,  2}, {2, -1}, {2,  1}}; // knight dirs
const vc dirChar = {'U', 'D', 'L', 'R'};
int modExpo(ll base, ll exp, ll mod) { ll res = 1; base %= mod; while(exp) { if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>= 1; } return res; }
ll extended_gcd(ll a, ll b, ll &x, ll &y) { if (b == 0) { x = 1; y = 0; return a; } ll d = extended_gcd(b, a % b, y, x); y -= (a / b) * x; return d; }
int modExpo_on_string(ll a, string exp, int mod) { ll b = 0; for(auto& ch : exp) b = (b * 10 + (ch - '0')) % (mod - 1); return modExpo(a, b, mod); }
ll sum_even_series(ll n) { return (n / 2) * (n / 2 + 1);} 
ll sum_odd_series(ll n) {return n - sum_even_series(n);} // sum of first n odd number is n ^ 2
ll sum_of_square(ll n) { return n * (n + 1) * (2 * n + 1) / 6; } // sum of 1 + 2 * 2 + 3 * 3 + 4 * 4 + ... + n * n
string make_lower(const string& t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return tolower(c); }); return s; }
string make_upper(const string&t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return toupper(c); }); return s; }
ll sqrt(ll n) { ll t = sqrtl(n); while(t * t < n) t++; while(t * t > n) t--; return t;}
template<typename T> T geometric_sum(ll n, ll k) { return (1 - T(n).pow(k + 1)) / (1 - n); } // return n^1 + n^2 + n^3 + n^4 + n^5 + ... + n^k
template<typename T> T geometric_power(ll p, ll k) { return (T(p).pow(k + 1) - 1) / T(p - 1); } // p^1 + p^2 + p^3 + ... + p^k
bool is_perm(ll sm, ll square_sum, ll len) {return sm == len * (len + 1) / 2 && square_sum == len * (len + 1) * (2 * len + 1) / 6;} // determine if an array is a permutation base on sum and square_sum
bool is_vowel(char c) {return c == 'a' || c == 'e' || c == 'u' || c == 'o' || c == 'i';}

template<typename T = int>
class GRAPH {
public:
	int n, m;
    vvi dp;
    vi parent, subtree;
    vi tin, tout, low, ord, depth;
    vll depth_by_weight;
    vvi weight;
    int timer = 0;
    vt<unsigned> in_label, ascendant;
    vi par_head;
    unsigned cur_lab = 1;
    vt<vt<T>> adj;

    GRAPH() {}

    GRAPH(const vt<vt<T>>& graph, int root = 0) {
        adj = graph;
        n = graph.size();
        m = log2(n) + 1;
//        depth_by_weight.rsz(n);
//        weight.rsz(n, vi(m));
        dp.rsz(n, vi(m, -1));
        depth.rsz(n);
        parent.rsz(n, -1);
        subtree.rsz(n, 1);
        tin.rsz(n);
        tout.rsz(n);
        ord.rsz(n);
        dfs(root);
        init();
        in_label.rsz(n);
        ascendant.rsz(n);
        par_head.rsz(n + 1);
        sv_dfs1(root);
        ascendant[root] = in_label[root];
        sv_dfs2(root);
    }

	void dfs(int node, int par = -1) {
        tin[node] = timer++;
        ord[tin[node]] = node;
        for (auto& nei : adj[node]) {
            if (nei == par) continue;
            depth[nei] = depth[node] + 1;
//            depth_by_weight[nei] = depth_by_weight[node] + w;
//            weight[nei][0] = w;
            dp[nei][0] = node;
            parent[nei] = node;
            dfs(nei, node);
            subtree[node] += subtree[nei];
			// timer++; // use when merging segtree nodes for different subtree, size then become 2 * n
        }
        tout[node] = timer - 1;
    }

    bool is_ancestor(int par, int child) { return tin[par] <= tin[child] && tin[child] <= tout[par]; }

	void init() {
        for (int j = 1; j < m; ++j) {
            for (int i = 0; i < n; ++i) {
                int p = dp[i][j - 1];
                if(p == -1) continue;
                //weight[i][j] = max(weight[i][j - 1], weight[p][j - 1]);
                dp[i][j] = dp[p][j - 1];
            }
        }
    }


    void sv_dfs1(int u, int p = -1) {
        in_label[u] = cur_lab++;
        for(auto& v : adj[u]) if (v != p) {
            sv_dfs1(v, u);
            if(std::__countr_zero(in_label[v]) > std::__countr_zero(in_label[u]))
                in_label[u] = in_label[v];
        }
    }

    void sv_dfs2(int u, int p = -1) {
        for(auto& v : adj[u]) if (v != p) {
            ascendant[v] = ascendant[u];
            if(in_label[v] != in_label[u]) {
                par_head[in_label[v]] = u;
                ascendant[v] += in_label[v] & -in_label[v];
            }
            sv_dfs2(v, u);
        }
    }

    int lift(int u, unsigned j) const {
        unsigned k = std::__bit_floor(ascendant[u] ^ j);
        return k == 0 ? u : par_head[(in_label[u] & -k) | k];
    }

    int lca(int a, int b) {
        if(is_ancestor(a, b)) return a;
        if(is_ancestor(b, a)) return b;
        auto [x, y] = std::minmax(in_label[a], in_label[b]);
        unsigned j = ascendant[a] & ascendant[b] & -std::__bit_floor((x - 1) ^ y);
        a = lift(a, j);
        b = lift(b, j);
        return depth[a] < depth[b] ? a : b;
    }

    int path_queries(int u, int v) { // lca in logn
        if(depth[u] < depth[v]) swap(u, v);
        int res = 0;
        int diff = depth[u] - depth[v];
        for(int i = 0; i < m; i++) {
            if(diff & (1 << i)) { 
                res = max(res, weight[u][i]);
                u = dp[u][i]; 
            }
        }
        if(u == v) return res;
        for(int i = m - 1; i >= 0; --i) {
            if(dp[u][i] != dp[v][i]) {
                res = max({res, weight[u][i], weight[v][i]});
                u = dp[u][i];
                v = dp[v][i];
            }
        }
        return max({res, weight[u][0], weight[v][0]});
    }

    int dist(int u, int v) {
        int a = lca(u, v);
        return depth[u] + depth[v] - 2 * depth[a];
    }
	
	ll dist_by_weight(int u, int v) {
        int a = lca(u, v);
        return depth_by_weight[u] + depth_by_weight[v] - 2 * depth_by_weight[a];
    }

	int kth_ancestor(int u, ll k) {
        if(u < 0 || k > depth[u]) return -1;
        for(int i = 0; i < m && u != -1; ++i) {
            if(k & (1LL << i)) {
                u = (u >= 0 ? dp[u][i] : -1);
            }
        }
        return u;
    }

    int kth_ancestor_on_path(int u, int v, ll k) {
        int d = dist(u, v);
        if(k >= d) return v;
        int w  = lca(u, v);
        int du = depth[u] - depth[w];
        if(k <= du) return kth_ancestor(u, k);
        int rem = k - du;
        int dv  = depth[v] - depth[w];
        return kth_ancestor(v, dv - rem);
    }

    int kth_downward(int v, ll k) {
        if(k < 1 || k > depth[v] + 1) return -1;
        ll steps_up = depth[v] - (k - 1);
        return kth_ancestor(v, steps_up);
    }

    int max_intersection(int a, int b, int c) { // # of common intersection between path(a, c) OR path(b, c)
        auto cal = [&](int u, int v, int goal){
            return (dist(u, goal) + dist(v, goal) - dist(u, v)) / 2 + 1;
        };
        int res = 0;
        res = max(res, cal(a, b, c));
        res = max(res, cal(a, c, b));
        res = max(res, cal(b, c, a));
        return res;
    }
	
	int intersection(int a, int b, int c, int d) { // common edges between path[a, b] OR path[c, d]
        vi arr = {a, b, c, d, lca(a, b), lca(a, c), lca(a, d), lca(b, c), lca(b, d), lca(c, d)};
        srtU(arr);
        vi s;
        int res = 0;
        for(auto& x : arr) {
            if(dist(x, a) + dist(x, b) == dist(a, b) && dist(c, x) + dist(x, d) == dist(c, d)) {
                s.pb(x);
                for(auto& y : s) {
                    res = max(res, dist(x, y)); // +1 if looking for maximum node
                }
            }
        }
        return res;
    }

    bool is_continuous_chain(int a, int b, int c, int d) { // determine if path[a, b][b, c][c, d] don't have any intersection
        return dist(a, b) <= dist(a, c) && dist(d, c) <= dist(d, b) && intersection(a, b, c, d) == 0;
    }

    int rooted_lca(int a, int b, int c) { return lca(a, c) ^ lca(a, b) ^ lca(b, c); } 

    int next_on_path(int u, int v) { // closest_next_node from u to v
        if(u == v) return -1;
        if(is_ancestor(u, v)) return kth_ancestor(v, depth[v] - depth[u] - 1);
        return parent[u];
    }

    void reroot(int root) {
        fill(all(parent), -1);
        timer = 0;
        dfs(root);
        init();
        cur_lab = 1;
        sv_dfs1(root);
        ascendant[root] = in_label[root];
        sv_dfs2(root);
    }

    int comp_size(int c,int v){
        if(parent[v] == c) return subtree[v];
        return n - subtree[c];
    }

    int rooted_lca_potential_node(int a, int b, int c) { // # of nodes where rooted at will make lca(a, b) = c
        if(rooted_lca(a, b, c) != c) return 0;
        int v1 = next_on_path(c, a);
        int v2 = next_on_path(c, b);
        return n - (v1 == -1 ? 0 : comp_size(c, v1)) - (v2 == -1 ? 0 : comp_size(c, v2));
    }
	
	vi get_path(int u, int v) { // get every node in path [u, v]
        vi path1, path2;
        int c = lca(u, v);
        while(u != c) {
            path1.pb(u);
            u = parent[u];
        }
        while(v != c) {
            path2.pb(v);
            v = parent[v];
        }
        rev(path2);
        path1.pb(c);
        path1.insert(end(path1), all(path2));
        return path1;
    }
};

template<class T, typename F = function<T(const T&, const T&)>>
class basic_segtree {
public:
    int n;    
    int size;  
    vt<T> root;
    F func;
    T DEFAULT;  
    
    basic_segtree() {}

    basic_segtree(int n, T DEFAULT, F func = [](const T& a, const T& b) {return a + b;}) : n(n), DEFAULT(DEFAULT), func(func) {
        size = 1;
        while (size < n) size <<= 1;
        root.assign(size << 1, DEFAULT);
    }
    
    void update_at(int idx, T val) {
        if(idx < 0 || idx >= n) return;
        idx += size, root[idx] = val;
        for(idx >>= 1; idx > 0; idx >>= 1) root[idx] = func(root[idx << 1], root[idx << 1 | 1]);
    }
    
    T queries_range(int l, int r) {
        l = max(0, l), r = min(r, n - 1);
        T res_left = DEFAULT, res_right = DEFAULT;
        l += size, r += size;
        while(l <= r) {
            if((l & 1) == 1) res_left = func(res_left, root[l++]);
            if((r & 1) == 0) res_right = func(root[r--], res_right);
            l >>= 1; r >>= 1;
        }
        return func(res_left, res_right);
    }
	
	T queries_at(int idx) {
        if(idx < 0 || idx >= n) return DEFAULT;
        return root[idx + size];
    }

    T get() {
        return root[1];
    }

    template<typename Pred>
    int max_right(int start, Pred P) const {
        if(start < 0) start = 0;
        if(start >= n) return n;
        T sm = DEFAULT;
        int idx = start + size;
        do {
            while((idx & 1) == 0) idx >>= 1;
            if(!P(func(sm, root[idx]))) {
                while(idx < size) {
                    idx <<= 1;
                    T cand = func(sm, root[idx]);
                    if(P(cand)) {
                        sm = cand;
                        idx++;
                    }
                }
                return idx - size - 1;
            }
            sm = func(sm, root[idx]);
            idx++;
        } while((idx & -idx) != idx);
        return n - 1;
    }

    template<typename Pred>
    int min_left(int ending, Pred P) const {
        if(ending < 0) return 0;
        if(ending >= n) ending = n - 1;
        T sm = DEFAULT;
        int idx = ending + size + 1;
        do {
            idx--;
            while(idx > 1 && (idx & 1)) idx >>= 1;
            if(!P(func(root[idx], sm))) {
                while(idx < size) {
                    idx = idx * 2 + 1;
                    T cand = func(root[idx], sm);
                    if(P(cand)) {
                        sm = cand;
                        idx--;
                    }
                }
                return idx + 1 - size;
            }
            sm = func(root[idx], sm);
        } while((idx & -idx) != idx);
        return 0;
    }
};
int N;

template<class T, typename TT = int, typename F = function<T(const T&, const T&)>>
class HLD {
    private:
	T path_queries_helper(int node, int par) { // only query up to parent, don't include parent info
        T res = DEFAULT;
        while(node != par && node != -1) {   
            if(g.depth[tp[node]] > g.depth[par]) {   
                T t = seg.queries_range(id[tp[node]], id[node]);
                res = func(t, res);
                node = parent[tp[node]];
            } else {   
                T t = seg.queries_range(id[par] + 1, id[node]);
                res = func(t, res);
                break;  
            } 
        }   
        return res; 
    }
    public:
    basic_segtree<T> seg;
    vi id, tp, sz, parent, chain_id;
    int chain_cnt;
    int ct;
    vt<vt<T>> graph;
    int n;
    GRAPH<TT> g;
    T DEFAULT;
    F func;
    HLD() {}

    HLD(vt<vt<TT>>& graph, vi a, F func, int root = 0, T DEFAULT = 0) : seg(graph.size(), DEFAULT, func), g(graph, root), n(graph.size()), func(func), DEFAULT(DEFAULT) {
        this->parent = move(g.parent);
        this->sz = move(g.subtree);
        chain_cnt = 0;
        chain_id.rsz(n);
        ct = 0;
        id.rsz(n), tp.rsz(n);
        dfs(graph, root, -1, root);
        for(int i = 0; i < n; i++) seg.update_at(id[i], a[i]);
    }
        
    void dfs(const vt<vt<TT>>& graph, int node = 0, int par = -1, int top = 0) {   
        id[node] = ct++;    
        tp[node] = top;
        if(node == top) chain_id[node] = chain_cnt++;
        else chain_id[node] = chain_id[top];
        int nxt = -1, max_size = -1;    
        for(auto& nei : graph[node]) {   
            if(nei == par) continue;    
            if(sz[nei] > max_size) {   
                max_size = sz[nei]; 
                nxt = nei;  
            }   
        }   
        if(nxt == -1) return;   
        dfs(graph, nxt, node, top);   
        for(auto& nei : graph[node]) {   
            if(nei != par && nei != nxt) dfs(graph, nei, node, nei);  
        }   
    }

    void update_at(int i, T v) {
        seg.update_at(id[i], v);
    }

	T path_queries(int u, int v) { // remember to include the info of parents
        int c = g.lca(u, v);
        T res = func(seg.queries_at(id[c]), func(path_queries_helper(u, c), path_queries_helper(v, c)));
        if(u == c || v == c) return res;
        if(c >= N) {
            int p = parent[c];
            res = min(res, seg.queries_at(id[p]));
        }
        return res;
    }
};

struct block_cut_tree {
    int n, m, timer; 
    vvpii graph;
    vi tin, low, id, is_art;
    vvi comps, comp_vertices;
    vpii edges;
    vvi tree;
    vi is_simple_cycle;
    stack<int> s;
    block_cut_tree(int n, const vpii& edges) : edges(edges), n(n), m(edges.size()) {
        tin.rsz(n), low.rsz(n), id.rsz(n, -1), is_art.rsz(n), graph.rsz(n);
        for(int i = 0; i < m; i++) {
            auto& [u, v] = edges[i];
            graph[u].pb({v, i});
            graph[v].pb({u, i});
        }
        for(int i = 0; i < n; i++) {
            if(!tin[i]) dfs(i, -1);
        }
        buildBCT();
    }

    void dfs(int node, int prev_id) {
        tin[node] = low[node] = ++timer;
        int child = 0;
        for(auto& [nei, j] : graph[node]) {
            if(j == prev_id) continue;
            if(!tin[nei]) {
                child++;
                s.push(j);
                dfs(nei, j);
                low[node] = min(low[node], low[nei]);
                if((prev_id == -1 && child > 1) || (prev_id != -1 && low[nei] >= tin[node])) is_art[node] = true;
                if(low[nei] >= tin[node]) {
                    comps.pb({});
                    auto& curr = comps.back();
                    while(true) {
                        int e = s.top(); s.pop();
                        curr.pb(e);
                        if(e == j) break;
                    }
                }
            } else if(tin[nei] < tin[node]) {
                s.push(j);
                low[node] = min(low[node], tin[nei]);
            }
        }
    }

    void buildBCT() {
        // https://codeforces.com/contest/487/problem/E
        // build a bipartile graph city -> supernode -> city -> supernode ...
        int C = comps.size();
        int N = n + C;
        tree.assign(N, {});
        for(int i = 0; i < C; i++) {
            int compNode = n + i;
            auto& comp = comps[i];
            vi verts;
            for(int eid : comp) {
                auto &[u, v] = edges[eid];
                verts.pb(u);
                verts.pb(v);
            }
            srtU(verts);
            for(int u : verts) {
                tree[compNode].pb(u);
                tree[u].pb(compNode);
            }
        }
    }

    void build() {
        int B = comps.size(); 
        is_simple_cycle.rsz(m);
        comp_vertices = vvi(B);
        int c = 0;
        for(int i = 0; i < n; i++) {
            if(is_art[i]) {
                id[i] = c++;
            }
        }
        int nc = c;
        for(int i = 0; i < B; i++) {
            int bn = nc++;
            for(auto& j : comps[i]) {
                auto& [u, v] = edges[j];
                comp_vertices[i].pb(u);
                comp_vertices[i].pb(v);
            }
            srtU(comp_vertices[i]);
            if(comps[i].size() == comp_vertices[i].size()) {
                for(auto& j : comps[i]) {
                    is_simple_cycle[j] = true;
                }
            }
            for(auto& j : comp_vertices[i]) {
                if(id[j] == -1) id[j] = bn; 
            }
        }
        for(int i = 0; i < n; i++) {
            if(id[i] == -1) {
                comp_vertices.pb({i});
                id[i] = nc++;
            }
        }
        tree.rsz(nc);
        for(int i = 0; i < (int)comp_vertices.size(); i++) {
            int bn = c++;
            for(auto& u : comp_vertices[i]) {
                if(is_art[u]) {
                    int j = id[u];
                    tree[j].pb(bn);
                    tree[bn].pb(j);
                } 
            }
        }
    }
};

void solve() {
    int n, m, q; cin >> n >> m >> q;
    N = n;
    vi a(n); cin >> a;
    vpii edges(m); cin >> edges;
    for(auto& [u, v] : edges) {
        u--, v--;
    }
    block_cut_tree bct(n, edges);
    auto& tree = bct.tree;
    const int N = tree.size();
    vi par(N, -1);
    auto dfs = [&](auto& dfs, int node = 0, int p = -1) -> void {
        for(auto& nei : tree[node]) {
            if(nei == p) continue;
            par[nei] = node;
            dfs(dfs, nei, node);
        }
    }; dfs(dfs);
    multiset<int> s[N];
    vi W(N);
    int C = bct.comps.size();
    for(int i = 0; i < C; ++i) {
        int compNode = n + i;
        vi verts;
        for(int eid: bct.comps[i]){
            auto [u,v] = edges[eid];
            verts.pb(u);
            verts.pb(v);
        }
        srtU(verts);
        for(int u: verts) {
            if(par[compNode] != u) {
                s[compNode].insert(a[u]);
            }
        }
    }
    for(int i = 0; i < N; i++) {
        W[i] = i < n ? a[i] : *s[i].begin();
    }
    HLD<int> G(tree, W, [](const int& a, const int& b) {return min(a, b);}, 0, inf);
    while(q--) {
        char op; cin >> op;
        if(op == 'C') {
            int u, w; cin >> u >> w;
            u--;
            int p = par[u];
            if(p != -1) {
                s[p].erase(s[p].find(a[u]));
                s[p].insert(w);
                G.update_at(p, *s[p].begin());
            }
            a[u] = w;
            G.update_at(u, a[u]);
        } else {
            int u, v; cin >> u >> v;
            u--, v--;
            cout << (u == v ? a[u] : G.path_queries(u, v)) << '\n';
        }
    }
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
