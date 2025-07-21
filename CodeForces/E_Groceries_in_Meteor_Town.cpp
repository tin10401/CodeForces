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
ll have_bit(ll x, int b) { return x & (1LL << b); }
int min_bit(ll x) { return __builtin_ctzll(x); }
int max_bit(ll x) { return 63 - __builtin_clzll(x); } 
const vvi dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}; // UP, DOWN, LEFT, RIGHT
const vvi knight_dirs = {{-2, -1}, {-2,  1}, {-1, -2}, {-1,  2}, {1, -2}, {1,  2}, {2, -1}, {2,  1}}; // knight dirs
const vc dirChar = {'U', 'D', 'L', 'R'};
int modExpo(ll base, ll exp, ll mod) { ll res = 1; base %= mod; while(exp) { if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>= 1; } return res; }
ll extended_gcd(ll a, ll b, ll &x, ll &y) { if (b == 0) { x = 1; y = 0; return a; } ll d = extended_gcd(b, a % b, y, x); y -= (a / b) * x; return d; }
ll modInv(ll a, ll m) { ll x, y; ll g = extended_gcd(a, m, x, y); if (g != 1) { return -1; } x %= m; if (x < 0) x += m; return x; }
int modExpo_on_string(ll a, string exp, int mod) { ll b = 0; for(auto& ch : exp) b = (b * 10 + (ch - '0')) % (mod - 1); return modExpo(a, b, mod); }
ll sum_even_series(ll n) { return (n / 2) * (n / 2 + 1);} 
ll sum_odd_series(ll n) {return n - sum_even_series(n);} // sum of first n odd number is n ^ 2
ll sum_of_square(ll n) { return n * (n + 1) * (2 * n + 1) / 6; } // sum of 1 + 2 * 2 + 3 * 3 + 4 * 4 + ... + n * n
string make_lower(const string& t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return tolower(c); }); return s; }
string make_upper(const string&t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return toupper(c); }); return s; }
ll sqrt(ll n) { ll t = sqrtl(n); while(t * t < n) t++; while(t * t > n) t--; return t;}
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
        int r1 = lca(a, b), r2 = lca(c, d);
        int q = depth[r1] > depth[r2] ? r1 : r2;
        int p = lca(a, c), t = lca(a, d);
        if (depth[t] > depth[p]) p = t;
        t = lca(b,c); if (depth[t] > depth[p]) p = t;
        t = lca(b,d); if (depth[t] > depth[p]) p = t;
        if (depth[p] < depth[q]) return 0;
        return depth[p] - depth[q];
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


template<typename T, typename I = ll, typename II = ll, typename F = function<T(const T, const T)>, typename G = function<void(int i, int left, int right, I)>>
class SGT { 
    public: 
    int n;  
    vt<T> root;
	vt<II> lazy;
    T DEFAULT;
    F func;
    G apply_func;
	SGT(int n, T DEFAULT, F func, G apply_func = [](int i, int left, int right, I val){}) : func(func), apply_func(apply_func) {    
        this->n = n;
        this->DEFAULT = DEFAULT;
		int k = 1;
        while(k < n) k <<= 1; 
        root.rsz(k << 1, DEFAULT);    
        lazy.rsz(k << 1, -INF);
    }
    
    void update_at(int id, T val) {  
        update_at(entireTree, id, val);
    }
    
    void update_at(iter, int id, T val) {  
		pushDown;
        if(left == right) { 
            root[i] = val;  
            return;
        }
        int middle = midPoint;  
        if(id <= middle) update_at(lp, id, val);   
        else update_at(rp, id, val);   
        root[i] = func(root[lc], root[rc]);
    }

    void update_range(int start, int end, I val) { 
        update_range(entireTree, start, end, val);
    }
    
    void update_range(iter, int start, int end, I val) {    
        pushDown;   
        if(left > end || start > right) return; 
        if(left >= start && right <= end) { 
			apply(i, left, right, val);
            pushDown;   
            return;
        }
        int middle = midPoint;  
        update_range(lp, start, end, val);    
        update_range(rp, start, end, val);    
        root[i] = func(root[lc], root[rc]);
    }

	void apply(iter, I val) {
        root[i].modify(val);
        lazy[i] = val;
    }

    void push(iter) {   
        if(lazy[i] != -INF && left != right) {
			int middle = midPoint;
            apply(lp, lazy[i]), apply(rp, lazy[i]);
            lazy[i] = -INF;
        }
    }

	T queries_at(int id) {
		return queries_at(entireTree, id);
	}
	
	T queries_at(iter, int id) {
		pushDown;
		if(left == right) {
			return root[i];
		}
		int middle = midPoint;
		if(id <= middle) return queries_at(lp, id);
		return queries_at(rp, id);
	}

    T queries_range(int start, int end) { 
        return queries_range(entireTree, start, end);
    }
    
    T queries_range(iter, int start, int end) {   
        pushDown;
        if(left > end || start > right) return DEFAULT;
        if(left >= start && right <= end) return root[i];   
        int middle = midPoint;  
        return func(queries_range(lp, start, end), queries_range(rp, start, end));
    }
	
	T get() {
		return root[0];
	}
	
	template<typename Pred> // seg.min_left(ending, [](const int& a) {return a > 0;});
        int min_left(int ending, Pred f) { // min index where f[l, ending] is true
            T a = DEFAULT;
            return find_left(entireTree, ending, f, a);
        }

    template<typename Pred>
        int max_right(int starting, Pred f) {
            T a = DEFAULT;
            return find_right(entireTree, starting, f, a);
        }

    template<typename Pred>
        int find_left(iter, int end, Pred f, T& now) {
            pushDown;
            if(left > end) return -2;
            if(right <= end && f(func(root[i], now))) {
                now = func(root[i], now);
                return left;
            }
            if(left == right) return -1;
            int middle = midPoint;
            int r = find_left(rp, end, f, now);
            if(r == -2) return find_left(lp, end, f, now);
            if(r == middle + 1) {
                int l = find_left(lp, end, f, now);
                if(l != -1) return l;
            }
            return r;
        }

    template<typename Pred>
        int find_right(iter, int start, Pred f, T &now) {
            pushDown;
            if(right < start) return -2;
            if(left >= start && f(func(now, root[i]))) {
                now = func(now, root[i]);
                return right;
            }
            if(left == right) return -1;
            int middle = midPoint;
            int l = find_right(lp, start, f, now);
            if(l == -2) return find_right(rp, start, f, now);
            if(l == middle) {
                int r = find_right(rp, start, f, now);
                if(r != -1) return r;
            }
            return l;
        }
};

struct info {
    int mn, mx, real_mn, real_mx;
    info(int x = -inf) : mn(x), mx(x), real_mn(inf), real_mx(-inf) {}

    void modify(int v) {
        if(v == 0) {
            real_mn = inf;
            real_mx = -inf;
        } else {
            real_mn = mn;
            real_mx = mx;
        }
    }

    friend info operator+(const info& a, const info& b) {
        if(a.mn == -inf) return b;
        if(b.mn == -inf) return a;
        info res;
        res.mx = max(a.mx, b.mx);
        res.mn = min(a.mn, b.mn);
        res.real_mn = min(a.real_mn, b.real_mn);
        res.real_mx = max(a.real_mx, b.real_mx);
        return res;
    }
};

struct Reachability_Tree {
    struct DSU {
        vi p, r;
        DSU(int n): p(n), r(n, 0) { iota(all(p), 0); }
        int find(int x) {
            return p[x] == x ? x : p[x] = find(p[x]);
        }
        bool same(int a, int b) {
            return find(a) == find(b);
        }
        void merge(int a, int b) {
            a = find(a); b = find(b);
            if(a == b) return;
            p[b] = a;
            if(r[a] == r[b]) r[a]++;
        }
    };
    int n;
    vi weight;
    GRAPH<int> g;
    Reachability_Tree(int n, var(3)& edges) : n(n) {
        build(edges);
    } 

    void build(var(3)& edges) {
        vvi graph(n * 2);
        weight.rsz(n * 2);
        sort(all(edges), [](const ar(3)& a, const ar(3)& b) {return a[2] < b[2];});
        DSU root(n * 2);
        int rt = n - 1;
        for(auto& [u, v, w] : edges) {
            if(!root.same(u, v)) {
                weight[++rt] = w;
                graph[rt].pb(root.find(u));
                graph[rt].pb(root.find(v));
                root.merge(rt, u);
                root.merge(rt, v);
            }
        }
        g = GRAPH<int>(graph, rt);
    }

    int lca_query(int a, int b, int c) {
        if(a < 0 || a > 2 * n)  return -1;
        a = g.ord[a];
        b = g.ord[b];
        if(a == b && b == c) return -1;
        int l = g.lca(g.lca(a, b), c);
        return weight[l];
    }

    int get(int c) {
        return g.tin[c];
    }
};

void solve() {
    int n, q; cin >> n >> q;
    var(3) edges;
    for(int i = 1; i < n; i++) {
        int u, v, w; cin >> u >> v >> w;
        u--, v--;
        edges.pb({u, v, w});
    }
    Reachability_Tree Tree(n, edges);
    SGT<info> root(n, info(), [](const info& a, const info& b) {return a + b;});
    for(int i = 0; i < n; i++) {
        root.update_at(i, info(Tree.get(i)));
    }
    while(q--) {
        int op; cin >> op;
        if(op == 3) {
            int x; cin >> x;
            x--;
            auto it = root.get();
            cout << Tree.lca_query(it.real_mn, it.real_mx, x) << '\n';
            continue;
        } 
        int l, r; cin >> l >> r;
        l--, r--;
        root.update_range(l, r, op == 1);
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
