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
template<typename T> T geometric_sum(ll n, ll k) { return (1 - T(n).pow(k + 1)) / (1 - n); } // return n^1 + n^2 + n^3 + n^4 + n^5 + ... + n^k
template<typename T> T geometric_power(ll p, ll k) { return (T(p).pow(k + 1) - 1) / T(p - 1); } // p^1 + p^2 + p^3 + ... + p^k
bool is_perm(ll sm, ll square_sum, ll len) {return sm == len * (len + 1) / 2 && square_sum == len * (len + 1) * (2 * len + 1) / 6;} // determine if an array is a permutation base on sum and square_sum
bool is_vowel(char c) {return c == 'a' || c == 'e' || c == 'u' || c == 'o' || c == 'i';}

template <int MOD>
struct mod_int {
    int value;
    
    mod_int(ll v = 0) { value = int(v % MOD); if (value < 0) value += MOD; }
    
    mod_int& operator+=(const mod_int &other) { value += other.value; if (value >= MOD) value -= MOD; return *this; }
    mod_int& operator-=(const mod_int &other) { value -= other.value; if (value < 0) value += MOD; return *this; }
    mod_int& operator*=(const mod_int &other) { value = int((ll)value * other.value % MOD); return *this; }
    mod_int pow(ll p) const { mod_int ans(1), a(*this); while (p) { if (p & 1) ans *= a; a *= a; p /= 2; } return ans; }
    
    mod_int inv() const { return pow(MOD - 2); }
    mod_int& operator/=(const mod_int &other) { return *this *= other.inv(); }
    
    friend mod_int operator+(mod_int a, const mod_int &b) { a += b; return a; }
    friend mod_int operator-(mod_int a, const mod_int &b) { a -= b; return a; }
    friend mod_int operator*(mod_int a, const mod_int &b) { a *= b; return a; }
    friend mod_int operator/(mod_int a, const mod_int &b) { a /= b; return a; }
    
    bool operator==(const mod_int &other) const { return value == other.value; }
    bool operator!=(const mod_int &other) const { return value != other.value; }
    bool operator<(const mod_int &other) const { return value < other.value; }
    bool operator>(const mod_int &other) const { return value > other.value; }
    bool operator<=(const mod_int &other) const { return value <= other.value; }
    bool operator>=(const mod_int &other) const { return value >= other.value; }
    
    mod_int operator&(const mod_int &other) const { return mod_int((ll)value & other.value); }
    mod_int& operator&=(const mod_int &other) { value &= other.value; return *this; }
    mod_int operator|(const mod_int &other) const { return mod_int((ll)value | other.value); }
    mod_int& operator|=(const mod_int &other) { value |= other.value; return *this; }
    mod_int operator^(const mod_int &other) const { return mod_int((ll)value ^ other.value); }
    mod_int& operator^=(const mod_int &other) { value ^= other.value; return *this; }
    mod_int operator<<(int shift) const { return mod_int(((ll)value << shift) % MOD); }
    mod_int& operator<<=(int shift) { value = int(((ll)value << shift) % MOD); return *this; }
    mod_int operator>>(int shift) const { return mod_int(value >> shift); }
    mod_int& operator>>=(int shift) { value >>= shift; return *this; }

    mod_int& operator++() { ++value; if (value >= MOD) value = 0; return *this; }
    mod_int operator++(int) { mod_int temp = *this; ++(*this); return temp; }
    mod_int& operator--() { if (value == 0) value = MOD - 1; else --value; return *this; }
    mod_int operator--(int) { mod_int temp = *this; --(*this); return temp; }

    explicit operator ll() const { return (ll)value; }
    explicit operator int() const { return value; }
    explicit operator db() const { return (db)value; }

    friend mod_int operator-(const mod_int &a) { return mod_int(0) - a; }
    friend ostream& operator<<(ostream &os, const mod_int &a) { os << a.value; return os; }
    friend istream& operator>>(istream &is, mod_int &a) { ll v; is >> v; a = mod_int(v); return is; }
};

const static int MOD = 1e9 + 7;
using mint = mod_int<MOD>;
using vmint = vt<mint>;
using vvmint = vt<vmint>;
using vvvmint = vt<vvmint>;
using pmm = pair<mint, mint>;
using vpmm = vt<pmm>;

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

template<typename T = int>
struct virtual_tree {
    GRAPH<T> g;
    using info = pair<int, ll>;
    vt<vt<info>> graph; // [node, dist]
    bool dist_by_weight;
    vi subtree, importance;
    int total;
    ll ans = 0;
    virtual_tree(const vt<vt<T>>& _graph, bool _dist_by_weight = false) : g(_graph), graph(_graph.size()), dist_by_weight(_dist_by_weight), subtree(_graph.size()), importance(_graph.size()) {}

    int build(vi& vertices) {
        int n = vertices.size();
        auto cmp = [&](const int& a, const int& b) -> bool {
            return g.tin[a] < g.tin[b];
        };
        sort(all(vertices), cmp);
        auto a(vertices);
        for(int i = 0; i < n - 1; i++) {
            int u = vertices[i], v = vertices[i + 1];
            a.pb(g.lca(u, v));
        }
        sort(all(a), cmp);
        a.erase(unique(all(a)), end(a));
        total = vertices.size();
        for(auto& u : a) {
            vt<info>().swap(graph[u]);
            subtree[u] = 0; 
            importance[u] = false;
        }
        for(auto& u : vertices) {
            importance[u] = true;
        }
        vi s;
        s.pb(a[0]);
        for(int i = 1; i < (int)a.size(); i++) {
            int u = a[i];
            while(!s.empty() && !g.is_ancestor(s.back(), u)) s.pop_back();
            int p = s.back();
            ll d = dist_by_weight ? g.dist_by_weight(p, u) : g.dist(p, u);
            graph[p].pb({u, d});
            graph[u].pb({p, d});
            s.pb(u);
        }
        return s[0];
    }

    vmint dfs(int u, int par, int m, int C, vmint &dp0, bool ok = false) {
        bool mark = (par != -1 && importance[u] || ok);
        if(mark) {
            auto old(dp0);
            int M = old.size();
            int N = min(m, M);
            dp0.assign(N + 1, 0);
            for(int i = 0; i < M; i++) {
                if(i + 1 <= N) dp0[i + 1] += old[i];
                int freeGroups = i - C;
                if(freeGroups > 0) {
                    dp0[i] += old[i] * freeGroups;
                }
            }
        }
        for(auto &[v, _] : graph[u]) {
            if(v == par) continue;
            dfs(v, u, m, C + mark, dp0);
        }
        return dp0;
    }
};

void solve() {
    int n, q; cin >> n >> q;
    vvi graph(n);
    for(int i = 1; i < n; i++) {
        int u, v; cin >> u >> v;
        u--, v--;
        graph[u].pb(v);
        graph[v].pb(u);
    }
    virtual_tree<int> g(graph);
    while(q--) {
        int k, m, r; cin >> k >> m >> r;
        vi a(k); cin >> a;
        bool ok = false;
        for(auto& x : a) x--;
        r--;
        for(auto& x : a) {
            if(x == r) ok = true;
        }
        a.pb(r);
        srtU(a);
        g.build(a);
        int C = 0;
        vmint dp0(1, mint(1));          
        auto res = g.dfs(r, -1, m, 0, dp0, ok);
        mint ans = 0;
        for (int i = 1; i < (int)res.size() && i <= m; i++) ans += res[i];
        cout << ans << "\n";
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
