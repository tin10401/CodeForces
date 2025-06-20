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
bool is_perm(ll sm, ll square_sum, ll len) {return sm == len * (len + 1) / 2 && square_sum == len * (len + 1) * (2 * len + 1) / 6;} // determine if an array is a permutation base on sum and square_sum
bool is_vowel(char c) {return c == 'a' || c == 'e' || c == 'u' || c == 'o' || c == 'i';}

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

	
	void update_range(int l, int r, ll v) {}

    T get() {
        return root[1];
    }

    template<typename Pred>
    int max_right(int start, Pred P) {
        if(start < 0) start = 0;
        if(start >= n) return n - 1;
        T sm = DEFAULT;
        int idx = start + size;
        while((idx & 1) == 0) idx >>= 1;

        do {
            T cand = func(sm, root[idx]);
            if(P(cand)) {
                sm = cand;
                ++idx;
                while((idx & 1) == 0) idx >>= 1;
            } else {
                while(idx < size) {
                    idx <<= 1;
                    cand = func(sm, root[idx]);
                    if(P(cand)) {
                        sm = cand;
                        ++idx;
                    }
                }
                int leaf = idx - size;
                return leaf - 1;
            }
        } while((idx & -idx) != idx);
        return n - 1;
    }

    template<typename Pred>
    int min_left(int ending, Pred P) {
        if(ending < 0) return 0;
        if(ending >= n) ending = n;
        T sm = DEFAULT;
        int idx = ending + size;
        while((idx & 1) == 1) idx >>= 1;
        do {
            T cand = func(root[idx], sm);
            if(P(cand)) {
                sm = cand;
                --idx;
                while((idx & 1) == 1) idx >>= 1;
            } else {
                while(idx < size) {
                    idx = idx * 2 + 1;
                    cand = func(root[idx], sm);
                    if (P(cand)) {
                        sm = cand;
                        --idx;       
                    }
                }
                return (idx + 1) - size;
            }
        } while(idx > 1);
        return 0;
    }
};

struct info {
    ll ans, prefix, suffix, sm;
    info(ll x = -INF) : ans(max(0LL, x)), prefix(max(0LL, x)), suffix(max(0LL, x)), sm(x) {}

    friend info operator+(const info& a, const info& b) {
        if(a.sm == -INF) return b;
        if(b.sm == -INF) return a;
        info res;
        res.ans = max({a.ans, b.ans, a.suffix + b.prefix});
        res.sm = a.sm + b.sm;
        res.prefix = max(a.prefix, a.sm + b.prefix);
        res.suffix = max(b.suffix, b.sm + a.suffix);
        return res;
    }
};

void solve() {
    int n, q; cin >> n >> q;
    vi a(n); cin >> a;
    for(auto& x : a) x--;
    vvar(3) op(n);
    for(int i = 0; i < n; i++) {
        op[a[i]].pb({0, i, 1});
    }
    vpii Q(q + 1);
    vvpii ans(q + 1);
    for(int i = 1; i <= q; i++) {
        auto& [id, x] = Q[i]; cin >> id >> x;
        id--, x--;
        op[a[id]].pb({i, id, -1});
        a[id] = x;
        op[a[id]].pb({i, id, 1});
    }
    for(int i = 0; i < n; i++) {
        op[a[i]].pb({q + 1, i, -1});
    }
    basic_segtree<info> root(n, info());
    for(int i = 0; i < n; i++) {
        root.update_at(i, info(-1));
    }
    for(int v = 0; v < n; v++) {
        int lst = 0;
        for(auto& [t, i, x] : op[v]) {
            auto res = root.get().ans;
            if(lst != t) {
                ans[lst].pb({res / 2, 1});
                if(t != q + 1) {
                    ans[t].pb({res / 2, -1});
                }
            }
            lst = t;
            root.update_at(i, info(x));
        }
    }
    multiset<int> s;
    for(int i = 0; i <= q; i++) {
        for(auto& [x, delta] : ans[i]) {
            if(delta == 1) {
                s.insert(x);
            } else {
                assert(s.count(x));
                s.erase(s.find(x));
            }
        }
        if(i) {
            cout << *s.rbegin() << (i == q ? '\n' : ' ');
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
    cin >> t;
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
