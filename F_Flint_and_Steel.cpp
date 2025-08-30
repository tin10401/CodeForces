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
#define vll vt<ll>  
#define vvll vt<vll>
#define pll pair<ll, ll>    
#define vpll vt<pll>
#define vvpll vt<vpll>
#define vi vt<int>
#define vvi vt<vi>
#define vvvi vt<vvi>
#define pii pair<int, int>
#define vpii vt<pii>
#define vs vt<string>
#define vvpii vt<vpii>
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
#define MP make_pair
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
template <class K, class V> using umap = std::unordered_map<K, V, custom>; 
template <class K> using uset = std::unordered_set<K, custom>;
template<class K, class V = int> using gpt = gp_hash_table<K, V, custom>;
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
template<typename T, typename Compare>
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
vs decode(const string& s, char off = ' ') {
    vs a;
    string t;
    for(auto& ch : s) {
        if(ch == off) {
            if(!t.empty()) a.pb(t);
            t = "";
        } else {
            t += ch;
        }
    }
    if(!t.empty()) a.pb(t);
    return a;
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
}
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

#define M_PI 3.14159265358979323846
const static string pi = "3141592653589793238462643383279";
const static ll INF = 1e18;
const static int inf = 1e9 + 100;
const static int MX = 1e5 + 5;
ll gcd(ll a, ll b) { while (b != 0) { ll temp = b; b = a % b; a = temp; } return a; }
ll lcm(ll a, ll b) { return (a / gcd(a, b)) * b; }
ll floor(ll a, ll b) { if(b < 0) a = -a, b = -b; if (a >= 0) return a / b; return a / b - (a % b ? 1 : 0); }
ll ceil(ll a, ll b) { if (b < 0) a = -a, b = -b; if (a >= 0) return (a + b - 1) / b; return a / b; }
int pct(ll x) { return __builtin_popcountll(x); }
int have_bit(ll x, int b) { return (x >> b) & 1; }
int min_bit(ll x) { return __builtin_ctzll(x); }
int max_bit(ll x) { return 63 - __builtin_clzll(x); } 
const vvi dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}; // UP, DOWN, LEFT, RIGHT
const vvi knight_dirs = {{-2, -1}, {-2,  1}, {-1, -2}, {-1,  2}, {1, -2}, {1,  2}, {2, -1}, {2,  1}}; // knight dirs
const string dirChar = {'U', 'D', 'L', 'R'};
int modExpo(ll base, ll exp, ll mod) { ll res = 1; base %= mod; while(exp) { if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>= 1; } return res; }
ll extended_gcd(ll a, ll b, ll &x, ll &y) { if (b == 0) { x = 1; y = 0; return a; } ll d = extended_gcd(b, a % b, y, x); y -= (a / b) * x; return d; }
int modExpo_on_string(ll a, string exp, int mod) { ll b = 0; for(auto& ch : exp) b = (b * 10 + (ch - '0')) % (mod - 1); return modExpo(a, b, mod); }
ll sum_even_series(ll n) { return (n / 2) * (n / 2 + 1);} 
ll sum_odd_series(ll n) { ll m = (n + 1) / 2; return m * m; }
ll sum_of_square(ll n) { return n * (n + 1) * (2 * n + 1) / 6; } // sum of 1 + 2 * 2 + 3 * 3 + 4 * 4 + ... + n * n
string make_lower(const string& t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return tolower(c); }); return s; }
string make_upper(const string&t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return toupper(c); }); return s; }
template<typename T> T geometric_sum(ll n, ll k) { return (1 - T(n).pow(k + 1)) / (1 - n); } // return n^1 + n^2 + n^3 + n^4 + n^5 + ... + n^k
template<typename T> T geometric_power(ll p, ll k) { return (T(p).pow(k + 1) - 1) / T(p - 1); } // p^1 + p^2 + p^3 + ... + p^k
bool is_perm(ll sm, ll square_sum, ll len) {return sm == len * (len + 1) / 2 && square_sum == len * (len + 1) * (2 * len + 1) / 6;} // determine if an array is a permutation base on sum and square_sum
//bool is_two_prime_sum(ll n) { return n >= 4 && (n % 2 == 0 || isPrime(n - 2)); }
//bool is_three_prime_sum(ll n) { return n >= 6 && (n % 2 || is_two_prime_sum(n - 2)); }
ll sqrt(ll n) { ll t = sqrtl(n); while(t * t < n) t++; while(t * t > n) t--; return t;}

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
        idx += size, root[idx] = func(root[idx], val);
        for(idx >>= 1; idx > 0; idx >>= 1) root[idx] = func(root[idx << 1], root[idx << 1 | 1]);
    }
    
	T queries_range(int l, int r) {
        l = max(0, l), r = min(r, n - 1);
        T res_left = DEFAULT, res_right = DEFAULT;
        l += size, r += size;
        bool has_left = false, has_right = false;
        while(l <= r) {
            if((l & 1) == 1) {
                if(!has_left) res_left = root[l++];
                else res_left = func(res_left, root[l++]); 
                has_left = true;
            }
            if((r & 1) == 0) {
                if(!has_right) res_right = root[r--];
                else res_right = func(root[r--], res_right);
                has_right = true;
            }
            l >>= 1; r >>= 1;
        }
        if(!has_left) return res_right;
        if(!has_right) return res_left;
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

template<class T>
struct PSGT {
    struct Node {
        int l, r;
        T key;
        Node(T key) : key(key), l(0), r(0) {}
    };
    int new_node(int prev) {
        F.pb(F[prev]);
        return F.size() - 1;
    }

    int new_node() {
        F.pb(0);
        return F.size() - 1;
    }
    vt<Node> F;
    vi t;
    int n;
    T DEFAULT;
    PSGT(int n, T DEFAULT) : n(n), DEFAULT(DEFAULT), t(n) {
        F.reserve(n * 20);
        F.pb(Node(DEFAULT));
    }

	int update(int prev, int id, T delta, int left, int right) {  
        int curr = new_node(prev);
        if(left == right) { 
            F[curr].key = merge(F[curr].key, delta);
            return curr;
        }
        int middle = midPoint;
        if(id <= middle) F[curr].l = update(F[prev].l, id, delta, left, middle);
        else F[curr].r = update(F[prev].r, id, delta, middle + 1, right);
        F[curr].key = merge(F[F[curr].l].key, F[F[curr].r].key);
        return curr;
    }

	T queries_at(int curr, int start, int end, int left, int right) { 
        if(!curr || left > end || start > right) return DEFAULT;
        if(left >= start && right <= end) return F[curr].key;
        int middle = midPoint;  
		return merge(queries_at(F[curr].l, start, end, left, middle), queries_at(F[curr].r, start, end, middle + 1, right));
    };

    void update_at(int i, int id, T delta) { 
        t[i] = update(t[i], id, delta, 0, n - 1);
    }

    T queries_at(int i, int start, int end) {
        return queries_at(t[i], start, end, 0, n - 1);
    }

    T merge(T left, T right) {
        return min(left, right);
    }
};

void solve() {
    int n; cin >> n;
    vi e(n + 1);
    for(int i = 1; i <= n; i++) {
        cin >> e[i];
    }
    vi L(n + 1), R(n + 1);
    for(int i = 1; i <= n; i++) {
        L[i] = max(1, i - e[i] + 1);
        R[i] = min(n, i + e[i] - 1);
    }
    basic_segtree<pii> dp_tree(n + 1, {inf, 0}, [](const auto& a, const auto& b) {return min(a, b);});
    PSGT<pii> pst(n + 1, {inf, 0});
    vi dp(n + 1, inf);
    vi par(n + 1);
    for(int i = 1; i <= n; i++) {
        pst.t[i] = pst.t[i - 1];
        if(e[i] > 0 && L[i] <= 1) {
            dp[i] = 1, par[i] = 0;
            dp_tree.update_at(R[i], {1, i});
            pst.update_at(i, R[i], {1, i});
            continue;
        }
        if(e[i] == 0) continue;
        auto ans = min(pst.queries_at(L[i] - 1, L[i] - 1, R[i]), dp_tree.queries_range(L[i] - 1, i - 1));
        if(ans.ff >= inf) continue;
        dp[i] = ++ans.ff;
        par[i] = ans.ss;
        ans.ss = i;
        dp_tree.update_at(R[i], ans);
        pst.update_at(i, R[i], ans);
    }
    int best = 0;
    for(int i = 1; i <= n; i++) {
        if(R[i] >= n && dp[i] < dp[best]) {
            best = i;
        }
    }
    if(dp[best] >= inf) {
        cout << -1 << '\n';
        return;
    }
    min_heap<pii> q;
    while(best) {
        q.push({e[best], best});
        best = par[best];
    }
    cout << q.size() << '\n';
    while(!q.empty()) {
        auto [_, u] = q.top(); q.pop();
        cout << u << (q.empty() ? '\n' : ' ');
    }
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
    endClock
    #ifdef LOCAL
      printMemoryUsage();
    #endif
    return 0;
}
