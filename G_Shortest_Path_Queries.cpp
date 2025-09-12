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
const static int MX = 2e5 + 5;

#define M_PI 3.14159265358979323846
const static string pi = "3141592653589793238462643383279";
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
template<typename T> T geometric_power(ll p, ll k) { return (T(p).pow(k + 1) - 1) / T(p - 1); } // p^0 + p^1 + p^2 + p^3 + ... + p^k
template<typename T> T geometric_power_range(T base, ll startExp, ll endExp) { // return base^startExp + base^(startExp + 1) + ... + base^endExp
    if(startExp > endExp) return 0;
    T first = base.pow(startExp);
    ll len = endExp - startExp + 1;
    return first * (base.pow(len) - 1) / (base - 1);
}
bool is_perm(ll sm, ll square_sum, ll len) {return sm == len * (len + 1) / 2 && square_sum == len * (len + 1) * (2 * len + 1) / 6;} // determine if an array is a permutation base on sum and square_sum
//bool is_two_prime_sum(ll n) { return n >= 4 && (n % 2 == 0 || isPrime(n - 2)); }
//bool is_three_prime_sum(ll n) { return n >= 6 && (n % 2 || is_two_prime_sum(n - 2)); }
ll sqrt(ll n) { ll t = sqrtl(n); while(t * t < n) t++; while(t * t > n) t--; return t;}

template<typename T, int BITS = 30>
struct xor_basis {
	// subsequence [l, r] having subsequence_xor of x is pow(2, (r - l + 1) - rank())
    T basis[BITS];
    int r;

    xor_basis() {
        r = 0;
        for(int b = 0; b < BITS; b++)
            basis[b] = 0;
    }

    bool insert(T x) {
        if(x == 0) return false;
        for(int b = BITS - 1; b >= 0; --b) {
            if(!have_bit(x, b)) continue;
            if(!basis[b]) {
                basis[b] = x;
                r++;
                return true;
            }
            x ^= basis[b];
        }
        return false;
    }

    bool contains(T x) const {
        for(int b = BITS - 1; b >= 0; --b) {
            if(!have_bit(x, b)) continue;
            if(!basis[b]) return false;
            x ^= basis[b];
        }
        return true;
    }

    T min_value(T x) const {
        for(int b = BITS - 1; b >= 0; --b) {
            if(basis[b] && (x ^ basis[b]) < x)
                x ^= basis[b];
        }
        return x;
    }

    T max_value(T x = 0) const {
        for (int b = BITS - 1; b >= 0; --b) {
            if (basis[b] && (x ^ basis[b]) > x)
                x ^= basis[b];
        }
        return x;
    }

    int rank() const {
        return r;
    }

    uint64_t size() const {
        return (r >= 64 ? 0ULL : (1ULL << r));
    }

	inline xor_basis operator+(const xor_basis& other) const {
        if (r == 0) return other;
        if (other.r == 0) return *this;
        const xor_basis* big   = (r >= other.r ? this  : &other);
        const xor_basis* small = (r >= other.r ? &other :  this);
        xor_basis res = *big;
        for (int i = 0; i < small->r; ++i) res.insert(small->basis[i]);
        return res;
    }

    bool operator==(const xor_basis &o) const {
        for(int b = 0; b < BITS; b++)
            if(basis[b] != o.basis[b])
                return false;
        return true;
    }
};

vvpii G(MX * 2);
using info = xor_basis<int>;
struct Undo_DSU {
    int n;
    using info = xor_basis<int>;
    struct Record {
        bool type;
        int u, v;
        int rank_u, rank_v;
        int xor_v_to_parent;
        info basis_u;
    };

    vi par, rank, xor_to_parent;
    vt<info> basis;
    stack<Record> st;

    Undo_DSU(int n) : n(n) {
        par.rsz(n);
        rank.rsz(n, 1);
        xor_to_parent.rsz(n, 0);
        basis.rsz(n);
        iota(par.begin(), par.end(), 0);
    }

    int find(int v) {
        if(v == par[v]) return v;
        return find(par[v]);
    }

    int xor_root(int v) {
        if(v == par[v]) return xor_to_parent[v];
        return xor_to_parent[v] ^ xor_root(par[v]);
    }

    bool merge(int u, int v, int w) {
        int ru = find(u), rv = find(v);
        int xu = xor_root(u), xv = xor_root(v);

        if(ru == rv) {
            st.push({0, ru, -1, -1, -1, -1, basis[ru]});
            basis[ru].insert(xu ^ xv ^ w);
            return false;
        }

        if(rank[ru] < rank[rv]) swap(ru, rv), swap(xu, xv);

        st.push({1, ru, rv, rank[ru], rank[rv], xor_to_parent[rv], basis[ru]});
        par[rv] = ru;
        rank[ru] += rank[rv];
        xor_to_parent[rv] = xu ^ xv ^ w;
        basis[ru] = basis[ru] + basis[rv];
        return true;
    }

    void rollBack() {
        if(st.empty()) return;
        auto rec = st.top();
        st.pop();
        if(rec.type == 0) {
            basis[rec.u] = rec.basis_u;
        } else {
            par[rec.v] = rec.v;
            rank[rec.u] = rec.rank_u;
            rank[rec.v] = rec.rank_v;
            xor_to_parent[rec.v] = rec.xor_v_to_parent;
            basis[rec.u] = rec.basis_u;
        }
    }

    ll query(int u, int v) {
        int xu = xor_root(u), xv = xor_root(v);
        return basis[find(u)].min_value(xu ^ xv);
    }
};

template<typename T>
struct DynaCon { 
    int SZ;  
    Undo_DSU A;
    vt<vt<T>> seg;
    vll ans;
    DynaCon(int n, int dsuSize) : A(dsuSize) {
		SZ = 1;
        while(SZ < n) SZ <<= 1;
        seg.resize(SZ << 1);
        ans.rsz(SZ);
    }

    void update_range(int l, int r, T p) {  
        l += SZ, r += SZ + 1;
        while(l < r) {
            if(l & 1) seg[l++].pb(p);
            if(r & 1) seg[--r].pb(p);
            l >>= 1; r >>= 1;
        }
    }
    
    void process(int ind = 1) {
        int c = A.st.size();
        for(auto &[u, v, w] : seg[ind]) {
            A.merge(u, v, w);
        }
        if(ind >= SZ) {
            for(auto& [u, v] : G[ind - SZ]) {
                ans[ind - SZ] = A.query(u, v);
            }
        }
        else { process(2 * ind); process(2 * ind + 1); }
        while(int(A.st.size()) > c) {
            A.rollBack();
        }
    }
};

void solve() {
    int n, m; cin >> n >> m;
    map<pii, pii> mp;
    for(int i = 0; i < m; i++) {
        int u, v, w; cin >> u >> v >> w;
        u--, v--;
        mp[{min(u, v), max(u, v)}] = {0, w};
    }
    int q; cin >> q;
    DynaCon<ar(3)> graph(q + 5, n);
    vi ops(q + 1);
    for(int i = 1; i <= q; i++) {
        cin >> ops[i];
        int u, v; cin >> u >> v;
        u--, v--;
        if(u > v) swap(u, v);
        if(ops[i] == 1) {
            int w; cin >> w;
            mp[{u, v}] = {i, w};
        } else if(ops[i] == 2) {
            auto [id, w] = mp[{u, v}];
            graph.update_range(id, i, {u, v, w});
            mp.erase({u, v});
        } else {
            G[i].pb({u, v});
        }
    }
    for(auto& [x, y] : mp) {
        auto& [st, w] = y;
        auto& [u, v] = x;
        graph.update_range(st, q, {u, v, w});
    }
    graph.process();
    for(int i = 1; i <= q; i++) {
        if(ops[i] == 3) {
            cout << graph.ans[i] << '\n';
        }
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
